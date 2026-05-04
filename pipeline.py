import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode


# Dimensão fixa da grade a ser varrida no ortho final.
ROWS = 37
COLS = 37

# Caminho padrão do arquivo de configuração com a allowlist de símbolos.
DEFAULT_SYMBOLS_CONFIG = Path(__file__).resolve().parent / "symbols_config.json"


def load_symbols_config(path=None):
    # Carrega o arquivo JSON com o vocabulário (allowlist) de símbolos válidos.
    #
    # Retorna um dicionário com as chaves "vocabulary" (frozenset|None) e
    # "max_length" (int). Em caso de ausência ou erro de leitura, devolve um
    # config "permissivo" — qualquer letra ASCII isolada é aceita.
    #
    # Esse config é um artefato de produção: contém só o vocabulário do projeto.
    # Gabaritos de teste (multiset esperado para um lote específico de fotos)
    # vivem separados, em ECC200Decode/tests/expected_*.json, e não passam por
    # esta função.
    config_path = Path(path) if path else DEFAULT_SYMBOLS_CONFIG
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"vocabulary": None, "max_length": 4}

    vocab = raw.get("vocabulary")
    return {
        "vocabulary": frozenset(vocab) if vocab else None,
        "max_length": int(raw.get("max_length", 4)),
    }


# Vocabulário padrão lido uma única vez no carregamento do módulo.
SYMBOLS_CONFIG = load_symbols_config()


def parse_args():
    # Define os argumentos aceitos na linha de comando.
    p = argparse.ArgumentParser()

    # Arquivos principais de entrada e saída.
    p.add_argument("--template", default="template.png")
    p.add_argument("--input", default="imagem.png")
    p.add_argument("--output", default="ortho.png")
    p.add_argument(
        "--margin", type=int, default=None,
        help="Margem do ortho em pixels. Default: auto (ceil de metade do passo da grade).",
    )
    p.add_argument("--grid-output", default="grid.txt")

    # Se ativado, salva cada tile recortado da grade.
    p.add_argument("--dump-elements", action="store_true")
    p.add_argument("--elements-dir", default="elements")

    # Parâmetros do decoder / pré-processamento.
    # 60 ms é o equilíbrio observado experimentalmente: tiles saudáveis
    # decodificam em <30 ms, mas alguns símbolos limítrofes (com baixa
    # estrutura ou pequenos desalinhamentos) precisam de uma folga adicional
    # para serem decodificados de forma determinística entre execuções —
    # principalmente quando os workers estão competindo por CPU. Valores mais
    # baixos (30–40 ms) introduzem perdas não determinísticas em
    # configurações com paralelismo alto, conforme observado em testes
    # repetidos no mesmo conjunto.
    p.add_argument("--decode-timeout", type=int, default=60)
    p.add_argument("--decode-shrink", type=int, default=2)
    p.add_argument("--decode-border", type=int, default=10)
    p.add_argument("--resize-factor", type=float, default=2.0)

    # Parâmetros de paralelismo/performance.
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Número de processos paralelos para decodificar tiles",
    )
    # Chunks maiores reduzem o overhead de IPC do ProcessPoolExecutor quando
    # o número de células é grande (1369 = 37×37). 64 mantém os workers cheios
    # sem deixar o último worker ocioso por muito tempo.
    p.add_argument("--chunksize", type=int, default=64)

    # Atalhos opcionais para acelerar:
    # - skip-empty: pula tiles que parecem vazios (ligado por padrão)
    # - use-edge-bounds: restringe o tamanho esperado do símbolo no decoder
    p.add_argument("--skip-empty", dest="skip_empty", action="store_true", default=True)
    p.add_argument("--no-skip-empty", dest="skip_empty", action="store_false")
    p.add_argument("--empty-std-threshold", type=float, default=7.0)
    p.add_argument("--empty-dark-threshold", type=float, default=0.06)
    p.add_argument("--use-edge-bounds", action="store_true")

    # Alinhamento fino: move levemente cada caixa para encaixar no símbolo real.
    # Desativado por padrão: a homografia já costuma alinhar bem o grid, e o
    # deslocamento agressivo pode empurrar a caixa para fora de símbolos limítrofes.
    # Útil em imagens com homografia ruidosa — habilite com --refine-cells.
    p.add_argument("--refine-cells", dest="refine_cells", action="store_true", default=False)
    p.add_argument("--no-refine-cells", dest="refine_cells", action="store_false")
    p.add_argument(
        "--refine-max-shift",
        type=int,
        default=0,
        help="Deslocamento máximo em pixels para refinamento de célula (0 = automático)",
    )

    # Modo de teste para uma célula específica da grade.
    p.add_argument(
        "--test-cell",
        default=None,
        help="Testa apenas uma célula no formato linha,coluna, ex: 5,6",
    )
    p.add_argument("--test-cell-dir", default="debug_cell")

    return p.parse_args()


def order_corners(points):
    # Ordena 4 pontos no formato:
    # top-left, top-right, bottom-right, bottom-left
    #
    # Usa:
    # - soma x+y para achar TL e BR
    # - diferença x-y para achar TR e BL
    s = points.sum(axis=1)
    d = points[:, 0] - points[:, 1]

    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmax(d)]
    bl = points[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_four_matches(response, radius):
    # Encontra os 4 melhores picos no mapa de resposta do matchTemplate.
    #
    # A cada pico encontrado, "zera" uma vizinhança ao redor para evitar
    # pegar o mesmo canto repetidamente.
    work = response.copy()
    h, w = work.shape
    r = max(1, int(radius))
    matches = []

    for _ in range(4):
        _, _, _, loc = cv2.minMaxLoc(work)
        x, y = int(loc[0]), int(loc[1])
        matches.append([x, y])

        # Supressão local ao redor do pico encontrado.
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w - 1, x + r), min(h - 1, y + r)
        work[y0:y1 + 1, x0:x1 + 1] = -1.0

    return np.array(matches, dtype=np.float32)


def clamp_int(v, lo, hi):
    # Garante que um valor inteiro fique dentro de [lo, hi].
    return max(lo, min(hi, v))


def crop_box(image, x0, y0, x1, y1):
    # Recorta uma subimagem delimitada por caixa retangular.
    # Retorna None se a caixa for inválida.
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1].copy()


def add_white_border(img, border):
    # Adiciona uma borda branca ao redor da imagem.
    # Isso ajuda a criar uma "quiet zone" artificial para o DataMatrix.
    if border <= 0:
        return img

    return cv2.copyMakeBorder(
        img,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def try_decode_text(img, timeout, shrink, min_edge=None, max_edge=None):
    # Tenta decodificar um DataMatrix usando pylibdmtx.
    #
    # Parâmetros importantes:
    # - timeout: tempo máximo por tentativa
    # - shrink: fator interno para acelerar busca
    # - min_edge/max_edge: restringem tamanho esperado do símbolo
    #
    # Importante: min_edge e max_edge só são passados se não forem None.
    kwargs = {
        "timeout": timeout,
        "max_count": 1,
        "shrink": shrink,
    }

    if min_edge is not None:
        kwargs["min_edge"] = min_edge
    if max_edge is not None:
        kwargs["max_edge"] = max_edge

    results = decode(img, **kwargs)
    if results:
        return results[0].data.decode("utf-8", errors="replace")
    return None


def looks_like_valid_symbol(text, vocabulary=None, max_length=None):
    # Valida se o texto decodificado parece um código de símbolo real.
    #
    # Corrupções típicas do libdmtx em imagens rotacionadas incluem sequências
    # como "H63E", "l07E", "V\x02E", "P07\x05" — isto é, uma letra real seguida
    # de dígitos ou caracteres de controle. Um símbolo legítimo do TCC é sempre
    # curto, ASCII e composto apenas por letras.
    #
    # Quando um vocabulário (allowlist) é fornecido, exige-se também que o texto
    # esteja contido nele. Isso filtra leituras espúrias como "N", "I" e variantes
    # acentuadas que escapariam só pela checagem de isalpha().
    if vocabulary is None:
        vocabulary = SYMBOLS_CONFIG.get("vocabulary")
    if max_length is None:
        max_length = SYMBOLS_CONFIG.get("max_length", 4)

    if not text:
        return False
    if len(text) > max_length:
        return False
    if not (text.isascii() and text.isalpha()):
        return False
    if vocabulary is not None and text not in vocabulary:
        return False
    return True


def tile_looks_empty(gray, std_threshold, dark_threshold):
    # Heurística barata para decidir se um tile parece vazio.
    # Avalia primeiro o desvio padrão (early-exit) e só depois a razão de pixels escuros,
    # evitando o custo de um np.mean desnecessário na imensa maioria das células vazias.
    if gray.size < 16:
        return True
    std = float(gray.std())
    if std < std_threshold:
        return True
    dark_ratio = float(np.count_nonzero(gray < 180)) / float(gray.size)
    return dark_ratio < dark_threshold


def refine_tile_box(gray, x0, y0, x1, y1, max_shift):
    # Move a caixa nominal da grade até o componente escuro mais denso próximo ao centro.
    #
    # Útil quando a ortorretificação está quase, mas não perfeitamente, alinhada ao símbolo.
    # Mantém o tamanho da caixa original e apenas translada.
    if max_shift <= 0:
        return x0, y0, x1, y1

    h, w = gray.shape[:2]
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return x0, y0, x1, y1

    sx0 = max(0, x0 - max_shift)
    sy0 = max(0, y0 - max_shift)
    sx1 = min(w, x1 + max_shift)
    sy1 = min(h, y1 + max_shift)
    region = gray[sy0:sy1, sx0:sx1]
    if region.size == 0 or region.std() < 8.0:
        return x0, y0, x1, y1

    # Binarização inversa: símbolos escuros viram primeiro plano.
    _, mask = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return x0, y0, x1, y1

    ocx = (x0 + bw / 2.0) - sx0
    ocy = (y0 + bh / 2.0) - sy0

    box_area = float(bw * bh)
    min_area = box_area * 0.05
    max_area = box_area * 1.10
    max_dist = max_shift * 1.5

    best = None
    best_score = -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        dist = float(np.hypot(cx - ocx, cy - ocy))
        if dist > max_dist:
            continue
        score = float(area) - dist * 10.0
        if score > best_score:
            best_score = score
            best = (cx, cy)

    if best is None:
        return x0, y0, x1, y1

    cx_full = sx0 + best[0]
    cy_full = sy0 + best[1]
    nx0 = max(0, min(w - bw, int(round(cx_full - bw / 2.0))))
    ny0 = max(0, min(h - bh, int(round(cy_full - bh / 2.0))))
    return nx0, ny0, nx0 + bw, ny0 + bh


def build_candidates_and_bounds(tile_gray, border=10, resize_factor=2.0, use_edge_bounds=False):
    # Gera as imagens candidatas a serem passadas ao decoder.
    #
    # Ordem, do mais provável para o mais caro:
    # 1. otsu          — contraste realçado + Otsu (cobre a maioria dos casos).
    # 2. otsu_bil      — mesmo caminho mas sobre imagem bilaterizada (símbolos ruidosos).
    # 3. adaptive      — limiar adaptativo Gaussiano (iluminação não-uniforme).
    # 4. sharp         — imagem em cinza com sharpening (fallback analógico).
    # 5. gray          — tile original redimensionado (último recurso).
    #
    # Também calcula min_edge/max_edge quando solicitado.

    if tile_gray is None or tile_gray.size == 0:
        return [], None, None

    h_t, w_t = tile_gray.shape[:2]
    if h_t < 4 or w_t < 4:
        return [], None, None

    # Tiles muito pequenos se beneficiam de ampliação extra — o libdmtx precisa de
    # pelo menos alguns pixels por módulo.
    rf = float(resize_factor)
    if min(h_t, w_t) < 60:
        rf = max(rf, 3.0)

    gray_up = cv2.resize(
        tile_gray,
        None,
        fx=rf,
        fy=rf,
        interpolation=cv2.INTER_CUBIC,
    )

    # Realce local de contraste.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_up)

    # Sharpening leve.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # Otsu sobre a versão realçada.
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Caminho alternativo com preservação de bordas (bilateral) antes do Otsu.
    bilateral = cv2.bilateralFilter(gray_up, 5, 40, 40)
    _, otsu_bil = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Limiar adaptativo para iluminação heterogênea.
    block_size = max(11, (min(gray_up.shape[:2]) // 6) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray_up,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        4,
    )

    candidates = [
        ("otsu", add_white_border(otsu, border)),
        ("otsu_bil", add_white_border(otsu_bil, border)),
        ("adaptive", add_white_border(adaptive, border)),
        ("sharp", add_white_border(sharp, border)),
        ("gray", add_white_border(gray_up, border)),
    ]

    min_edge = None
    max_edge = None

    # Opcionalmente restringe o tamanho esperado do símbolo.
    if use_edge_bounds:
        inner_side = min(gray_up.shape[:2])
        min_edge = max(8, int(inner_side * 0.35))
        max_edge = max(min_edge + 1, int(inner_side * 0.98))

    return candidates, min_edge, max_edge


def decode_datamatrix_gray_with_method(
    tile_gray,
    timeout=40,
    shrink=2,
    border=10,
    resize_factor=2.0,
    skip_empty=False,
    empty_std_threshold=7.0,
    empty_dark_threshold=0.06,
    use_edge_bounds=False,
):
    # Decodifica um tile em grayscale e retorna:
    # - texto decodificado (vencedor do voto entre pré-processamentos)
    # - método cuja leitura foi escolhida ("otsu", "otsu_bil", "adaptive",
    #   "sharp" ou "gray", possivelmente sufixado com "_cNN" se veio do
    #   fallback de multi-crop)
    #
    # Caso não encontre nada, retorna (None, None).

    if tile_gray is None or tile_gray.size == 0:
        return None, None

    # Se ativado, pula tiles considerados vazios.
    if skip_empty and tile_looks_empty(tile_gray, empty_std_threshold, empty_dark_threshold):
        return None, None

    # Gera candidatos e, opcionalmente, limites de tamanho do símbolo.
    candidates, min_edge, max_edge = build_candidates_and_bounds(
        tile_gray,
        border=border,
        resize_factor=resize_factor,
        use_edge_bounds=use_edge_bounds,
    )

    # Voto entre os pré-processamentos: roda os dois primeiros (otsu, otsu_bil),
    # e se concordarem em uma leitura válida, aceita imediatamente (caminho
    # rápido — cobre a maioria dos tiles bem-comportados). Se discordarem, ou
    # se ambos falharem, roda o resto e desempata por maioria. Cascade-only
    # falhava em tiles onde o caminho otsu (CLAHE+sharpen) flipa bits dentro
    # do símbolo por causa do overshoot do sharpen — vide caso do DataMatrix
    # colado no furo da base que retornava "G" quando a leitura correta era "H".
    results = {}
    for method, img in candidates[:2]:
        text = try_decode_text(
            img, timeout=timeout, shrink=shrink,
            min_edge=min_edge, max_edge=max_edge,
        )
        if text is not None and looks_like_valid_symbol(text):
            results[method] = text

    distinct = set(results.values())
    if len(distinct) == 1 and len(results) == 2:
        winner = next(iter(distinct))
        winning_method = next(m for m, _ in candidates if results.get(m) == winner)
        return winner, winning_method

    # Discordância (ou ambos None): roda o restante pra ter votos suficientes.
    for method, img in candidates[2:]:
        text = try_decode_text(
            img, timeout=timeout, shrink=shrink,
            min_edge=min_edge, max_edge=max_edge,
        )
        if text is not None and looks_like_valid_symbol(text):
            results[method] = text

    if results:
        counts = Counter(results.values())
        max_count = max(counts.values())
        top = [t for t, c in counts.items() if c == max_count]
        if len(top) == 1:
            winner = top[0]
        else:
            # Empate: desempata pela ordem do cascade (otsu primeiro).
            winner = next(
                results[m] for m, _ in candidates if results.get(m) in top
            )
        winning_method = next(
            m for m, _ in candidates if results.get(m) == winner
        )
        return winner, winning_method

    # Fallback legítimo: variação do tamanho do crop para células cujo
    # pré-processamento canônico falhou mas que parecem ter conteúdo real.
    # NÃO rotacionamos o tile — apenas variamos o tamanho da janela ao redor
    # do mesmo conteúdo.
    tile_std = float(tile_gray.std())
    if tile_std < 18.0:
        return None, None

    h, w = tile_gray.shape[:2]
    if min(h, w) < 16:
        return None, None

    for shrink_ratio in (0.75, 0.65, 0.55):
        nh = int(round(h * shrink_ratio))
        nw = int(round(w * shrink_ratio))
        if nh < 12 or nw < 12 or nh >= h or nw >= w:
            continue
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        cropped = tile_gray[y0:y0 + nh, x0:x0 + nw]
        if cropped.size == 0:
            continue
        sub_cands, mn, mx = build_candidates_and_bounds(
            cropped, border=border, resize_factor=resize_factor,
            use_edge_bounds=use_edge_bounds,
        )
        for method, img in sub_cands:
            text = try_decode_text(
                img, timeout=timeout, shrink=shrink,
                min_edge=mn, max_edge=mx,
            )
            if text is not None and looks_like_valid_symbol(text):
                return text, f"{method}_c{int(shrink_ratio * 100)}"

    return None, None


def build_ortho(image_bgr, template_bgr, margin=None):
    # Faz a ortorretificação da imagem:
    # 1. encontra 4 ocorrências do template
    # 2. calcula seus centros
    # 3. ordena os 4 cantos
    # 4. aplica warpPerspective para obter imagem "reta"
    #
    # Se margin=None (padrão), a margem é calculada automaticamente como
    # ceil(metade do menor passo da grade). Como as cruzes de canto estão
    # centralizadas nos furos dos cantos da grade igualmente espaçada, essa
    # margem garante que todas as células — inclusive as de canto — tenham
    # recorte de tamanho nominal sem truncamento pela borda da imagem.

    import math

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray_template.shape[:2]

    # Procura o template na imagem.
    response = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    matches = find_four_matches(response, radius=min(tw, th) * 0.6)

    # Converte top-left dos matches em centros do template.
    centers = matches + np.array([tw / 2.0, th / 2.0], np.float32)
    corners = order_corners(centers)

    tl, tr, br, bl = corners

    # Calcula largura e altura do retângulo destino.
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(1, width)
    height = max(1, height)

    # Margem automática: ceil(metade do menor passo da grade 37×37).
    if margin is None:
        step = min(width / (COLS - 1), height / (ROWS - 1))
        margin = int(math.ceil(step / 2.0))
    m = max(0, int(margin))

    # Define o retângulo destino já com margem embutida.
    destination = np.array(
        [[m, m], [m + width, m], [m + width, m + height], [m, m + height]],
        dtype=np.float32,
    )

    # Calcula homografia e gera imagem ortorretificada.
    H = cv2.getPerspectiveTransform(corners, destination)
    ortho = cv2.warpPerspective(image_bgr, H, (width + 2 * m, height + 2 * m))
    return ortho, m


def compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS):
    # Calcula as caixas retangulares de cada célula da grade sobre a imagem ortho.
    #
    # A grade é distribuída uniformemente entre as margens. Como a base
    # perfurada tem espaçamento perfeitamente regular, o cálculo nominal
    # (half_side = metade do menor passo) já produz caixas que contêm o
    # símbolo centralizado por inteiro.
    m = float(max(0, int(margin)))
    step_x = (width - 2.0 * m) / (cols - 1)
    step_y = (height - 2.0 * m) / (rows - 1)
    half_side = min(step_x, step_y) / 2.0

    boxes = []
    for r in range(rows):
        cy = m + r * step_y
        for c in range(cols):
            cx = m + c * step_x

            x0 = clamp_int(int(round(cx - half_side)), 0, width - 1)
            y0 = clamp_int(int(round(cy - half_side)), 0, height - 1)
            x1 = clamp_int(int(round(cx + half_side)), 0, width)
            y1 = clamp_int(int(round(cy + half_side)), 0, height)

            boxes.append((r, c, x0, y0, x1, y1))

    return boxes


def _decode_worker(job):
    # Worker executado em cada processo paralelo.
    (
        r, c, tile_gray,
        timeout, shrink, border, resize_factor,
        skip_empty, empty_std_threshold, empty_dark_threshold,
        use_edge_bounds,
    ) = job

    text, _method = decode_datamatrix_gray_with_method(
        tile_gray,
        timeout=timeout,
        shrink=shrink,
        border=border,
        resize_factor=resize_factor,
        skip_empty=skip_empty,
        empty_std_threshold=empty_std_threshold,
        empty_dark_threshold=empty_dark_threshold,
        use_edge_bounds=use_edge_bounds,
    )

    return r, c, text if text is not None else "?"


def decode_grid(
    ortho_bgr,
    margin,
    dump_elements,
    elements_dir,
    decode_timeout,
    decode_shrink,
    decode_border,
    resize_factor,
    workers,
    chunksize,
    skip_empty,
    empty_std_threshold,
    empty_dark_threshold,
    use_edge_bounds,
    refine_cells=True,
    refine_max_shift=0,
):
    # Decodifica toda a grade 37x37 da imagem ortorretificada.
    #
    # Quando a margem do ortho foi calculada automaticamente por build_ortho
    # (ceil de metade do passo da grade), nenhuma célula tem seu recorte
    # truncado pela borda — todas têm exatamente o tamanho nominal com o
    # símbolo centralizado. Isso elimina a necessidade de padding auxiliar
    # e de votação de borda.

    height, width = ortho_bgr.shape[:2]
    ortho_gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    boxes = compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS)

    if refine_cells and refine_max_shift <= 0 and boxes:
        _, _, bx0, by0, bx1, by1 = boxes[0]
        cell_side = max(1, min(bx1 - bx0, by1 - by0))
        refine_max_shift = max(4, int(round(cell_side * 0.30)))

    out_dir = Path(elements_dir)
    if dump_elements:
        out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for r, c, x0, y0, x1, y1 in boxes:
        if refine_cells and refine_max_shift > 0:
            x0, y0, x1, y1 = refine_tile_box(
                ortho_gray, x0, y0, x1, y1, refine_max_shift,
            )

        tile_gray = crop_box(ortho_gray, x0, y0, x1, y1)

        if dump_elements:
            tile_bgr = crop_box(ortho_bgr, x0, y0, x1, y1)
            if tile_bgr is not None:
                cv2.imwrite(str(out_dir / f"r{r}c{c}.png"), tile_bgr)

        jobs.append(
            (
                r, c, tile_gray,
                decode_timeout, decode_shrink, decode_border,
                resize_factor, skip_empty,
                empty_std_threshold, empty_dark_threshold,
                use_edge_bounds,
            )
        )

    # Inicializa a grade de saída com "?".
    grid = [["?"] * COLS for _ in range(ROWS)]

    # Modo serial.
    if workers <= 1:
        for job in jobs:
            r, c, value = _decode_worker(job)
            grid[r][c] = value

    # Modo paralelo com múltiplos processos. Cada worker roda em processo
    # separado, o que isola o estado interno do libdmtx — que não é
    # projetado para acesso concorrente por threads.
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r, c, value in ex.map(_decode_worker, jobs, chunksize=chunksize):
                grid[r][c] = value

    # A foto e capturada pela face inferior da maquete, entao o eixo
    # horizontal sai espelhado em relacao a vista de topo real. Espelhamos
    # as colunas antes de serializar para que o grid.txt fique em frame
    # de topo (mesmo frame que a saida JSON do PyAppArq).
    grid = [row[::-1] for row in grid]

    # Converte a matriz para string final do grid.txt.
    return "\n".join(" ".join(row) for row in grid)


def parse_test_cell(value):
    # Faz o parse de --test-cell no formato "linha,coluna".
    try:
        left, right = value.split(",", 1)
        r = int(left.strip())
        c = int(right.strip())
    except Exception as e:
        raise ValueError("--test-cell deve estar no formato linha,coluna, ex: 5,6") from e

    # Valida se a célula está dentro da grade.
    if not (0 <= r < ROWS and 0 <= c < COLS):
        raise ValueError(f"--test-cell fora do intervalo válido: 0..{ROWS - 1}, 0..{COLS - 1}")

    return r, c


def run_test_cell(ortho_bgr, margin, args):
    # Executa o pipeline apenas para uma célula específica,
    # salvando os candidatos e imprimindo os resultados por método.

    r, c = parse_test_cell(args.test_cell)

    height, width = ortho_bgr.shape[:2]
    ortho_gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    boxes = compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS)

    # Localiza a caixa correspondente à célula desejada.
    box = next(b for b in boxes if b[0] == r and b[1] == c)
    _, _, x0, y0, x1, y1 = box

    tile_bgr = crop_box(ortho_bgr, x0, y0, x1, y1)
    tile_gray = crop_box(ortho_gray, x0, y0, x1, y1)

    out_dir = Path(args.test_cell_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Salva o tile original.
    if tile_bgr is not None:
        cv2.imwrite(str(out_dir / f"r{r}c{c}.png"), tile_bgr)

    if tile_gray is None or tile_gray.size == 0:
        print(f"r{r}c{c} => None")
        return

    # Se skip-empty estiver ligado e o tile parecer vazio, para cedo.
    if args.skip_empty and tile_looks_empty(tile_gray, args.empty_std_threshold, args.empty_dark_threshold):
        print(f"r{r}c{c} => None (skip-empty)")
        return

    # Gera candidatos para teste.
    candidates, min_edge, max_edge = build_candidates_and_bounds(
        tile_gray,
        border=args.decode_border,
        resize_factor=args.resize_factor,
        use_edge_bounds=args.use_edge_bounds,
    )

    winner = None
    for method, img in candidates:
        # Salva cada candidato para inspeção visual.
        cv2.imwrite(str(out_dir / f"r{r}c{c}_{method}.png"), img)

        # Tenta decodificar o candidato atual.
        text = try_decode_text(
            img,
            timeout=args.decode_timeout,
            shrink=args.decode_shrink,
            min_edge=min_edge,
            max_edge=max_edge,
        )

        print(f"r{r}c{c} [{method}]: {repr(text)}")

        # Guarda o primeiro método vencedor.
        if winner is None and text is not None:
            winner = text

    print(f"r{r}c{c} => {repr(winner)}")


def main():
    # Função principal do programa.
    a = parse_args()

    # Abre imagem de entrada e template.
    image = cv2.imread(a.input, cv2.IMREAD_COLOR)
    template = cv2.imread(a.template, cv2.IMREAD_COLOR)
    if image is None or template is None:
        raise RuntimeError("Falha ao abrir input/template")

    # Gera imagem ortorretificada.
    ortho, margin = build_ortho(image, template, a.margin)

    # Salva ortho para inspeção.
    if not cv2.imwrite(a.output, ortho):
        raise RuntimeError("Falha ao salvar ortho")

    # Se foi pedido teste de uma célula específica, executa só esse caminho.
    if a.test_cell is not None:
        run_test_cell(ortho, margin, a)
        return

    # Caso contrário, decodifica a grade inteira.
    grid_text = decode_grid(
        ortho_bgr=ortho,
        margin=margin,
        dump_elements=a.dump_elements,
        elements_dir=a.elements_dir,
        decode_timeout=a.decode_timeout,
        decode_shrink=a.decode_shrink,
        decode_border=a.decode_border,
        resize_factor=a.resize_factor,
        workers=a.workers,
        chunksize=a.chunksize,
        skip_empty=a.skip_empty,
        empty_std_threshold=a.empty_std_threshold,
        empty_dark_threshold=a.empty_dark_threshold,
        use_edge_bounds=a.use_edge_bounds,
        refine_cells=a.refine_cells,
        refine_max_shift=a.refine_max_shift,
    )

    # Salva a grade textual final.
    Path(a.grid_output).write_text(grid_text, encoding="utf-8")


if __name__ == "__main__":
    main()