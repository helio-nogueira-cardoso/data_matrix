"""
pipeline.py - Ortorretificacao e decodificacao DataMatrix em grade.

Adaptado de ECC200Decode/pipeline.py para funcionar como modulo de biblioteca.
Usa a mesma ortorretificacao por template matching e decodificacao DataMatrix
com multiplos pre-processamentos, mas retorna dados estruturados em vez de
escrever arquivos.
"""

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode


# Dimensao fixa da grade.
ROWS = 37
COLS = 37


# ---------------------------------------------------------------------------
# Vocabulario de simbolos validos (allowlist) carregado do config
# ---------------------------------------------------------------------------

def _find_symbols_config():
    """Procura symbols_config.json em locais usuais.

    Tenta primeiro o diretorio deste arquivo, depois sobe um nivel para
    ECC200Decode/. Devolve None se nenhum encontrado.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "symbols_config.json", here.parent / "symbols_config.json"):
        if cand.exists():
            return cand
    return None


def load_symbols_config(path=None):
    """Carrega o JSON de allowlist de simbolos.

    Retorna dict com vocabulary (frozenset|None) e max_length (int). Em caso
    de erro de leitura devolve um config permissivo (qualquer letra ASCII).

    Esse config e um artefato de producao: contem so o vocabulario do projeto.
    Gabaritos de teste (multiset esperado para um lote especifico de fotos)
    vivem separados, em ECC200Decode/tests/expected_*.json, e nao passam por
    esta funcao.
    """
    config_path = Path(path) if path else _find_symbols_config()
    if config_path is None:
        return {"vocabulary": None, "max_length": 4}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"vocabulary": None, "max_length": 4}
    vocab = raw.get("vocabulary")
    return {
        "vocabulary": frozenset(vocab) if vocab else None,
        "max_length": int(raw.get("max_length", 4)),
    }


SYMBOLS_CONFIG = load_symbols_config()


# ---------------------------------------------------------------------------
# Auxiliares de ortorretificacao
# ---------------------------------------------------------------------------

def order_corners(points):
    """Ordena 4 pontos como: sup-esq, sup-dir, inf-dir, inf-esq."""
    s = points.sum(axis=1)
    d = points[:, 0] - points[:, 1]
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmax(d)]
    bl = points[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class OrthoError(RuntimeError):
    """Levantada quando a ortorretificacao falha."""
    pass


def find_four_matches(response, radius, min_score=0.3):
    """Encontra os 4 melhores picos no mapa de resposta do matchTemplate.

    Levanta OrthoError se menos de 4 picos excedem min_score.
    """
    work = response.copy()
    h, w = work.shape
    r = max(1, int(radius))
    matches = []
    scores = []

    for i in range(4):
        _, max_val, _, loc = cv2.minMaxLoc(work)

        if max_val < min_score:
            raise OrthoError(
                f"Apenas {i} dos 4 marcadores foram encontrados na imagem "
                f"(correlacao do {i+1}o pico: {max_val:.3f}, "
                f"minimo exigido: {min_score:.3f}). "
                f"Verifique se os 4 marcadores em cruz estao visiveis."
            )

        x, y = int(loc[0]), int(loc[1])
        matches.append([x, y])
        scores.append(max_val)

        # Supressao local ao redor do pico encontrado.
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w - 1, x + r), min(h - 1, y + r)
        work[y0:y1 + 1, x0:x1 + 1] = -1.0

    return np.array(matches, dtype=np.float32), scores


def _validate_corners(corners, img_shape):
    """Verifica se os 4 cantos formam um quadrilatero razoavel.

    Levanta OrthoError se os cantos sao degenerados.
    """
    img_h, img_w = img_shape[:2]
    img_diag = np.sqrt(img_w**2 + img_h**2)

    # Verifica distancia minima entre quaisquer dois cantos.
    min_dist_threshold = img_diag * 0.05  # pelo menos 5% da diagonal
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(corners[i] - corners[j])
            if dist < min_dist_threshold:
                raise OrthoError(
                    f"Marcadores {i+1} e {j+1} estao muito proximos "
                    f"({dist:.0f} px, minimo: {min_dist_threshold:.0f} px). "
                    f"Os 4 marcadores podem nao ter sido detectados corretamente."
                )

    # Verifica se o quadrilatero tem area positiva (formula do cadarco).
    area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0

    min_area = (img_diag * 0.05) ** 2
    if area < min_area:
        raise OrthoError(
            f"Os 4 marcadores detectados formam uma area degenerada "
            f"({area:.0f} px², minimo: {min_area:.0f} px²). "
            f"Verifique se a imagem contem a maquete completa."
        )

    # Verifica se as dimensoes resultantes sao razoaveis.
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    aspect = max(width, height) / max(1, min(width, height))

    if aspect > 5.0:
        raise OrthoError(
            f"Os marcadores formam um retangulo muito distorcido "
            f"(proporcao {aspect:.1f}:1). "
            f"Verifique se os 4 marcadores em cruz estao visiveis."
        )


def build_ortho(image_bgr, template_bgr, margin=None):
    """Ortorretifica usando template matching nos 4 cantos + warp perspectivo.

    Se margin=None (padrao), a margem e calculada automaticamente como
    ceil(metade do menor passo da grade 37x37). Isso garante que todas as
    celulas — inclusive as de canto — tenham recorte de tamanho nominal
    sem truncamento pela borda da imagem.

    Levanta OrthoError se os cantos nao podem ser encontrados ou sao invalidos.
    """
    import math

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray_template.shape[:2]

    response = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    matches, scores = find_four_matches(response, radius=min(tw, th) * 0.6)
    centers = matches + np.array([tw / 2.0, th / 2.0], np.float32)
    corners = order_corners(centers)

    _validate_corners(corners, image_bgr.shape)

    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(1, width)
    height = max(1, height)

    if margin is None:
        step = min(width / (COLS - 1), height / (ROWS - 1))
        margin = int(math.ceil(step / 2.0))
    m = max(0, int(margin))

    destination = np.array(
        [[m, m], [m + width, m], [m + width, m + height], [m, m + height]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, destination)
    ortho = cv2.warpPerspective(image_bgr, H, (width + 2 * m, height + 2 * m))
    return ortho, m


# ---------------------------------------------------------------------------
# Computacao da grade
# ---------------------------------------------------------------------------

def clamp_int(v, lo, hi):
    """Garante que um valor inteiro fique dentro de [lo, hi]."""
    return max(lo, min(hi, v))


def compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS):
    """Calcula as caixas retangulares de cada celula da grade 37x37.

    A grade e distribuida uniformemente entre as margens. Como a base
    perfurada tem espacamento perfeitamente regular, o calculo nominal
    (half_side = metade do menor passo) ja produz caixas que contem o
    simbolo centralizado por inteiro.
    """
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


def crop_box(image, x0, y0, x1, y1):
    """Recorta uma subimagem. Retorna None se a caixa for invalida."""
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1].copy()


# ---------------------------------------------------------------------------
# Decodificacao DataMatrix
# ---------------------------------------------------------------------------

def add_white_border(img, border):
    """Adiciona borda branca para criar quiet zone artificial."""
    if border <= 0:
        return img
    return cv2.copyMakeBorder(
        img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )


def try_decode_text(img, timeout, shrink, min_edge=None, max_edge=None):
    """Tenta decodificar um DataMatrix usando pylibdmtx."""
    kwargs = {"timeout": timeout, "max_count": 1, "shrink": shrink}
    if min_edge is not None:
        kwargs["min_edge"] = min_edge
    if max_edge is not None:
        kwargs["max_edge"] = max_edge
    results = decode(img, **kwargs)
    if results:
        return results[0].data.decode("utf-8", errors="replace")
    return None


def looks_like_valid_symbol(text, vocabulary=None, max_length=None):
    """Valida se o texto decodificado parece um codigo de simbolo real.

    Rejeita corrupcoes tipicas do libdmtx em imagens rotacionadas (sequencias
    como "H63E", "l07E", "V\\x02E", "P07\\x05" — letra real seguida de digitos
    ou caracteres de controle). Simbolos legitimos sao curtos, ASCII e alfa.

    Quando um vocabulario (allowlist) e fornecido, exige tambem que o texto
    esteja contido nele. Isso filtra leituras espurias como "N", "I" e outras
    letras que escapariam apenas pela checagem de isalpha().
    """
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


def tile_looks_empty(gray, std_threshold=7.0, dark_threshold=0.06):
    """Heuristica para decidir se um tile parece vazio.

    Faz early-exit no desvio padrao, evitando calcular dark_ratio na grande
    maioria das celulas vazias (fundo uniforme). Em seguida valida a razao
    de pixels escuros como segunda condicao.
    """
    if gray.size < 16:
        return True
    std = float(gray.std())
    if std < std_threshold:
        return True
    dark_ratio = float(np.count_nonzero(gray < 180)) / float(gray.size)
    return dark_ratio < dark_threshold


def refine_tile_box(gray, x0, y0, x1, y1, max_shift):
    """Move a caixa para o componente escuro mais denso proximo ao centro.

    Util quando a ortorretificacao esta quase, mas nao perfeitamente, alinhada
    ao simbolo. Mantem o tamanho da caixa original e so translada.
    Retorna a caixa original se nao houver candidato plausivel.
    """
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


def build_candidates_and_bounds(tile_gray, border=10, resize_factor=2.0,
                                use_edge_bounds=False):
    """Gera imagens candidatas para o decodificador.

    Ordem, do mais provavel para o mais caro:
      1. otsu       - contraste realcado + Otsu (caminho padrao).
      2. otsu_bil   - Otsu apos filtro bilateral (simbolos com ruido).
      3. adaptive   - limiar adaptativo Gaussiano (iluminacao nao uniforme).
      4. sharp      - versao cinza com sharpening.
      5. gray       - tile original so redimensionado (ultimo recurso).
    """
    if tile_gray is None or tile_gray.size == 0:
        return [], None, None

    h_t, w_t = tile_gray.shape[:2]
    if h_t < 4 or w_t < 4:
        return [], None, None

    # Tiles muito pequenos se beneficiam de ampliacao extra.
    rf = float(resize_factor)
    if min(h_t, w_t) < 60:
        rf = max(rf, 3.0)

    gray_up = cv2.resize(
        tile_gray, None,
        fx=rf, fy=rf,
        interpolation=cv2.INTER_CUBIC,
    )

    # Realce de contraste local.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_up)

    # Sharpening leve.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # Otsu sobre a versao realcada (caminho campeao).
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Caminho alternativo: bilateral preserva bordas e suprime ruido.
    bilateral = cv2.bilateralFilter(gray_up, 5, 40, 40)
    _, otsu_bil = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Limiar adaptativo para iluminacao heterogenea.
    block_size = max(11, (min(gray_up.shape[:2]) // 6) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray_up, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, 4,
    )

    candidates = [
        ("otsu", add_white_border(otsu, border)),
        ("otsu_bil", add_white_border(otsu_bil, border)),
        ("adaptive", add_white_border(adaptive, border)),
        ("sharp", add_white_border(sharp, border)),
        ("gray", add_white_border(gray_up, border)),
    ]

    min_edge = max_edge = None
    if use_edge_bounds:
        inner_side = min(gray_up.shape[:2])
        min_edge = max(8, int(inner_side * 0.35))
        max_edge = max(min_edge + 1, int(inner_side * 0.98))

    return candidates, min_edge, max_edge


def decode_datamatrix_gray(tile_gray, timeout=40, shrink=2, border=10,
                           resize_factor=2.0, skip_empty=False,
                           use_edge_bounds=False):
    """Decodifica um tile em grayscale. Retorna (texto, metodo) ou (None, None).

    Voto entre os pre-processamentos: roda os dois primeiros (otsu, otsu_bil)
    e, se concordam em uma leitura valida, aceita imediatamente (caminho
    rapido). Se discordam ou ambos falham, roda o resto e desempata por
    maioria, com a ordem do cascade como tiebreaker. Cascade-only falhava em
    tiles onde o caminho otsu (CLAHE+sharpen) flipa bits dentro do simbolo
    por causa do overshoot do sharpen — vide DataMatrix colado no furo da
    base que retornava "G" quando a leitura correta era "H".
    """
    if tile_gray is None or tile_gray.size == 0:
        return None, None
    if skip_empty and tile_looks_empty(tile_gray):
        return None, None

    candidates, min_edge, max_edge = build_candidates_and_bounds(
        tile_gray, border=border, resize_factor=resize_factor,
        use_edge_bounds=use_edge_bounds,
    )

    results = {}
    for method, img in candidates[:2]:
        text = try_decode_text(img, timeout=timeout, shrink=shrink,
                               min_edge=min_edge, max_edge=max_edge)
        if text is not None and looks_like_valid_symbol(text):
            results[method] = text

    distinct = set(results.values())
    if len(distinct) == 1 and len(results) == 2:
        winner = next(iter(distinct))
        winning_method = next(m for m, _ in candidates if results.get(m) == winner)
        return winner, winning_method

    # Discordancia (ou ambos None): roda o restante pra ter votos suficientes.
    for method, img in candidates[2:]:
        text = try_decode_text(img, timeout=timeout, shrink=shrink,
                               min_edge=min_edge, max_edge=max_edge)
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

    # Fallback legitimo: simbolos colocados manualmente na maquete podem cair
    # ligeiramente fora do centro da celula. Recortar o centro da celula com
    # um pouco menos de margem reduz a vizinhanca ruidosa e as vezes faz o
    # decoder achar o simbolo. Variamos apenas o tamanho da janela ao redor
    # do mesmo conteudo — sem rotacao.
    tile_std = float(tile_gray.std())
    if tile_std < 18.0:
        return None, None

    h, w = tile_gray.shape[:2]
    if min(h, w) < 16:
        return None, None

    # Tres niveis de crop central — 75%, 65% e 55% do lado original — cobrem
    # misalinhamentos suaves, medios e fortes sem explodir o custo.
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
            text = try_decode_text(img, timeout=timeout, shrink=shrink,
                                   min_edge=mn, max_edge=mx)
            if text is not None and looks_like_valid_symbol(text):
                return text, f"{method}_c{int(shrink_ratio * 100)}"

    return None, None


# ---------------------------------------------------------------------------
# Decodificacao paralela da grade
# ---------------------------------------------------------------------------

def _decode_worker(job):
    """Worker executado em cada processo paralelo."""
    r, c, tile_gray, timeout, shrink, border, resize_factor, skip_empty, use_edge_bounds = job
    text, _method = decode_datamatrix_gray(
        tile_gray, timeout=timeout, shrink=shrink, border=border,
        resize_factor=resize_factor, skip_empty=skip_empty,
        use_edge_bounds=use_edge_bounds,
    )
    return r, c, text if text is not None else "?"


def decode_grid(ortho_bgr, margin, decode_timeout=60, decode_shrink=2,
                decode_border=10, resize_factor=2.0, workers=None,
                chunksize=16, skip_empty=True, use_edge_bounds=False,
                refine_cells=False, refine_max_shift=0):
    """Decodifica a grade 37x37 completa. Retorna lista 2D de simbolos."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    height, width = ortho_bgr.shape[:2]
    ortho_gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    boxes = compute_grid_boxes(height, width, margin)

    if refine_cells and refine_max_shift <= 0 and boxes:
        _, _, bx0, by0, bx1, by1 = boxes[0]
        cell_side = max(1, min(bx1 - bx0, by1 - by0))
        refine_max_shift = max(4, int(round(cell_side * 0.30)))

    jobs = []
    for r, c, x0, y0, x1, y1 in boxes:
        if refine_cells and refine_max_shift > 0:
            x0, y0, x1, y1 = refine_tile_box(
                ortho_gray, x0, y0, x1, y1, refine_max_shift,
            )
        tile_gray = crop_box(ortho_gray, x0, y0, x1, y1)
        jobs.append((r, c, tile_gray, decode_timeout, decode_shrink,
                     decode_border, resize_factor, skip_empty, use_edge_bounds))

    # Inicializa a grade com "?".
    grid = [["?"] * COLS for _ in range(ROWS)]

    if workers <= 1:
        for job in jobs:
            r, c, value = _decode_worker(job)
            grid[r][c] = value
    else:
        # Processos em vez de threads: o libdmtx mantem estado interno nao
        # projetado para acesso concorrente por threads.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r, c, value in ex.map(_decode_worker, jobs, chunksize=chunksize):
                grid[r][c] = value

    return grid


# ---------------------------------------------------------------------------
# API de alto nivel
# ---------------------------------------------------------------------------

def process_image(image_path, template_path, margin=None, decode_timeout=60,
                  decode_shrink=2, decode_border=10, resize_factor=2.0,
                  workers=None, chunksize=16, skip_empty=True,
                  use_edge_bounds=False, refine_cells=False,
                  refine_max_shift=0, progress_callback=None):
    """Executa o pipeline completo sobre uma imagem.

    Args:
        image_path: Caminho da imagem de entrada.
        template_path: Caminho do template de marcadores de canto.
        margin: Margem da ortorretificacao em pixels.
        progress_callback: Funcao opcional(mensagem_str) para atualizacoes.

    Returns:
        ortho_bgr: Imagem ortorretificada (array numpy BGR).
        grid: Lista 37x37 de simbolos decodificados ("?" para falhas).
        boxes: Lista de (linha, coluna, x0, y0, x1, y1) das celulas.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    _progress("Carregando imagens...")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Falha ao abrir imagem: {image_path}")
    if template is None:
        raise RuntimeError(f"Falha ao abrir template: {template_path}")

    _progress("Ortorretificando...")
    ortho, margin = build_ortho(image, template, margin)

    _progress("Decodificando grade 37x37...")
    grid = decode_grid(
        ortho, margin,
        decode_timeout=decode_timeout,
        decode_shrink=decode_shrink,
        decode_border=decode_border,
        resize_factor=resize_factor,
        workers=workers,
        chunksize=chunksize,
        skip_empty=skip_empty,
        use_edge_bounds=use_edge_bounds,
        refine_cells=refine_cells,
        refine_max_shift=refine_max_shift,
    )

    height, width = ortho.shape[:2]
    boxes = compute_grid_boxes(height, width, margin)

    decoded_count = sum(1 for row in grid for s in row if s != "?")
    _progress(f"Concluido - {decoded_count} simbolos decodificados.")

    return ortho, grid, boxes
