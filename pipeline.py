import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode


# Dimensão fixa da grade a ser varrida no ortho final.
ROWS = 37
COLS = 37


def parse_args():
    # Define os argumentos aceitos na linha de comando.
    p = argparse.ArgumentParser()

    # Arquivos principais de entrada e saída.
    p.add_argument("--template", default="template.png")
    p.add_argument("--input", default="imagem.png")
    p.add_argument("--output", default="ortho.png")
    p.add_argument("--margin", type=int, default=60)
    p.add_argument("--grid-output", default="grid.txt")

    # Se ativado, salva cada tile recortado da grade.
    p.add_argument("--dump-elements", action="store_true")
    p.add_argument("--elements-dir", default="elements")

    # Parâmetros do decoder / pré-processamento.
    p.add_argument("--decode-timeout", type=int, default=40)
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
    p.add_argument("--chunksize", type=int, default=32)

    # Atalhos opcionais para acelerar:
    # - skip-empty: tenta pular tiles que parecem vazios
    # - use-edge-bounds: restringe o tamanho esperado do símbolo no decoder
    p.add_argument("--skip-empty", action="store_true")
    p.add_argument("--empty-std-threshold", type=float, default=7.0)
    p.add_argument("--empty-dark-threshold", type=float, default=0.06)
    p.add_argument("--use-edge-bounds", action="store_true")

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


def tile_looks_empty(gray, std_threshold, dark_threshold):
    # Heurística barata para decidir se um tile parece vazio.
    #
    # std: desvio-padrão dos níveis de cinza
    # dark_ratio: proporção de pixels relativamente escuros
    #
    # Se quase não há contraste OU quase não há pixels escuros,
    # o tile é tratado como "provavelmente vazio".
    std = float(gray.std())
    dark_ratio = float(np.mean(gray < 180))
    return std < std_threshold or dark_ratio < dark_threshold


def build_candidates_and_bounds(tile_gray, border=10, resize_factor=2.0, use_edge_bounds=False):
    # Gera as imagens candidatas a serem passadas ao decoder.
    #
    # Ordem de prioridade:
    # 1. otsu
    # 2. sharp
    # 3. gray
    #
    # Também calcula min_edge/max_edge quando solicitado.

    # Ampliação do tile para facilitar a leitura do símbolo.
    gray_up = cv2.resize(
        tile_gray,
        None,
        fx=resize_factor,
        fy=resize_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # Melhora contraste local.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_up)

    # Aplica sharpening leve.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # Binarização automática com Otsu.
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Candidatos em ordem de tentativa.
    candidates = [
        ("otsu", add_white_border(otsu, border)),
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
    # - texto decodificado
    # - método vencedor ("otsu", "sharp" ou "gray")
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

    # Testa cada candidato na ordem definida.
    for method, img in candidates:
        text = try_decode_text(
            img,
            timeout=timeout,
            shrink=shrink,
            min_edge=min_edge,
            max_edge=max_edge,
        )
        if text is not None:
            return text, method

    return None, None


def build_ortho(image_bgr, template_bgr, margin):
    # Faz a ortorretificação da imagem:
    # 1. encontra 4 ocorrências do template
    # 2. calcula seus centros
    # 3. ordena os 4 cantos
    # 4. aplica warpPerspective para obter imagem "reta"

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray_template.shape[:2]

    # Procura o template na imagem.
    response = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    matches = find_four_matches(response, radius=min(tw, th) * 0.6)

    # Converte top-left dos matches em centros do template.
    centers = matches + np.array([tw / 2.0, th / 2.0], np.float32)
    corners = order_corners(centers)

    m = max(0, int(margin))
    tl, tr, br, bl = corners

    # Calcula largura e altura do retângulo destino.
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(1, width)
    height = max(1, height)

    # Define o retângulo destino já com margem embutida.
    destination = np.array(
        [[m, m], [m + width, m], [m + width, m + height], [m, m + height]],
        dtype=np.float32,
    )

    # Calcula homografia e gera imagem ortorretificada.
    H = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image_bgr, H, (width + 2 * m, height + 2 * m))


def compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS):
    # Calcula as caixas retangulares de cada célula da grade sobre a imagem ortho.
    #
    # A grade é distribuída uniformemente entre as margens.
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
    #
    # Recebe:
    # - posição (r, c)
    # - tile já recortado
    # - parâmetros do decoder
    #
    # Retorna:
    # - linha
    # - coluna
    # - valor encontrado, ou "?" se falhar
    (
        r,
        c,
        tile_gray,
        timeout,
        shrink,
        border,
        resize_factor,
        skip_empty,
        empty_std_threshold,
        empty_dark_threshold,
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
):
    # Decodifica toda a grade 37x37 da imagem ortorretificada.

    height, width = ortho_bgr.shape[:2]

    # Converte para cinza uma única vez para evitar trabalho repetido.
    ortho_gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)

    # Calcula todas as caixas da grade.
    boxes = compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS)

    out_dir = Path(elements_dir)
    if dump_elements:
        out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for r, c, x0, y0, x1, y1 in boxes:
        # Recorta o tile em grayscale, que é o que será realmente decodificado.
        tile_gray = crop_box(ortho_gray, x0, y0, x1, y1)

        # Opcionalmente salva o tile colorido para inspeção.
        if dump_elements:
            tile_bgr = crop_box(ortho_bgr, x0, y0, x1, y1)
            if tile_bgr is not None:
                cv2.imwrite(str(out_dir / f"r{r}c{c}.png"), tile_bgr)

        # Empacota o job para execução serial ou paralela.
        jobs.append(
            (
                r,
                c,
                tile_gray,
                decode_timeout,
                decode_shrink,
                decode_border,
                resize_factor,
                skip_empty,
                empty_std_threshold,
                empty_dark_threshold,
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

    # Modo paralelo com múltiplos processos.
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r, c, value in ex.map(_decode_worker, jobs, chunksize=chunksize):
                grid[r][c] = value

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
    ortho = build_ortho(image, template, a.margin)

    # Salva ortho para inspeção.
    if not cv2.imwrite(a.output, ortho):
        raise RuntimeError("Falha ao salvar ortho")

    # Se foi pedido teste de uma célula específica, executa só esse caminho.
    if a.test_cell is not None:
        run_test_cell(ortho, a.margin, a)
        return

    # Caso contrário, decodifica a grade inteira.
    grid_text = decode_grid(
        ortho_bgr=ortho,
        margin=a.margin,
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
    )

    # Salva a grade textual final.
    Path(a.grid_output).write_text(grid_text, encoding="utf-8")


if __name__ == "__main__":
    main()