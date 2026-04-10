"""
pipeline.py - Ortorretificacao e decodificacao DataMatrix em grade.

Adaptado de ECC200Decode/pipeline.py para funcionar como modulo de biblioteca.
Usa a mesma ortorretificacao por template matching e decodificacao DataMatrix
com multiplos pre-processamentos, mas retorna dados estruturados em vez de
escrever arquivos.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode


# Dimensao fixa da grade.
ROWS = 37
COLS = 37


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


def build_ortho(image_bgr, template_bgr, margin):
    """Ortorretifica usando template matching nos 4 cantos + warp perspectivo.

    Levanta OrthoError se os cantos nao podem ser encontrados ou sao invalidos.
    """
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray_template.shape[:2]

    response = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    matches, scores = find_four_matches(response, radius=min(tw, th) * 0.6)
    centers = matches + np.array([tw / 2.0, th / 2.0], np.float32)
    corners = order_corners(centers)

    _validate_corners(corners, image_bgr.shape)

    m = max(0, int(margin))
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(1, width)
    height = max(1, height)

    # Retangulo destino ja com margem embutida.
    destination = np.array(
        [[m, m], [m + width, m], [m + width, m + height], [m, m + height]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image_bgr, H, (width + 2 * m, height + 2 * m))


# ---------------------------------------------------------------------------
# Computacao da grade
# ---------------------------------------------------------------------------

def clamp_int(v, lo, hi):
    """Garante que um valor inteiro fique dentro de [lo, hi]."""
    return max(lo, min(hi, v))


def compute_grid_boxes(height, width, margin, rows=ROWS, cols=COLS):
    """Calcula as caixas retangulares de cada celula da grade 37x37."""
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


def tile_looks_empty(gray, std_threshold=7.0, dark_threshold=0.06):
    """Heuristica para decidir se um tile parece vazio."""
    std = float(gray.std())
    dark_ratio = float(np.mean(gray < 180))
    return std < std_threshold or dark_ratio < dark_threshold


def build_candidates_and_bounds(tile_gray, border=10, resize_factor=2.0,
                                use_edge_bounds=False):
    """Gera imagens candidatas para o decodificador.

    Ordem de prioridade: otsu, sharp, gray.
    """
    # Ampliacao do tile para facilitar a leitura.
    gray_up = cv2.resize(
        tile_gray, None,
        fx=resize_factor, fy=resize_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # Melhora contraste local.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_up)

    # Sharpening leve.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # Binarizacao automatica com Otsu.
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = [
        ("otsu", add_white_border(otsu, border)),
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
    """Decodifica um tile em grayscale. Retorna (texto, metodo) ou (None, None)."""
    if tile_gray is None or tile_gray.size == 0:
        return None, None
    if skip_empty and tile_looks_empty(tile_gray):
        return None, None

    candidates, min_edge, max_edge = build_candidates_and_bounds(
        tile_gray, border=border, resize_factor=resize_factor,
        use_edge_bounds=use_edge_bounds,
    )
    for method, img in candidates:
        text = try_decode_text(img, timeout=timeout, shrink=shrink,
                               min_edge=min_edge, max_edge=max_edge)
        if text is not None:
            return text, method
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


def decode_grid(ortho_bgr, margin, decode_timeout=40, decode_shrink=2,
                decode_border=10, resize_factor=2.0, workers=None,
                chunksize=32, skip_empty=True, use_edge_bounds=False):
    """Decodifica a grade 37x37 completa. Retorna lista 2D de simbolos."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) // 2)

    height, width = ortho_bgr.shape[:2]
    ortho_gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    boxes = compute_grid_boxes(height, width, margin)

    jobs = []
    for r, c, x0, y0, x1, y1 in boxes:
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
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r, c, value in ex.map(_decode_worker, jobs, chunksize=chunksize):
                grid[r][c] = value

    return grid


# ---------------------------------------------------------------------------
# API de alto nivel
# ---------------------------------------------------------------------------

def process_image(image_path, template_path, margin=60, decode_timeout=40,
                  decode_shrink=2, decode_border=10, resize_factor=2.0,
                  workers=None, chunksize=32, skip_empty=True,
                  use_edge_bounds=False, progress_callback=None):
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
    ortho = build_ortho(image, template, margin)

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
    )

    height, width = ortho.shape[:2]
    boxes = compute_grid_boxes(height, width, margin)

    decoded_count = sum(1 for row in grid for s in row if s != "?")
    _progress(f"Concluido - {decoded_count} simbolos decodificados.")

    return ortho, grid, boxes
