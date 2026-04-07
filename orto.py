"""
Uso no terminal (CLI):

python3 orto.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --margin 60

Isso também cria a pasta "elements/" com os 37x37 recortes: r0c0.png ... r36c36.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default="template.png")
    p.add_argument("--input", default="imagem.png")
    p.add_argument("--output", default="ortho.png")
    p.add_argument("--margin", type=int, default=60)
    return p.parse_args()


def order_corners(points):
    # ordena os 4 pontos como: TL, TR, BR, BL
    s = points.sum(axis=1)
    d = points[:, 0] - points[:, 1]
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmax(d)]
    bl = points[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_four_matches(response, radius):
    # encontra os 4 melhores picos do template matching usando supressão local
    response_copy = response.copy()
    h, w = response_copy.shape
    r = max(1, int(radius))
    matches = []

    for _ in range(4):
        _, _, _, loc = cv2.minMaxLoc(response_copy)
        x, y = int(loc[0]), int(loc[1])
        matches.append([x, y])

        # zera uma região ao redor do pico
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w - 1, x + r), min(h - 1, y + r)
        response_copy[y0:y1 + 1, x0:x1 + 1] = -1.0

    return np.array(matches, dtype=np.float32)


def clamp_int(v, lo, hi):
    return max(lo, min(hi, v))


def crop_square(image, cx, cy, half_side):
    # recorta um quadrado centrado em (cx, cy)
    h, w = image.shape[:2]

    x0 = int(round(cx - half_side))
    y0 = int(round(cy - half_side))
    x1 = int(round(cx + half_side))
    y1 = int(round(cy + half_side))

    x0 = clamp_int(x0, 0, w - 1)
    y0 = clamp_int(y0, 0, h - 1)
    x1 = clamp_int(x1, 0, w)
    y1 = clamp_int(y1, 0, h)

    if x1 <= x0 or y1 <= y0:
        return None

    return image[y0:y1, x0:x1].copy()


def save_grid_elements(ortho, margin, rows=37, cols=37, out_dir="elements"):
    # divide a grade regular e salva cada quadrado
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    H, W = ortho.shape[:2]
    m = float(margin)

    # distância entre centros
    step_x = (W - 2.0 * m) / (cols - 1)
    step_y = (H - 2.0 * m) / (rows - 1)

    # metade do lado do quadrado de recorte
    half_side = min(step_x, step_y) / 2.0

    for r in range(rows):
        cy = m + r * step_y
        for c in range(cols):
            cx = m + c * step_x
            tile = crop_square(ortho, cx, cy, half_side)

            if tile is None:
                continue

            cv2.imwrite(str(out_path / f"r{r}c{c}.png"), tile)


def main():
    args = parse_args()

    image = cv2.imread(args.input)
    template = cv2.imread(args.template)
    if image is None or template is None:
        raise RuntimeError("Erro ao abrir imagem")

    # converte para grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    template_h, template_w = gray_template.shape[:2]

    # template matching
    response = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    # encontra os 4 cantos
    matches = find_four_matches(response, radius=min(template_w, template_h) * 0.6)

    # converte top-left do template para centro
    centers = matches + np.array([template_w / 2.0, template_h / 2.0], np.float32)

    # ordena cantos
    corners = order_corners(centers)

    margin = max(0, int(args.margin))
    tl, tr, br, bl = corners

    # tamanho do retângulo ortorretificado
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(1, width)
    height = max(1, height)

    # coordenadas destino
    destination = np.array(
        [[margin, margin],
         [margin + width, margin],
         [margin + width, margin + height],
         [margin, margin + height]],
        dtype=np.float32
    )

    # homografia e warp
    transform = cv2.getPerspectiveTransform(corners, destination)
    ortho = cv2.warpPerspective(image, transform, (width + 2 * margin, height + 2 * margin))

    if not cv2.imwrite(args.output, ortho):
        raise RuntimeError("Erro ao salvar output")

    # extrai os 37x37 elementos
    save_grid_elements(ortho, margin, rows=37, cols=37, out_dir="elements")


if __name__ == "__main__":
    main()