import argparse
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
    s = points.sum(axis=1)
    d = points[:, 0] - points[:, 1]
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmax(d)]
    bl = points[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_four_matches(response, radius):
    response_copy = response.copy()
    h, w = response_copy.shape
    r = max(1, int(radius))
    matches = []

    for _ in range(4):
        _, _, _, loc = cv2.minMaxLoc(response_copy)
        x, y = int(loc[0]), int(loc[1])
        matches.append([x, y])

        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w - 1, x + r), min(h - 1, y + r)
        response_copy[y0:y1 + 1, x0:x1 + 1] = -1.0

    return np.array(matches, dtype=np.float32)


def main():
    args = parse_args()

    image = cv2.imread(args.input)
    template = cv2.imread(args.template)
    if image is None or template is None:
        raise RuntimeError("Erro ao abrir imagem")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    template_h, template_w = gray_template.shape[:2]

    response = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    matches = find_four_matches(response, radius=min(template_w, template_h) * 0.6)

    centers = matches + np.array([template_w / 2.0, template_h / 2.0], np.float32)
    corners = order_corners(centers)

    margin = max(0, int(args.margin))
    tl, tr, br, bl = corners

    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(1, width)
    height = max(1, height)

    destination = np.array(
        [[margin, margin],
         [margin + width, margin],
         [margin + width, margin + height],
         [margin, margin + height]],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(corners, destination)
    ortho = cv2.warpPerspective(image, transform, (width + 2 * margin, height + 2 * margin))

    if not cv2.imwrite(args.output, ortho):
        raise RuntimeError("Erro ao salvar output")


if __name__ == "__main__":
    main()