import sys
import cv2
from pylibdmtx.pylibdmtx import decode


def try_decode(img, method):
    results = decode(
        img,               # pode passar numpy.ndarray direto
        timeout=40,        # limita tempo por tentativa (ms)
        max_count=1,       # para no primeiro achado
        shrink=2,          # acelera a busca; teste 1 ou 2
    )
    if results:
        r = results[0]
        return {
            "text": r.data.decode("utf-8", errors="replace"),
            "rect": r.rect,
            "method": method,
        }
    return None


def decode_datamatrix(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Não consegui abrir: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ampliação leve; evite crescer demais
    gray2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray2x)

    # nitidez leve (mais barato que fastNlMeansDenoising)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # binarização
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # adicionar quiet zone artificial
    candidates = [
        ("gray", cv2.copyMakeBorder(gray2x, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)),
        ("sharp", cv2.copyMakeBorder(sharp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)),
        ("otsu", cv2.copyMakeBorder(otsu, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)),
    ]

    for method, img in candidates:
        r = try_decode(img, method)
        if r:
            return r

    return None


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 decode.py caminho/para/imagem.png")
        sys.exit(1)

    image_path = sys.argv[1]
    result = decode_datamatrix(image_path)

    if result:
        print("✅ DataMatrix encontrado!")
        print("Método:", result["method"])
        print("Texto :", result["text"])
        print("BBox  :", result["rect"])
    else:
        print("❌ Nenhum DataMatrix reconhecido")


if __name__ == "__main__":
    main()