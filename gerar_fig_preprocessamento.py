#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera a figura preprocessamento_candidatos.png com as CINCO variantes reais do
pipeline (otsu, otsu_bil, adaptive, sharp, gray), aplicadas a uma célula real
extraída da imagem ortorretificada.

A fonte da verdade é a própria função build_candidates_and_bounds de pipeline.py:
o script reusa essa função, de modo que a figura nunca diverge da implementação.

Uso:
    python3 gerar_fig_preprocessamento.py --ortho ortho.png --out preprocessamento_candidatos.png
"""
import argparse
import math
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline as P


def margem_auto(w_total, h_total):
    """Recupera a margem usada na ortorretificação (ceil(metade do passo))."""
    m = 0
    for _ in range(100):
        w = w_total - 2 * m
        h = h_total - 2 * m
        step = min(w / (P.COLS - 1), h / (P.ROWS - 1))
        mn = int(math.ceil(step / 2.0))
        if mn == m:
            break
        m = mn
    return m


def achar_celulas(ortho):
    h_total, w_total = ortho.shape[:2]
    margin = margem_auto(w_total, h_total)
    boxes = P.compute_grid_boxes(h_total, w_total, margin)
    achados = []
    for (r, c, x0, y0, x1, y1) in boxes:
        tile = P.crop_box(ortho, x0, y0, x1, y1)
        if tile is None or tile.size == 0:
            continue
        if float(tile.std()) < 18.0:           # pula células claramente vazias
            continue
        cell_side = max(1, min(x1 - x0, y1 - y0))
        rb = P.refine_tile_box(ortho, x0, y0, x1, y1, max(4, int(cell_side * 0.30)))
        rtile = P.crop_box(ortho, *rb)
        cands, mn, mx = P.build_candidates_and_bounds(rtile, use_edge_bounds=False)
        for _name, img in cands:
            t = P.try_decode_text(img, timeout=300, shrink=1, min_edge=mn, max_edge=mx)
            if t:
                achados.append((r, c, t, rtile.copy()))
                break
        if len(achados) >= 40:
            break
    return achados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", default="ortho.png")
    ap.add_argument("--out", default="preprocessamento_candidatos.png")
    ap.add_argument("--prefer", default="F", help="letra preferida para a célula de exemplo")
    a = ap.parse_args()

    ortho = cv2.imread(a.ortho, cv2.IMREAD_GRAYSCALE)
    if ortho is None:
        sys.exit("Não foi possível abrir " + a.ortho)

    achados = achar_celulas(ortho)
    if not achados:
        sys.exit("Nenhuma célula decodificável encontrada na ortho.")

    # prioriza uma célula de letra única igual a --prefer (como no exemplo original),
    # depois qualquer letra única, sempre com o maior contraste disponível.
    def chave(it):
        _r, _c, t, tile = it
        if len(t) == 1 and t == a.prefer:
            pref = 0
        elif len(t) == 1 and t.isalpha():
            pref = 1
        else:
            pref = 2
        return (pref, -float(tile.std()))

    achados.sort(key=chave)
    r, c, texto, tile = achados[0]
    print("Célula escolhida: (%d, %d)  código='%s'  std=%.1f" % (r, c, texto, tile.std()))

    cands, _mn, _mx = P.build_candidates_and_bounds(tile, use_edge_bounds=False)
    por_nome = {name: img for name, img in cands}

    # Ordem pedagógica (cinza -> realçada -> binarizada -> variações) com rótulos
    # fiéis ao código.
    painel = [
        ("gray",     "(a) Escala de cinza"),
        ("sharp",    "(b) Realce CLAHE + nitidez"),
        ("otsu",     "(c) Otsu sobre o realce"),
        ("otsu_bil", "(d) Otsu + filtragem bilateral"),
        ("adaptive", "(e) Adaptativa gaussiana"),
    ]
    fig, axes = plt.subplots(1, len(painel), figsize=(15, 3.3))
    for ax, (key, titulo) in zip(axes, painel):
        ax.imshow(por_nome[key], cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("Figura salva em:", a.out)


if __name__ == "__main__":
    main()
