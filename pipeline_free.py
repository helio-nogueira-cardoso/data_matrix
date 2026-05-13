"""
pipeline_livre.py

Pipeline de detecção de códigos DataMatrix ECC200 sem uso de grade fixa.

====================================================
VISÃO GERAL
====================================================

Este arquivo implementa uma abordagem "livre" para localizar e decodificar
símbolos ECC200 em uma imagem ortorretificada, sem depender de percorrer
uma grade fixa célula por célula.

A lógica geral do pipeline é:

1. Ler a imagem de entrada e o template.
2. Fazer a ortorretificação da imagem.
3. Gerar regiões candidatas por duas estratégias:
   - heatmap local de contraste/estrutura
   - componentes conectados / contornos
4. Mesclar e filtrar essas regiões candidatas.
5. Tentar decodificar cada região candidata.
6. Remover duplicatas.
7. Salvar:
   - um JSON com os símbolos encontrados e suas posições
   - uma imagem anotada com bounding boxes
   - opcionalmente, os candidatos recortados para debug

====================================================
PARÂMETROS DE CONFIGURAÇÃO
====================================================

Arquivos:
---------
--template
    Caminho para a imagem template usada na ortorretificação.

--input
    Caminho para a imagem de entrada que será processada.

--output
    Caminho da imagem ortorretificada que será salva.

--margin
    Margem usada na ortorretificação.
    Também influencia a estimativa de tamanho esperado dos símbolos.

--results-json
    Caminho do arquivo JSON onde serão salvas as detecções finais.

--annotated-output
    Caminho da imagem anotada com caixas e rótulos.

Debug:
------
--dump-candidates
    Se ativado, salva em disco todas as ROIs candidatas consideradas
    após a etapa de filtragem global.

--candidates-dir
    Pasta onde serão salvos os candidatos quando --dump-candidates for usado.

Parâmetros do decoder:
----------------------
--decode-timeout
    Timeout, em milissegundos, dado ao decoder para cada tentativa.
    Valores maiores podem recuperar mais códigos, mas deixam a execução mais lenta.

--decode-shrink
    Parâmetro repassado ao decoder para controlar redução interna.
    Pode afetar desempenho e taxa de sucesso.

--decode-border
    Quantidade de borda adicionada na construção das imagens candidatas
    para decodificação.

--resize-factor
    Fator de redimensionamento aplicado às ROIs antes de tentar decodificar.
    Pode ajudar quando o símbolo está pequeno.

--use-edge-bounds
    Se ativado, usa limites baseados em bordas ao construir imagens candidatas.

Paralelismo:
------------
--workers
    Número de processos usados para decodificar candidatos em paralelo.

--chunksize
    Quantos payloads são enviados por vez para cada worker.
    Pode influenciar desempenho.

Proposição de candidatos:
-------------------------
--proposal-scale
    Escala da imagem usada para propor candidatos.
    Exemplo:
      0.70 = propõe candidatos em imagem reduzida a 70%.
    Escalas menores aceleram o processo, mas podem perder detalhes.

--max-candidates
    Número máximo de candidatos mantidos após a filtragem global.

--max-candidates-per-family
    Número máximo de candidatos mantidos por família de proposta
    (por exemplo, cada tamanho de janela do heatmap, ou cada mapa binário
    em componentes conectados).

--nms-iou
    Limiar de IoU usado no NMS.
    Se duas caixas se sobrepõem demais, a pior é descartada.

--merge-distance
    Distância máxima entre centros para considerar duas caixas como equivalentes,
    mesmo quando o IoU não é tão alto.

--pad
    Padding mínimo adicionado ao redor da caixa antes da decodificação.

Filtros rápidos:
----------------
--skip-empty
    Se ativado, tenta descartar ROIs que parecem vazias antes de decodificar.

--empty-std-threshold
    Limiar de desvio padrão usado para definir se uma ROI parece vazia.

--empty-dark-threshold
    Limiar de razão de pixels escuros para definir se uma ROI parece vazia.

--min-local-dark-ratio
    Proporção mínima de pixels escuros para uma ROI ser considerada interessante.

Parâmetros herdados da grade:
-----------------------------
--rows
    Número de linhas da grade esperada no pipeline original.
    Aqui não é usado para varredura rígida, mas para estimar o tamanho típico
    dos símbolos na imagem.

--cols
    Número de colunas da grade esperada.

Parâmetros do heatmap:
----------------------
--window-size-ratios
    Lista de razões multiplicativas usadas para gerar diferentes tamanhos
    de janela no heatmap.
    Ex.: "0.70,0.90,1.10"

--heatmap-threshold
    Limiar mínimo do mapa de pontuação local para aceitar um máximo local
    como candidato.

--stride-ratio
    Reservado para possíveis variações futuras.
    No código atual não tem papel central.
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Funções importadas do pipeline original:
# - build_ortho: gera a imagem ortorretificada
# - build_candidates_and_bounds: monta variações de ROI para decodificação
# - try_decode_text: tenta decodificar um DataMatrix
# - tile_looks_empty: tenta identificar ROIs vazias
from pipeline import (
    build_ortho,
    build_candidates_and_bounds,
    try_decode_text,
    tile_looks_empty,
    looks_like_valid_symbol,
    refine_tile_box,
    compute_grid_boxes,
)

# Quantidade de linhas e colunas usadas como referência de escala.
ROWS = 37
COLS = 37


def parse_args():
    """
    Define e lê todos os argumentos de linha de comando.

    Esta função centraliza a configuração do pipeline.
    Mesmo na abordagem sem grade fixa, alguns parâmetros do pipeline antigo
    continuam úteis para estimar o tamanho típico dos símbolos.
    """
    p = argparse.ArgumentParser(
        description="Detecção híbrida de DataMatrix ECC200 sem varrer rigidamente a grade."
    )

    # =========================
    # Arquivos de entrada/saída
    # =========================
    p.add_argument("--template", default="template.png")
    p.add_argument("--input", default="imagem.png")
    p.add_argument("--output", default="ortho.png")
    # Default None: build_ortho calcula automaticamente como ceil(passo/2),
    # mesma escolha do pipeline.py grid. Margem fixa em 60 gerava clamping
    # de células de borda (config_4_sample_1 (3,1) o grid candidate vinha
    # com tile 134x148 cortado, enquanto auto dá 147x147 completo).
    p.add_argument("--margin", type=int, default=None)
    p.add_argument("--results-json", default="symbols.json")
    p.add_argument("--annotated-output", default="annotated.png")

    # =========================
    # Debug
    # =========================
    p.add_argument("--dump-candidates", action="store_true")
    p.add_argument("--candidates-dir", default="candidates")

    # =========================
    # Parâmetros do decoder
    # =========================
    # timeout em ms por tentativa de decodificação. O libdmtx resolve imagens
    # saudáveis em <30 ms; tiles limítrofes precisam de mais folga,
    # especialmente quando os workers estão competindo por CPU. 80 ms cobre
    # o caso comum sem inflar demais o tempo em tiles impossíveis.
    p.add_argument("--decode-timeout", type=int, default=80)
    p.add_argument("--decode-shrink", type=int, default=2)
    p.add_argument("--decode-border", type=int, default=10)
    p.add_argument("--resize-factor", type=float, default=2.0)
    p.add_argument("--use-edge-bounds", action="store_true")

    # =========================
    # Paralelismo
    # =========================
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    # chunksize maior reduz drasticamente o overhead de IPC quando o número de
    # candidatos é grande (375+); mantém os workers ocupados.
    p.add_argument("--chunksize", type=int, default=32)

    # =========================
    # Proposição de candidatos
    # =========================
    p.add_argument("--proposal-scale", type=float, default=0.70)
    # Lista de escalas extras para multi-pass de proposição. Default "0.90"
    # porque a escala única 0.70 perde tiles em paredes densas: o filtro
    # de máximos locais com dilatação suprime peaks vizinhos quando o
    # entorno tem muitos componentes escuros adjacentes. Rodar também a
    # 0.90 (janela maior) recupera esses peaks suprimidos. Recuperação
    # típica: 9-23 células em config_1_sample_2/_3, config_3_sample_1 e
    # 1 célula S em imagem_90 do baseline, sem introduzir misreads
    # (escala 0.50 testada introduzia 'F'→'E' em imagem_90; 0.90 não).
    p.add_argument("--proposal-scales", default="0.90")
    p.add_argument("--max-candidates", type=int, default=20000)
    # Cap por família (heatmap e components, antes do NMS global). Default 200:
    # com a adição de propose_from_grid como terceira fonte cobrindo as
    # 1369 células nominais, podemos manter o heatmap/components em 200 por
    # família — qualquer célula que o NMS interno descarte (e que tenha
    # símbolo real) cai no grid candidate e é recuperada pelo dedup
    # posicional. Manter 500 (valor anterior) só inflava o tempo de decode
    # sem ganho final em acurácia depois do grid.
    p.add_argument("--max-candidates-per-family", type=int, default=200)
    p.add_argument("--nms-iou", type=float, default=0.30)
    p.add_argument("--merge-distance", type=int, default=12)
    p.add_argument("--pad", type=int, default=8)

    # =========================
    # Filtros rápidos de descarte
    # =========================
    p.add_argument("--skip-empty", action="store_true", default=True)
    p.add_argument("--empty-std-threshold", type=float, default=7.0)
    p.add_argument("--empty-dark-threshold", type=float, default=0.05)
    p.add_argument("--min-local-dark-ratio", type=float, default=0.025)

    # =========================
    # Padding na borda do ortho
    # =========================
    # Quando símbolos ficam encostados no topo (y=0) ou em qualquer borda do
    # ortho — acontece quando os marcadores de canto não cobrem toda a área de
    # interesse — o heatmap perde contexto e a proposta deixa o símbolo de fora.
    # Pré-padder com branco resolve sem mexer na homografia.
    p.add_argument("--edge-pad", type=int, default=80)

    # =========================
    # Escala típica do símbolo
    # =========================
    p.add_argument("--rows", type=int, default=ROWS)
    p.add_argument("--cols", type=int, default=COLS)

    # =========================
    # Parâmetros da proposta via heatmap
    # =========================
    p.add_argument("--window-size-ratios", default="0.70,0.90,1.10")
    p.add_argument("--heatmap-threshold", type=float, default=0.10)
    p.add_argument("--stride-ratio", type=float, default=0.35)

    return p.parse_args()


def clamp_box(box, w, h):
    """
    Ajusta uma bounding box para que ela fique contida dentro dos limites da imagem.

    Parâmetros:
        box: tupla (x, y, largura, altura)
        w: largura da imagem
        h: altura da imagem

    Retorno:
        Tupla (x, y, largura, altura) corrigida.
    """
    x, y, bw, bh = box

    # Garante que o canto superior esquerdo esteja dentro da imagem
    x = max(0, min(int(round(x)), w - 1))
    y = max(0, min(int(round(y)), h - 1))

    # Garante que o canto inferior direito também fique dentro da imagem
    x2 = max(x + 1, min(int(round(x + bw)), w))
    y2 = max(y + 1, min(int(round(y + bh)), h))

    return x, y, x2 - x, y2 - y


def expand_box(box, pad, w, h):
    """
    Expande uma caixa em todas as direções adicionando 'pad' pixels de margem.

    Isso é útil antes da decodificação, pois o símbolo pode precisar de um pequeno
    contexto extra ao redor para ser decodificado com mais estabilidade.
    """
    x, y, bw, bh = box
    return clamp_box((x - pad, y - pad, bw + 2 * pad, bh + 2 * pad), w, h)


def box_iou(a, b):
    """
    Calcula o IoU (Intersection over Union) entre duas bounding boxes.

    IoU = área da interseção / área da união

    Quanto maior o IoU, maior a sobreposição entre as caixas.
    Esse valor é usado para remover duplicatas.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b):
    """
    Calcula a distância euclidiana entre os centros de duas caixas.

    Mesmo que o IoU seja baixo, duas caixas podem representar o mesmo símbolo
    se os seus centros estiverem muito próximos.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0

    return float(np.hypot(acx - bcx, acy - bcy))


def nms_candidates(candidates, nms_iou, merge_distance, max_keep=None):
    """
    Aplica uma forma simples de NMS (Non-Maximum Suppression).

    Objetivo:
        Remover candidatos redundantes mantendo preferencialmente os de maior score.

    Estratégia:
        1. Ordena os candidatos por score (maior primeiro)
        2. Percorre os candidatos em ordem
        3. Aceita um candidato apenas se ele não for muito parecido com
           nenhum já aceito
        4. A semelhança é definida por:
           - IoU alto, ou
           - centros muito próximos

    Parâmetros:
        candidates: lista de dicionários com pelo menos "box" e "score"
        nms_iou: limiar de IoU para considerar sobreposição excessiva
        merge_distance: distância máxima entre centros para considerar duplicata
        max_keep: número máximo de candidatos a manter (ou None)

    Retorno:
        Lista filtrada de candidatos.
    """
    candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)
    out = []

    for cand in candidates:
        ok = True
        for prev in out:
            if (
                box_iou(cand["box"], prev["box"]) > nms_iou
                or center_distance(cand["box"], prev["box"]) <= merge_distance
            ):
                ok = False
                break

        if ok:
            out.append(cand)
            if max_keep is not None and len(out) >= max_keep:
                break

    return out


def estimate_symbol_side(gray, margin, rows, cols, ref_shape=None):
    """
    Estima o lado típico de um símbolo na imagem ortorretificada.

    Embora o pipeline livre não percorra uma grade fixa, ele ainda usa
    a densidade esperada de linhas/colunas como dica para estimar o tamanho
    aproximado dos símbolos.

    Isso ajuda a escolher janelas plausíveis na etapa de proposição.

    Quando ``ref_shape`` é informado (formato (h, w)), ele substitui o shape
    real do ``gray``. É usado quando o gray foi pré-padded com uma borda
    branca para detecção de símbolos de borda — nesse caso o tamanho do
    símbolo deve continuar baseado no ortho original, não no padded.
    """
    if ref_shape is not None:
        h, w = ref_shape[:2]
    else:
        h, w = gray.shape[:2]

    step_x = max(4.0, (w - 2.0 * margin) / max(1, cols - 1))
    step_y = max(4.0, (h - 2.0 * margin) / max(1, rows - 1))

    # Usa o menor passo como tamanho de referência, mas impõe um mínimo
    return max(8.0, min(step_x, step_y))


def local_score_map(gray_small):
    """
    Gera um mapa de score local para destacar regiões visualmente interessantes.

    A ideia aqui é identificar regiões que podem conter um DataMatrix usando:
    - desvio padrão local: mede variação/estrutura
    - blackhat morfológico: destaca regiões escuras sobre fundo mais claro

    O resultado é um mapa em que regiões potencialmente úteis tendem a ter
    score maior.
    """
    # Suavização leve para reduzir ruído
    blur = cv2.GaussianBlur(gray_small, (0, 0), 1.0)

    # Média local
    mean = cv2.boxFilter(blur, ddepth=-1, ksize=(11, 11), normalize=True)

    # Média local dos quadrados, usada para obter a variância local
    sqmean = cv2.boxFilter(
        (blur.astype(np.float32) ** 2),
        ddepth=-1,
        ksize=(11, 11),
        normalize=True,
    )

    # Variância = E[x²] - (E[x])²
    var = np.maximum(0.0, sqmean - mean.astype(np.float32) ** 2)
    std = np.sqrt(var)

    # Normaliza o desvio padrão para [0, 255]
    std = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Blackhat realça padrões escuros pequenos/estruturados
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # Combinação linear dos dois mapas
    score = cv2.addWeighted(std, 0.55, blackhat.astype(np.uint8), 0.45, 0)
    return score


def propose_from_heatmap(gray, args, ref_shape=None):
    """
    Gera candidatos usando um heatmap local.

    Fluxo:
        1. Reduz a imagem para acelerar
        2. Estima tamanhos plausíveis de janela
        3. Calcula mapa de score local
        4. Procura máximos locais acima do limiar
        5. Cria caixas em torno desses máximos
        6. Filtra duplicatas dentro da própria família

    Retorno:
        Lista de candidatos no formato:
        {
            "box": [x, y, w, h],
            "score": float,
            "source": "heat_<lado>"
        }
    """
    h, w = gray.shape[:2]
    scale = float(args.proposal_scale)

    # Reduz a imagem para acelerar a proposta de candidatos
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 0.999 else gray
    )
    sh, sw = small.shape[:2]

    # Estima o lado típico do símbolo na imagem original e adapta para a escala reduzida
    est_side_full = estimate_symbol_side(
        gray, args.margin, args.rows, args.cols, ref_shape=ref_shape,
    )
    est_side_small = max(6.0, est_side_full * scale)

    # Gera diferentes tamanhos de janela a partir de razões multiplicativas
    sizes = [
        max(6, int(round(est_side_small * r)))
        for r in [float(x) for x in args.window_size_ratios.split(",") if x.strip()]
    ]

    heat = local_score_map(small)
    heat_f = heat.astype(np.float32) / 255.0

    candidates = []

    for side in sizes:
        kernel = max(5, int(round(side)))
        if kernel % 2 == 0:
            kernel += 1

        # Suaviza o heatmap numa escala parecida com a janela atual
        pooled = cv2.GaussianBlur(heat_f, (kernel, kernel), 0)

        # Dilatação para localizar máximos locais
        maxf = cv2.dilate(pooled, np.ones((kernel, kernel), np.uint8))

        # Máscara de máximos locais acima do limiar
        mask = (pooled >= maxf - 1e-6) & (pooled >= args.heatmap_threshold)

        ys, xs = np.where(mask)
        local = []

        for y, x in zip(ys.tolist(), xs.tolist()):
            # Cria uma caixa centrada no máximo local
            x0 = x - side // 2
            y0 = y - side // 2

            box_small = clamp_box((x0, y0, side, side), sw, sh)
            bx, by, bw, bh = box_small

            roi = small[by:by + bh, bx:bx + bw]
            if roi.size == 0:
                continue

            # Medidas rápidas para ver se a ROI faz sentido
            dark_ratio = float(np.mean(roi < 180))
            std = float(roi.std())

            # ROI clara demais / sem estrutura relevante
            if dark_ratio < args.min_local_dark_ratio:
                continue

            # Score combinado
            score = float(pooled[y, x] * 120.0 + std * 0.7 + dark_ratio * 40.0)

            # Reprojeta caixa da escala reduzida para a escala original
            full_box = clamp_box((bx / scale, by / scale, bw / scale, bh / scale), w, h)

            local.append({
                "box": full_box,
                "score": score,
                "source": f"heat_{side}",
            })

        # Remove redundâncias dentro desta família de candidatos
        local = nms_candidates(
            local,
            args.nms_iou,
            max(8, args.merge_distance),
            args.max_candidates_per_family,
        )
        candidates.extend(local)

    return candidates


def propose_from_components(gray, args, ref_shape=None):
    """
    Gera candidatos a partir de componentes conectados / contornos.

    Ideia:
        1. Reduz a imagem
        2. Realça contraste local
        3. Binariza
        4. Cria mapas auxiliares
        5. Encontra contornos
        6. Filtra por geometria e conteúdo visual

    Essa estratégia complementa a abordagem por heatmap.
    """
    h, w = gray.shape[:2]
    scale = float(args.proposal_scale)

    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 0.999 else gray
    )
    sh, sw = small.shape[:2]

    # Realce local de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(small)

    # Leve sharpening
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # Binarização automática por Otsu
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - otsu

    # Diferentes mapas para capturar tipos de estrutura diferentes
    maps = {
        "inv_otsu": inv,
        "closed": cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
        "grad": cv2.morphologyEx(inv, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
    }

    # Define faixa plausível de tamanho do símbolo nessa escala
    est_side_small = estimate_symbol_side(
        gray, args.margin, args.rows, args.cols, ref_shape=ref_shape,
    ) * scale
    min_side = max(5, int(round(est_side_small * 0.40)))
    max_side = max(min_side + 2, int(round(est_side_small * 1.80)))

    all_cands = []

    for name, binary in maps.items():
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        local = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Filtra por tamanho plausível
            if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
                continue

            # Filtra por proporção de aspecto plausível
            aspect = bw / max(1.0, bh)
            if not (0.45 <= aspect <= 1.9):
                continue

            roi = small[y:y + bh, x:x + bw]
            if roi.size == 0:
                continue

            area = cv2.contourArea(cnt)
            fill_ratio = float(area / max(1.0, bw * bh))
            dark_ratio = float(np.mean(roi < 180))
            std = float(roi.std())

            # Remove regiões muito vazias ou pouco plausíveis
            if dark_ratio < args.min_local_dark_ratio or fill_ratio < 0.08:
                continue

            score = std * 1.0 + dark_ratio * 45.0 + fill_ratio * 18.0

            # Reprojeta para a escala original
            full_box = clamp_box((x / scale, y / scale, bw / scale, bh / scale), w, h)

            local.append({
                "box": full_box,
                "score": float(score),
                "source": name,
            })

        # Remove redundâncias dentro desta família
        local = nms_candidates(
            local,
            max(0.20, args.nms_iou - 0.03),
            max(6, args.merge_distance - 2),
            args.max_candidates_per_family,
        )
        all_cands.extend(local)

    return all_cands


def propose_from_grid(args, ortho_shape, edge_pad):
    """Propostas alinhadas à grade nominal 37x37 (mesma geometria que
    pipeline.py grid).

    Cobre regiões onde heatmap e components propõem caixas off-center
    cujos bits decodificam para um símbolo válido pelo allowlist mas
    diferente do real (caso (3,1)='H' lido como 'X' por otsu_bil em
    config_4_sample_1). Para esses tiles, a caixa nominal da grade
    decodifica corretamente, e a deduplicação posterior (por posição,
    preferindo a leitura com mais votos do voto majoritário) escolhe a
    leitura certa entre as duas.

    Score baixo (~10) faz com que essas propostas só vençam o dedup
    quando têm mais votos que o heatmap concorrente.
    """
    oh, ow = ortho_shape
    boxes = compute_grid_boxes(oh, ow, args.margin, rows=args.rows, cols=args.cols)
    candidates = []
    for r, c, x0, y0, x1, y1 in boxes:
        px = x0 + edge_pad
        py = y0 + edge_pad
        pw = x1 - x0
        ph = y1 - y0
        candidates.append({
            "box": (int(px), int(py), int(pw), int(ph)),
            "score": 10.0,
            "source": f"grid_{r}_{c}",
        })
    return candidates


def _vote_decode_candidates(candidates, timeout, shrink, min_edge, max_edge):
    # Voto majoritário entre pré-processamentos. Mesma lógica de
    # pipeline.py::decode_datamatrix_gray_with_method: roda os dois primeiros
    # (otsu, otsu_bil) e, se concordam num símbolo válido, aceita imediatamente
    # (caminho rápido). Se discordam ou ambos falham, roda o resto e escolhe a
    # leitura mais frequente, com a ordem do cascade como tiebreaker.
    #
    # Retorna (texto, metodo, n_votes) onde n_votes é o número de
    # pré-processamentos que concordaram na leitura vencedora. Útil para
    # comparar "confiança" entre voto na caixa nominal e na caixa refinada
    # (caso de (3,1)=X em config_4_sample_1: nominal tem só 1 voto em 'X';
    # refinada tem 2+ votos em 'H'). Retorna (None, None, 0) se nenhum
    # pré-processamento der leitura válida.
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
        return winner, winning_method, 2

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
        return winner, winning_method, max_count

    return None, None, 0


def decode_candidate_from_box(gray, box, args):
    """
    Tenta decodificar um único candidato.

    Estratégia em duas fases para equilibrar tempo e recall:

    Fase 1 — pad nominal: usa o pad dinâmico baseado no tamanho do box e
    varre todos os pré-processamentos (otsu/otsu_bil/adaptive/sharp/gray).
    Cobre quase todos os tiles "saudáveis".

    Fase 2 — pad apertado, só rodada se a fase 1 falhou e o tile parece ter
    conteúdo real (heurística cheap baseada em std e dark_ratio). Símbolos
    cuja vizinhança contém ruído ou um símbolo adjacente decodificam melhor
    com pad menor. Aqui usamos todos os pré-processamentos.

    Cada caixa é decodificada apenas na orientação em que aparece no ortho —
    não rotacionamos o tile como fallback. Rotacionar o tile seria explorar
    a falta de invariância de rotação interna do libdmtx em casos limítrofes,
    quebrando a premissa de que cada imagem deve ser processada na sua
    orientação real.

    Retorna o primeiro acerto válido pelo allowlist; None caso contrário.
    """
    h, w = gray.shape[:2]
    base_pad = max(args.pad, int(0.12 * min(box[2], box[3])))

    # ---- Fase 1: caixa expandida pelo pad nominal -------------------------
    x, y, bw, bh = expand_box(box, base_pad, w, h)
    tile_gray = gray[y:y + bh, x:x + bw]
    if tile_gray.size == 0:
        return None

    if args.skip_empty and tile_looks_empty(
        tile_gray, args.empty_std_threshold, args.empty_dark_threshold,
    ):
        return None

    local_dark_ratio = float(np.mean(tile_gray < 180)) if tile_gray.size else 0.0
    if local_dark_ratio < args.min_local_dark_ratio:
        return None

    candidate_imgs, min_edge, max_edge = build_candidates_and_bounds(
        tile_gray,
        border=args.decode_border,
        resize_factor=args.resize_factor,
        use_edge_bounds=args.use_edge_bounds,
    )

    # Voto majoritário entre os pré-processamentos canônicos (mesma lógica de
    # pipeline.py grid). Cascade-com-early-exit pegava a primeira leitura
    # válida pelo allowlist, mesmo quando outros pré-processamentos votariam
    # diferente — caso típico: tile na borda do ortho onde otsu_bil lê 'X'
    # mas otsu/adaptive/sharp/gray leem 'H'.
    winner, winning_method, n_votes = _vote_decode_candidates(
        candidate_imgs, args.decode_timeout, args.decode_shrink,
        min_edge, max_edge,
    )

    # Cross-check com refine: quando o vencedor nominal tem só 1 voto
    # (baixa confiança), tenta refinar a caixa (centrar no componente conexo
    # mais próximo) e re-votar. Se o refinado der ≥2 votos, preferi-lo;
    # senão, mantém o nominal. Resolve casos onde o heatmap propõe uma caixa
    # ligeiramente off-center cujos bits enviesam o decoder para uma leitura
    # cosmética válida pelo allowlist (caso (3,1)=H lido como 'X' por
    # otsu_bil em config_4_sample_1).
    if winner is not None and n_votes <= 1:
        rx, ry, rxe, rye = refine_tile_box(
            gray, x, y, x + bw, y + bh,
            max_shift=max(4, int(round(min(bw, bh) * 0.30))),
        )
        if (rx, ry, rxe, rye) != (x, y, x + bw, y + bh):
            r_tile = gray[ry:rye, rx:rxe]
            if r_tile.size > 0:
                r_cands, r_mn, r_mx = build_candidates_and_bounds(
                    r_tile,
                    border=args.decode_border,
                    resize_factor=args.resize_factor,
                    use_edge_bounds=args.use_edge_bounds,
                )
                r_winner, r_method, r_votes = _vote_decode_candidates(
                    r_cands, args.decode_timeout, args.decode_shrink,
                    r_mn, r_mx,
                )
                if r_winner is not None and r_votes >= 2 and r_votes > n_votes:
                    return {
                        "text": r_winner,
                        "method": f"{r_method}_refined",
                        "box": [int(rx), int(ry), int(rxe - rx), int(rye - ry)],
                        "center": [
                            float(rx + (rxe - rx) / 2.0),
                            float(ry + (rye - ry) / 2.0),
                        ],
                        "n_votes": int(r_votes),
                    }

    if winner is not None:
        return {
            "text": winner,
            "method": winning_method,
            "box": [int(x), int(y), int(bw), int(bh)],
            "center": [float(x + bw / 2.0), float(y + bh / 2.0)],
            "n_votes": int(n_votes),
        }

    # Decide se vale a pena pagar as tentativas extras: só se o tile parece
    # ter conteúdo real (alto contraste interno). Tiles ruidosos com pouca
    # estrutura quase nunca decodificam mesmo com todas as variações.
    # Threshold 14 (abaixo da pipeline.py grid que usa 18) porque os tiles
    # de grid candidate em pipeline_free são maiores que os da grid (pad=17
    # em vez de margem nominal), diluindo o std do símbolo na vizinhança
    # do furo da base. Caso (3,26)='H' em config_1_sample_2: std=17.08 no
    # tile 182x182 expandido, abaixo do limiar 18, mas o crop central 55%
    # decodifica 'H' corretamente.
    tile_std = float(tile_gray.std())
    if tile_std < 14.0 or local_dark_ratio < 0.04:
        return None

    # ---- Fase 2: pad apertado com todos os pré-processamentos.
    # Símbolos cuja vizinhança contém ruído ou um símbolo adjacente decodificam
    # melhor com pad menor. Como esta fase só roda em tiles "promissores"
    # (filtro acima), o custo agregado fica controlado.
    tight_pad = max(2, base_pad // 2)
    if tight_pad != base_pad:
        ax, ay, aw, ah = expand_box(box, tight_pad, w, h)
        if aw > 0 and ah > 0:
            alt_tile = gray[ay:ay + ah, ax:ax + aw]
            if alt_tile.size > 0:
                alt_imgs, mn2, mx2 = build_candidates_and_bounds(
                    alt_tile,
                    border=args.decode_border,
                    resize_factor=args.resize_factor,
                    use_edge_bounds=args.use_edge_bounds,
                )
                for method, img in alt_imgs:
                    text = try_decode_text(
                        img,
                        timeout=args.decode_timeout,
                        shrink=args.decode_shrink,
                        min_edge=mn2,
                        max_edge=mx2,
                    )
                    if text is not None and looks_like_valid_symbol(text):
                        return {
                            "text": text,
                            "method": f"{method}_tight",
                            "box": [int(ax), int(ay), int(aw), int(ah)],
                            "center": [float(ax + aw / 2.0), float(ay + ah / 2.0)],
                            "n_votes": 1,
                        }

    # ---- Fase 3: multi-crop central (mesma estratégia do pipeline.py grid).
    # Para símbolos físicos pequenos relativos à célula (caso (3,26)='H' em
    # config_1_sample_2 onde o símbolo ocupa ~30% do tile 148x148 nominal),
    # os pré-processamentos canônicos e o pad apertado falham porque a
    # vizinhança domina o cálculo de Otsu. Crops centrais de 75%, 65% e
    # 55% do lado isolam o símbolo, melhorando a binarização.
    h_tile, w_tile = tile_gray.shape[:2]
    if min(h_tile, w_tile) >= 16:
        for shrink_ratio in (0.75, 0.65, 0.55):
            nh = int(round(h_tile * shrink_ratio))
            nw = int(round(w_tile * shrink_ratio))
            if nh < 12 or nw < 12 or nh >= h_tile or nw >= w_tile:
                continue
            y0c = (h_tile - nh) // 2
            x0c = (w_tile - nw) // 2
            cropped = tile_gray[y0c:y0c + nh, x0c:x0c + nw]
            if cropped.size == 0:
                continue
            sub_cands, mn3, mx3 = build_candidates_and_bounds(
                cropped, border=args.decode_border,
                resize_factor=args.resize_factor,
                use_edge_bounds=args.use_edge_bounds,
            )
            # Aqui usamos voto majoritário: o caso (3,26) tem 2 votos (otsu
            # e adaptive) para 'H' no crop 55%, e queremos colher a leitura
            # consistente em vez de aceitar a primeira do cascade.
            c_winner, c_method, c_votes = _vote_decode_candidates(
                sub_cands, args.decode_timeout, args.decode_shrink, mn3, mx3,
            )
            if c_winner is not None:
                # Reporta box do tile original (não o cropado) para que o
                # dedup posicional siga funcionando.
                return {
                    "text": c_winner,
                    "method": f"{c_method}_c{int(shrink_ratio * 100)}",
                    "box": [int(x), int(y), int(bw), int(bh)],
                    "center": [float(x + bw / 2.0), float(y + bh / 2.0)],
                    "n_votes": int(c_votes),
                }

    return None


def decode_worker(payload):
    """
    Worker usado pelo ProcessPoolExecutor.

    Como argumentos de linha de comando não são passados diretamente como objeto
    simples entre processos da forma mais prática, aqui reconstruímos um pequeno
    objeto 'args' a partir de um dicionário.
    """
    gray, cand, args_dict = payload

    class A:
        pass

    args = A()
    for k, v in args_dict.items():
        setattr(args, k, v)

    result = decode_candidate_from_box(gray, cand["box"], args)

    if result is not None:
        result["proposal_score"] = float(cand["score"])
        result["proposal_source"] = cand["source"]

    return result


def deduplicate_decoded(results, iou_threshold=0.22, center_threshold=12,
                         position_threshold=70):
    """
    Remove duplicatas das detecções já decodificadas em dois regimes:

    Regime 1 — mesmo texto, caixas próximas (default IoU>0.22 ou distância
    de centro ≤ center_threshold): mantém só a melhor (proposal_score).
    Caso clássico: duas escalas do heatmap propõem o mesmo símbolo.

    Regime 2 — textos diferentes, caixas muito próximas (distância de centro
    ≤ position_threshold, default 70px ~ meia célula): mantém a leitura com
    mais `n_votes` (saída do voto majoritário sobre os 5 pré-processamentos).
    Tiebreak: maior proposal_score. Resolve o caso onde a caixa do heatmap
    decodifica para um símbolo cosmético do allowlist (ex. 'X' por otsu_bil
    isoladamente) enquanto a caixa nominal da grade decodifica para o real
    (ex. 'H' com 3+ pré-processamentos concordando) — (3,1) em
    config_4_sample_1 é exatamente esse caso.

    A ordenação por (n_votes, proposal_score) decrescente garante que a
    leitura mais confiável seja inserida primeiro e use a posição como
    "lugar reservado".
    """
    unique = []

    def sort_key(d):
        # Prioridade: (n_votes, prefere_grid_em_baixa_confiança, proposal_score).
        # Em empates de votos onde o vencedor tem só 1 voto (single
        # preprocessing válido), preferimos a fonte 'grid_*' à 'heat_*' ou
        # 'comp_*'. Caso (3,26)='H' em config_1_sample_2: heatmap propôs uma
        # caixa que decodifica para 'G' (1 voto, score=77.9); grid_2_11
        # decodifica 'H' (1 voto, score=10.0). Sem este bias, score venceria
        # e geraria misread; com o bias, grid vence porque está alinhada à
        # geometria nominal da maquete. Para n_votes ≥ 2 (alta confiança em
        # qualquer fonte), score volta a desempatar.
        n = d.get("n_votes", 0)
        src = d.get("proposal_source", "")
        is_grid = src.startswith("grid_")
        grid_priority = 1 if (n <= 1 and is_grid) else 0
        return (n, grid_priority, d.get("proposal_score", 0.0))

    for cand in sorted(results, key=sort_key, reverse=True):
        dup = False

        for prev in unique:
            same_text = cand["text"] == prev["text"]
            near_text = (
                box_iou(tuple(cand["box"]), tuple(prev["box"])) > iou_threshold
                or center_distance(tuple(cand["box"]), tuple(prev["box"])) <= center_threshold
            )
            very_close = (
                center_distance(tuple(cand["box"]), tuple(prev["box"]))
                <= position_threshold
            )
            if same_text and near_text:
                dup = True
                break
            if not same_text and very_close:
                # Conflito posicional: já tem leitura melhor (n_votes maior
                # ou proposal_score maior, devido à ordenação) aqui.
                dup = True
                break

        if not dup:
            unique.append(cand)

    return unique


def annotate_image(image, results):
    """
    Desenha as detecções sobre a imagem final.

    Para cada símbolo encontrado:
        - desenha um retângulo verde
        - desenha um rótulo com índice e texto decodificado
    """
    vis = image.copy()

    for i, item in enumerate(results, start=1):
        x, y, w, h = item["box"]

        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{i}:{item['text']}"
        cv2.putText(
            vis,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def main():
    """
    Função principal do pipeline.

    Etapas:
        1. Ler argumentos
        2. Carregar imagens
        3. Fazer ortorretificação
        4. Gerar candidatos por heatmap
        5. Gerar candidatos por componentes
        6. Aplicar NMS global
        7. Decodificar candidatos
        8. Remover duplicatas
        9. Ordenar espacialmente
        10. Salvar JSON e imagem anotada
        11. Opcionalmente salvar ROIs candidatas
    """
    args = parse_args()

    # Leitura das imagens de entrada
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    template = cv2.imread(args.template, cv2.IMREAD_COLOR)

    if image is None or template is None:
        raise RuntimeError("Falha ao abrir input/template")

    # Etapa 1: ortorretificação
    ortho, _margin = build_ortho(image, template, args.margin)
    # Atualiza args.margin com o valor efetivamente usado pelo ortho (que pode
    # ter sido auto-calculado quando args.margin é None/0). propose_from_grid e
    # estimate_symbol_side dependem desse valor para alinhar a grade nominal.
    args.margin = _margin
    if not cv2.imwrite(args.output, ortho):
        raise RuntimeError("Falha ao salvar ortho")

    # Conversão para tons de cinza para facilitar as etapas seguintes
    gray_orig = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)
    oh, ow = gray_orig.shape[:2]

    # Adiciona uma borda branca artificial em volta do ortho. Isso garante que
    # símbolos colados às bordas (que aparecem em y=0 ou x=W-1, por exemplo
    # quando a maquete é fotografada rotacionada) tenham contexto suficiente
    # para o heatmap detectá-los e para o decoder ter quiet zone. As coordenadas
    # finais são corrigidas no momento de salvar o JSON e desenhar a imagem
    # anotada. O tamanho do símbolo continua sendo estimado a partir do ortho
    # original, ignorando a borda artificial — caso contrário a janela do
    # heatmap cresce e estima caixas grandes demais.
    pad = max(0, int(args.edge_pad))
    if pad > 0:
        # Branco puro cria uma borda dura preto/branco que o blackhat realça
        # como pseudo-feature, dominando o NMS perto das bordas. Replicar a
        # borda mantém a estatística local sem introduzir gradiente artificial.
        gray = cv2.copyMakeBorder(
            gray_orig, pad, pad, pad, pad,
            cv2.BORDER_REPLICATE,
        )
        ortho_padded = cv2.copyMakeBorder(
            ortho, pad, pad, pad, pad,
            cv2.BORDER_REPLICATE,
        )
    else:
        gray = gray_orig
        ortho_padded = ortho

    # Etapa 2: proposição de candidatos por múltiplas estratégias.
    # Forçamos a estimativa de tamanho de símbolo a usar as dimensões do ortho
    # original (sem o edge_pad) para que a janela do heatmap não cresça quando
    # adicionarmos a borda branca.
    raw_candidates = []
    scales = [args.proposal_scale]
    extra = [
        float(s) for s in args.proposal_scales.split(",") if s.strip()
    ]
    for s in extra:
        if all(abs(s - existing) > 1e-6 for existing in scales):
            scales.append(s)
    for scale in scales:
        # Cria uma cópia mínima do args com proposal_scale ajustada para esta
        # rodada. Mantém todos os outros parâmetros originais.
        scale_args = argparse.Namespace(**vars(args))
        scale_args.proposal_scale = scale
        raw_candidates.extend(propose_from_heatmap(gray, scale_args, ref_shape=(oh, ow)))
        raw_candidates.extend(propose_from_components(gray, scale_args, ref_shape=(oh, ow)))

    # Adiciona TODAS as 1369 caixas da grade nominal como propostas extras.
    # Necessário pra resolver misreads onde heatmap propõe caixa ligeiramente
    # off-center cujos bits decodificam para letra cosmética do allowlist
    # (caso (3,1)='H' lido como 'X' em config_4_sample_1, caso (3,26)='H'
    # lido como 'G' em config_1_sample_2). O grid candidate decodifica o
    # símbolo real e vence no dedup posicional por n_votes.
    #
    # Trade-off: adicionar todas as 1369 caixas mais que dobra o tempo de
    # decode por imagem (de ~3 min para ~10 min com a config atual), mas
    # mantém 100% de acerto no lote de teste. Filtros mais agressivos
    # (cover_tol ou std≥12) economizam tempo mas re-introduzem misreads
    # pontuais — preferimos perfeição a velocidade aqui.
    grid_cands = propose_from_grid(args, (oh, ow), pad)

    # Etapa 3: NMS global para reduzir redundância entre heatmap+components.
    # As propostas da grade NÃO passam pela NMS global — vão direto para
    # a fase de decode, e o dedup posterior (deduplicate_decoded) faz a
    # arbitragem por votos.
    candidates = nms_candidates(
        raw_candidates,
        args.nms_iou,
        args.merge_distance,
        args.max_candidates,
    )
    candidates.extend(grid_cands)

    # Monta um dicionário apenas com o necessário para os workers
    args_dict = {
        "decode_timeout": args.decode_timeout,
        "decode_shrink": args.decode_shrink,
        "decode_border": args.decode_border,
        "resize_factor": args.resize_factor,
        "use_edge_bounds": args.use_edge_bounds,
        "pad": args.pad,
        "skip_empty": args.skip_empty,
        "empty_std_threshold": args.empty_std_threshold,
        "empty_dark_threshold": args.empty_dark_threshold,
        "min_local_dark_ratio": args.min_local_dark_ratio,
    }

    payloads = [(gray, cand, args_dict) for cand in candidates]

    # Etapa 4: decodificação
    decoded = []

    if args.workers <= 1:
        # Modo sequencial
        for payload in payloads:
            r = decode_worker(payload)
            if r is not None:
                decoded.append(r)
    else:
        # Modo paralelo com processos. O libdmtx mantém estado interno não
        # projetado para acesso concorrente por threads.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(decode_worker, payloads, chunksize=args.chunksize):
                if r is not None:
                    decoded.append(r)

    # Etapa 4.5: corrige coordenadas para o sistema do ortho original
    # (descontando a borda artificial adicionada antes da proposição).
    if pad > 0:
        oh, ow = gray_orig.shape[:2]
        for r in decoded:
            x, y, bw, bh = r["box"]
            x -= pad
            y -= pad
            # Clampa caixa para dentro do ortho original — caixas que ficam
            # parcialmente fora do ortho são "encolhidas" mas mantidas.
            nx = max(0, min(ow - 1, x))
            ny = max(0, min(oh - 1, y))
            nx2 = max(nx + 1, min(ow, x + bw))
            ny2 = max(ny + 1, min(oh, y + bh))
            r["box"] = [int(nx), int(ny), int(nx2 - nx), int(ny2 - ny)]
            r["center"] = [
                float(r["center"][0] - pad),
                float(r["center"][1] - pad),
            ]

    # Etapa 5: remoção de duplicatas
    decoded = deduplicate_decoded(decoded)

    # Etapa 6: ordenação espacial (primeiro por y, depois por x)
    decoded.sort(key=lambda d: (d["center"][1], d["center"][0]))

    # Debug opcional: salvar candidatos recortados
    # As coordenadas dos candidatos estão no sistema do gray padded, então
    # recortamos do ortho_padded para garantir consistência.
    if args.dump_candidates:
        out_dir = Path(args.candidates_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, cand in enumerate(candidates):
            x, y, w, h = cand["box"]
            roi = ortho_padded[y:y + h, x:x + w]
            if roi.size:
                cv2.imwrite(
                    str(out_dir / f"cand_{idx:04d}_{cand['source']}_{cand['score']:.1f}.png"),
                    roi,
                )

    # Etapa 7: salvar saída JSON
    #
    # A foto é capturada pela face inferior da maquete, então o eixo
    # horizontal sai espelhado em relação à vista de topo real.
    # Espelhamos `box` e `center` para que o JSON saia em frame de topo
    # (alinhado com o grid.txt de pipeline.py e o maquete_objetos.json
    # do PyAppArq). A imagem anotada (etapa 8) continua em frame de foto
    # porque é diagnóstico visual sobre o ortho original — os índices
    # entre JSON e imagem anotada permanecem alinhados (mesma ordem).
    ortho_w = ortho.shape[1]
    decoded_top_view = []
    for r in decoded:
        x, y, bw, bh = r["box"]
        cx, cy = r["center"]
        flipped = dict(r)
        flipped["box"] = [int(ortho_w - x - bw), int(y), int(bw), int(bh)]
        flipped["center"] = [float(ortho_w - 1 - cx), float(cy)]
        decoded_top_view.append(flipped)

    payload = {
        "count": len(decoded_top_view),
        "candidates_considered": len(candidates),
        "symbols": decoded_top_view,
    }

    Path(args.results_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Etapa 8: salvar imagem anotada (em frame de foto, sobre o ortho original)
    annotated = annotate_image(ortho, decoded)
    if not cv2.imwrite(args.annotated_output, annotated):
        raise RuntimeError("Falha ao salvar imagem anotada")

    # Também imprime no terminal para facilitar inspeção rápida
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
