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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Funções importadas do pipeline original:
# - build_ortho: gera a imagem ortorretificada
# - build_candidates_and_bounds: monta variações de ROI para decodificação
# - try_decode_text: tenta decodificar um DataMatrix
# - tile_looks_empty: tenta identificar ROIs vazias
from pipeline import build_ortho, build_candidates_and_bounds, try_decode_text, tile_looks_empty

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
    p.add_argument("--margin", type=int, default=60)
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
    p.add_argument("--decode-timeout", type=int, default=2000)
    p.add_argument("--decode-shrink", type=int, default=2)
    p.add_argument("--decode-border", type=int, default=10)
    p.add_argument("--resize-factor", type=float, default=2.0)
    p.add_argument("--use-edge-bounds", action="store_true")

    # =========================
    # Paralelismo
    # =========================
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--chunksize", type=int, default=8)

    # =========================
    # Proposição de candidatos
    # =========================
    p.add_argument("--proposal-scale", type=float, default=0.70)
    p.add_argument("--max-candidates", type=int, default=20000)
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


def estimate_symbol_side(gray, margin, rows, cols):
    """
    Estima o lado típico de um símbolo na imagem ortorretificada.

    Embora o pipeline livre não percorra uma grade fixa, ele ainda usa
    a densidade esperada de linhas/colunas como dica para estimar o tamanho
    aproximado dos símbolos.

    Isso ajuda a escolher janelas plausíveis na etapa de proposição.
    """
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


def propose_from_heatmap(gray, args):
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
    est_side_full = estimate_symbol_side(gray, args.margin, args.rows, args.cols)
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


def propose_from_components(gray, args):
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
    est_side_small = estimate_symbol_side(gray, args.margin, args.rows, args.cols) * scale
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


def decode_candidate_from_box(gray, box, args):
    """
    Tenta decodificar um único candidato.

    Fluxo:
        1. Expande a caixa para adicionar contexto
        2. Descarta regiões vazias
        3. Gera versões pré-processadas da ROI
        4. Tenta decodificar cada versão
        5. Retorna o primeiro sucesso

    Retorno:
        Dicionário com texto, método, caixa e centro,
        ou None se não conseguir decodificar.
    """
    h, w = gray.shape[:2]

    # Padding dinâmico:
    # cresce junto com o símbolo, mas nunca fica menor que args.pad
    dynamic_pad = max(args.pad, int(0.12 * min(box[2], box[3])))

    x, y, bw, bh = expand_box(box, dynamic_pad, w, h)
    tile_gray = gray[y:y + bh, x:x + bw]

    if tile_gray.size == 0:
        return None

    # Descarte rápido de ROIs vazias
    if args.skip_empty and tile_looks_empty(
        tile_gray,
        args.empty_std_threshold,
        args.empty_dark_threshold,
    ):
        return None

    local_dark_ratio = float(np.mean(tile_gray < 180)) if tile_gray.size else 0.0
    if local_dark_ratio < args.min_local_dark_ratio:
        return None

    # Gera imagens candidatas (gray, otsu, sharp etc.) e limites auxiliares
    candidate_imgs, min_edge, max_edge = build_candidates_and_bounds(
        tile_gray,
        border=args.decode_border,
        resize_factor=args.resize_factor,
        use_edge_bounds=args.use_edge_bounds,
    )

    # Tenta decodificar cada pré-processamento
    for method, img in candidate_imgs:
        text = try_decode_text(
            img,
            timeout=args.decode_timeout,
            shrink=args.decode_shrink,
            min_edge=min_edge,
            max_edge=max_edge,
        )
        if text is not None:
            return {
                "text": text,
                "method": method,
                "box": [int(x), int(y), int(bw), int(bh)],
                "center": [float(x + bw / 2.0), float(y + bh / 2.0)],
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


def deduplicate_decoded(results, iou_threshold=0.22, center_threshold=12):
    """
    Remove duplicatas das detecções já decodificadas.

    Uma detecção é considerada duplicata quando:
        - o texto é igual
        - e as caixas são suficientemente próximas / sobrepostas

    A ordenação por proposal_score garante que candidatos melhores tendam
    a ser mantidos primeiro.
    """
    unique = []

    for cand in sorted(results, key=lambda d: d.get("proposal_score", 0.0), reverse=True):
        dup = False

        for prev in unique:
            same_text = cand["text"] == prev["text"]
            near = (
                box_iou(tuple(cand["box"]), tuple(prev["box"])) > iou_threshold
                or center_distance(tuple(cand["box"]), tuple(prev["box"])) <= center_threshold
            )
            if same_text and near:
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
    ortho = build_ortho(image, template, args.margin)
    if not cv2.imwrite(args.output, ortho):
        raise RuntimeError("Falha ao salvar ortho")

    # Conversão para tons de cinza para facilitar as etapas seguintes
    gray = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)

    # Etapa 2: proposição de candidatos por múltiplas estratégias
    raw_candidates = []
    raw_candidates.extend(propose_from_heatmap(gray, args))
    raw_candidates.extend(propose_from_components(gray, args))

    # Etapa 3: NMS global para reduzir redundância entre as duas famílias
    candidates = nms_candidates(
        raw_candidates,
        args.nms_iou,
        args.merge_distance,
        args.max_candidates,
    )

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
        # Modo paralelo
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(decode_worker, payloads, chunksize=args.chunksize):
                if r is not None:
                    decoded.append(r)

    # Etapa 5: remoção de duplicatas
    decoded = deduplicate_decoded(decoded)

    # Etapa 6: ordenação espacial (primeiro por y, depois por x)
    decoded.sort(key=lambda d: (d["center"][1], d["center"][0]))

    # Debug opcional: salvar candidatos recortados
    if args.dump_candidates:
        out_dir = Path(args.candidates_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, cand in enumerate(candidates):
            x, y, w, h = cand["box"]
            roi = ortho[y:y + h, x:x + w]
            if roi.size:
                cv2.imwrite(
                    str(out_dir / f"cand_{idx:04d}_{cand['source']}_{cand['score']:.1f}.png"),
                    roi,
                )

    # Etapa 7: salvar saída JSON
    payload = {
        "count": len(decoded),
        "candidates_considered": len(candidates),
        "symbols": decoded,
    }

    Path(args.results_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Etapa 8: salvar imagem anotada
    annotated = annotate_image(ortho, decoded)
    if not cv2.imwrite(args.annotated_output, annotated):
        raise RuntimeError("Falha ao salvar imagem anotada")

    # Também imprime no terminal para facilitar inspeção rápida
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
