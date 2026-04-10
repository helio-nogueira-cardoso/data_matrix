# ECC200Decode

Pipeline para ortorretificação, detecção e decodificação de códigos DataMatrix ECC200 a partir de imagens de uma maquete física modular.

---

# Contexto do Projeto

Este repositório integra um projeto mais amplo que combina arquitetura, sistemas físicos modulares e visão computacional, com o objetivo de converter uma maquete física configurável em uma representação digital estruturada.

A maquete é construída sobre uma base com uma grade regular de furos igualmente espaçados, permitindo o encaixe de módulos físicos que representam elementos arquitetônicos, como paredes, portas, janelas e mobiliário. A configuração espacial é definida pela disposição desses módulos na grade.

Cada elemento físico pode conter múltiplos códigos DataMatrix ECC200 (por exemplo, 2, 3 ou 4 códigos). Esses códigos não são interpretados isoladamente neste repositório. A combinação entre eles é utilizada posteriormente por um sistema externo responsável pela interpretação semântica dos elementos, incluindo tipo, orientação e relações espaciais.

---

# Escopo do Repositório

Este repositório implementa exclusivamente a etapa de visão computacional responsável por:

1. Processar imagens da face inferior da maquete
2. Corrigir distorções geométricas (ortorretificação)
3. Localizar os códigos DataMatrix na imagem
4. Decodificar os identificadores
5. Determinar suas posições relativas na grade

Não faz parte do escopo deste repositório:

* interpretação semântica dos códigos
* identificação direta de elementos arquitetônicos
* geração do modelo arquitetônico final

Essas etapas pertencem ao sistema maior ao qual este projeto está integrado.

---

# Aquisição de Imagem

As imagens de entrada são capturadas a partir da face inferior da maquete, onde os códigos DataMatrix estão visíveis.

Testes foram realizados utilizando câmera de dispositivo móvel (por exemplo, Samsung Galaxy S23 Ultra), com captura em alta resolução (50 MP, formato RAW convertido para PNG).

Requisitos recomendados:

* resolução mínima de aproximadamente 8160 × 6120 pixels
* boa iluminação e contraste
* foco adequado nos códigos
* baixa distorção geométrica (ou corrigível via ortorretificação)

A qualidade da imagem impacta diretamente a taxa de detecção e decodificação.

---

# Papel dos Códigos DataMatrix

Os códigos DataMatrix ECC200 funcionam como identificadores primitivos. Cada elemento arquitetônico é descrito por um conjunto de códigos, e não por um único marcador.

A interpretação desses conjuntos (tipo, orientação e relações espaciais) é realizada por um módulo externo ao presente repositório.

---

# Integração com o Sistema Completo

A saída deste repositório consiste em:

* identificadores decodificados
* posições na imagem
* mapeamento aproximado na grade

Esses dados são utilizados por um sistema posterior que realiza:

* análise semântica
* reconstrução da configuração espacial
* geração de modelo arquitetônico digital

Esse modelo pode ser exportado para ferramentas como o Autodesk Revit.

---

# Visão Geral do Pipeline

O sistema realiza:

1. Ortorretificação da imagem
2. Detecção de regiões candidatas
3. Decodificação de códigos ECC200
4. Geração de saídas estruturadas

---

# Abordagens Implementadas

## Pipeline baseado em grade

Arquivo: `pipeline.py`

### Descrição

* Divide a imagem em uma grade fixa (37x37)
* Cada célula é processada individualmente
* Simples e determinístico

### Uso

```bash
python pipeline.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --grid-output grid.txt
```

### Saída

* `grid.txt` → matriz 37x37 com os símbolos detectados

### Debug

```bash
python pipeline.py --test-cell 5,6
python pipeline.py --dump-elements
```

---

## Pipeline livre (sem grade)

Arquivo: `pipeline_livre.py`

### Descrição

* Não depende de grade fixa
* Detecta automaticamente regiões contendo códigos
* Retorna posições reais (bounding boxes)
* Para evolução do projeto livre da grade igualmente espaçada

### Etapas

1. Ortorretificação
2. Proposição de candidatos:

   * heatmap local
   * componentes conectados
3. Filtragem (NMS)
4. Decodificação
5. Remoção de duplicatas

### Uso

```bash
python pipeline_livre.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --results-json symbols.json \
  --annotated-output annotated.png
```

### Saída

* `symbols.json` → lista de códigos e posições
* `annotated.png` → imagem com bounding boxes
* `candidates/` (opcional) → regiões candidatas

### Debug

```bash
python pipeline_livre.py --dump-candidates
```

---

# Instalação

## Sistema (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libdmtx0t64 libdmtx-dev
```

## Python

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

# requirements.txt

```
pylibdmtx
Pillow
opencv-python
numpy
```

---

# Problemas comuns

## libdmtx

```bash
sudo apt install libdmtx0t64 libdmtx-dev
```

## OpenCV

```bash
pip install opencv-python
```

---

# Síntese do Fluxo

Maquete física → imagem → detecção de códigos → decodificação → estruturação dos dados → modelo digital

---

# Autor

Hélio Nogueira Cardoso
