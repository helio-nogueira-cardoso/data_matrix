# ECC200Decode

Pipeline para ortorretificação, detecção e decodificação de códigos DataMatrix ECC200 a partir de imagens de uma maquete física modular.

---

# Contexto do Projeto

Este repositório integra um projeto mais amplo que combina arquitetura, sistemas físicos modulares e visão computacional, com o objetivo de converter uma maquete física configurável em uma representação digital estruturada.

A maquete é construída sobre uma base com uma grade regular de furos igualmente espaçados, permitindo o encaixe de módulos físicos que representam elementos arquitetônicos, como paredes, portas, janelas e mobiliário. A configuração espacial é definida pela disposição desses módulos na grade.

Cada elemento físico pode conter múltiplos códigos DataMatrix ECC200 (por exemplo, 2, 3 ou 4 códigos). Esses códigos não são interpretados isoladamente neste repositório. A combinação entre eles é utilizada posteriormente por um sistema responsável pela interpretação semântica dos elementos, incluindo tipo, orientação e relações espaciais.

---

# Escopo do Repositório

Este repositório contém dois componentes:

## 1. Pipelines de visão computacional (raiz)

Responsáveis por:

1. Processar imagens da face inferior da maquete
2. Corrigir distorções geométricas (ortorretificação)
3. Localizar os códigos DataMatrix na imagem
4. Decodificar os identificadores
5. Determinar suas posições relativas na grade

## 2. Aplicação gráfica PyAppArq (`PyAppArq/`)

Reescrita em Python da aplicação original AppArq (C++/Qt). Integra o pipeline de visão computacional com a interpretação semântica e exportação para BIM, incluindo:

* Interface gráfica GTK3 para carregamento de imagens
* Dois modos de operação: interativo (verificação manual) e automático
* Decodificação DataMatrix ECC200 (substituindo SIFT do sistema original)
* Interpretação semântica da grade (paredes, portas, janelas, hospedados, mobiliário)
* Exportação JSON no formato compatível com o Revit

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

A interpretação desses conjuntos (tipo, orientação e relações espaciais) é realizada pelo módulo `PyAppArq/objects_handler.py`, com base nas definições em `PyAppArq/objetos.json`.

---

# Integração com o Sistema Completo

O fluxo completo do sistema é:

```
Maquete física
    → captura com smartphone
    → ortorretificação (template matching nos 4 cantos)
    → decodificação DataMatrix (grade 37×37)
    → interpretação semântica (paredes, janelas, portas, mobiliário)
    → exportação JSON (maquete_objetos.json)
    → importação no Revit (via plugin externo)
```

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

## Aplicação gráfica PyAppArq

Diretório: `PyAppArq/`

### Descrição

Reescrita em Python da aplicação original AppArq (C++/Qt), com as seguintes mudanças:

* **Interface**: GTK3 em vez de Qt
* **Detecção de símbolos**: DataMatrix ECC200 (pylibdmtx) em vez de SIFT
* **Câmera**: removida (apenas upload de imagem de smartphone)
* **Saída**: mesmo formato JSON para integração com o Revit

### Estrutura

| Arquivo | Função |
|---------|--------|
| `main.py` | Ponto de entrada |
| `gui.py` | Interface gráfica GTK3 |
| `pipeline.py` | Ortorretificação + decodificação DataMatrix |
| `objects_handler.py` | Interpretação semântica (paredes, janelas, portas, mobiliário) |
| `objetos.json` | Definições dos elementos arquitetônicos |
| `template.png` | Template dos marcadores de canto |
| `requirements.txt` | Dependências Python |

### Modos de operação

* **Interativo**: exibe a imagem ortorretificada com os símbolos sobrepostos (vermelho = incerto, azul = decodificado, verde = corrigido pelo usuário). O usuário pode clicar em qualquer célula para corrigir o valor.
* **Automático**: processa e salva diretamente sem verificação do usuário.

### Uso

```bash
cd PyAppArq
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Saída

O projeto salvo contém:

* `maquete_objetos.json` → elementos arquitetônicos no formato Revit
* `maquete_imagem.png` → imagem original
* `maquete_grade.txt` → grade 37×37 de símbolos

---

# Instalação

## Sistema (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libdmtx0t64 libdmtx-dev
```

## Pipelines (raiz do repositório)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## PyAppArq

```bash
cd PyAppArq
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

A flag `--system-site-packages` é necessária para acessar o GTK3 (PyGObject) instalado no sistema.

---

# requirements.txt

## Raiz (pipelines)

```
pylibdmtx
Pillow
opencv-python
numpy
```

## PyAppArq

```
opencv-python>=4.5.0
numpy>=1.20.0
pylibdmtx>=0.1.10
setuptools
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

## GTK3 / PyGObject (para PyAppArq)

O PyGObject é instalado via sistema, não via pip:

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

Ao criar o venv, use `--system-site-packages` para herdar esses pacotes.

---

# Síntese do Fluxo

```
Maquete física
    → imagem (smartphone)
    → detecção de códigos (pipeline.py ou PyAppArq)
    → decodificação DataMatrix
    → interpretação semântica (PyAppArq/objects_handler.py)
    → JSON estruturado (maquete_objetos.json)
    → modelo digital (Revit via plugin externo)
```

---

# Autor

Hélio Nogueira Cardoso
