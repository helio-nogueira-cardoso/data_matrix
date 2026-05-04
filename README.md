# ECC200Decode

Pipeline para ortorretificação, detecção e decodificação de códigos DataMatrix ECC200 a partir de imagens de uma maquete física modular. Base para projeto de TCC.

---

# Contexto do Projeto

Este projeto integra um trabalho mais amplo que combina arquitetura, sistemas físicos modulares e visão computacional, com o objetivo de converter uma maquete física configurável em uma representação digital estruturada.

A maquete é construída sobre uma base com uma grade regular de furos igualmente espaçados, permitindo o encaixe de módulos físicos que representam elementos arquitetônicos, como paredes, portas, janelas e mobiliário. A configuração espacial é definida pela disposição desses módulos na grade.

Cada elemento físico pode conter múltiplos códigos DataMatrix ECC200 (por exemplo, 2, 3 ou 4 códigos). Esses códigos não são interpretados isoladamente neste diretório. A combinação entre eles é utilizada posteriormente por um sistema responsável pela interpretação semântica dos elementos, incluindo tipo, orientação e relações espaciais.

---

# Escopo do Projeto

Este projeto contém dois componentes:

## 1. Pipelines de visão computacional (raiz)

Responsáveis por:

1. Processar imagens da face inferior da maquete
2. Corrigir distorções geométricas (ortorretificação)
3. Localizar os códigos DataMatrix na imagem
4. Decodificar os identificadores
5. Determinar suas posições relativas na grade

## 2. Aplicação gráfica PyAppArq (`PyAppArq/`)

Subprojeto Python do AppArq original (C++/Qt), mantido dentro deste diretório e compartilhando a mesma venv. Integra o pipeline de visão computacional com a interpretação semântica e exportação para BIM, incluindo:

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

# Validador de produção

A allowlist de letras válidas vive em `symbols_config.json` (artefato de
produção, somente leitura em runtime). Os três pipelines carregam esse
arquivo no carregamento do módulo e o passam para `looks_like_valid_symbol`,
que rejeita qualquer leitura fora do vocabulário. Isso filtra corrupções
típicas do `libdmtx` em tiles limítrofes — ele às vezes devolve letras
cosméticas como `'N'`, `'I'` ou strings com caracteres de controle (`'H63E'`,
`'P07\\x05'`) que escapariam de uma checagem só por `isalpha()`.

```json
{
  "version": 1,
  "max_length": 4,
  "vocabulary": [
    "A","B","C","E","F","G","H","J",
    "M","O","P","R","S","T","V","X",
    "h"
  ]
}
```

Editar o arquivo é a única forma de mudar o conjunto de símbolos aceitos —
não há flag de linha de comando para isso, deliberadamente: o vocabulário
descreve o projeto, não a execução.

## Princípio: nada de rotação como fallback

Cada imagem é decodificada na orientação em que foi capturada. Os pipelines
**não** rotacionam tile, ortho ou imagem como fallback para "fishar" leituras
do `libdmtx`. Apesar de o ECC200 ser nominalmente invariante a rotação, o
`libdmtx` ocasionalmente devolve resultados diferentes para versões
rotacionadas de um mesmo tile binarizado em casos limítrofes — explorar isso
seria assumir que a foto pode ter "qualquer" orientação e, na prática,
geraria leituras potencialmente erradas em produção, onde não há gabarito
para validar. As recuperações vêm exclusivamente de variações *legítimas*:

* **Pré-processamentos diferentes** (Otsu, bilateral, adaptativo, sharpening,
  escala de cinza com borda) decididos por **voto majoritário** entre os 5
  candidatos, em vez de cascade-com-early-exit. Cobre o caso de o `otsu` ler
  uma letra errada mas válida (no allowlist) em tiles onde CLAHE+sharpen
  flipa bits dentro do símbolo.
* **Multi-crop** — crops centrais com lados menores (75 %, 65 %, 55 %)
  como recurso quando todos os pré-processamentos falham num tile que
  ainda parece ter conteúdo real.
* **Allowlist** — rejeição de leituras fora do vocabulário do projeto.

---

# Abordagens Implementadas

## Pipeline baseado em grade

Arquivo: `pipeline.py`

### Descrição

* Divide a imagem em uma grade fixa 37×37 cuja geometria espelha o
  espaçamento regular da base perfurada.
* Cada célula passa por 5 pré-processamentos
  (`otsu`, `otsu_bil`, `adaptive`, `sharp`, `gray`) com **voto majoritário**.
  Caminho rápido: se `otsu` e `otsu_bil` concordam numa leitura válida pelo
  allowlist, ela vence sem rodar o resto. Caso contrário, roda os outros
  três e desempata pela leitura mais frequente (ordem do cascade como
  tiebreaker em empate). Isso evita que o `otsu` curto-circuite o cascade
  com leitura errada-mas-válida em tiles onde CLAHE+sharpen flipa bits
  dentro do símbolo (overshoot nas bordas dos módulos).
* A margem do ortho é calculada automaticamente como `ceil(passo/2)` do
  menor lado da grade. Como os marcadores de canto ficam centralizados
  nos furos extremos, isso garante que toda célula — inclusive as de
  borda — tenha recorte do tamanho nominal com o símbolo centralizado.
  Não há mais necessidade de votação de borda ou padding auxiliar para
  resolver clamping na margem.
* Quando todos os pré-processamentos canônicos falham e o tile parece ter
  conteúdo real (`std ≥ 18`), a decodificação tenta crops centrais a 75 %,
  65 % e 55 % do lado da célula. Esse multi-crop recupera símbolos cuja
  vizinhança ruidosa atrapalha sem precisar rotacionar nada.

### Uso

```bash
python pipeline.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --grid-output grid.txt
```

### Saída

* `grid.txt` → matriz 37×37 com os símbolos detectados (`?` em cada célula
  vazia), em **frame de topo** — colunas espelhadas em relação à foto, de
  modo que o arquivo bate com a vista superior real da maquete e com o
  JSON do PyAppArq. Os modos de debug (`--test-cell`, `--dump-elements`)
  continuam usando coordenadas em frame de foto, porque operam diretamente
  sobre o ortho — para localizar uma célula vista no `grid.txt` em modo
  debug, use `(linha, COLS-1-coluna)`.

### Debug

```bash
python pipeline.py --test-cell 5,6        # roda só uma célula e dump dos preprocessings
python pipeline.py --dump-elements        # salva todos os tiles recortados em ./elements/
```

---

## Pipeline livre (sem grade)

Arquivo: `pipeline_free.py`

### Descrição

* Não depende de grade fixa.
* Detecta automaticamente regiões contendo códigos via heatmap de
  contraste/estrutura local + componentes conectados.
* Retorna posições reais (bounding boxes) de cada símbolo encontrado.
* É a base para evoluir o projeto além da grade igualmente espaçada.

### Etapas

1. Ortorretificação (homografia a partir dos 4 marcadores em cruz nos
   cantos da maquete).
2. Padding lateral do ortho com `BORDER_REPLICATE` (`--edge-pad`, default
   80 px) para dar quiet zone aos símbolos colados nas bordas.
3. Proposição de candidatos:
   * heatmap local (variância + black-hat morfológico)
   * componentes conectados sobre múltiplos mapas binários
4. NMS global por IoU + distância entre centros.
5. Decodificação com cascata de pré-processamentos (idem grid).
6. Multi-crop (pad apertado) como fallback legítimo para tiles cujo
   crop nominal não fechou e que ainda parecem ter conteúdo.
7. Deduplicação por texto + proximidade.
8. Ordenação espacial (linha → coluna).

### Uso

```bash
python pipeline_free.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --results-json symbols.json \
  --annotated-output annotated.png
```

### Saída

* `symbols.json` → lista de códigos com `text`, `box` e `center` em
  coordenadas do ortho, com o eixo `x` espelhado para sair em **frame
  de topo** (alinhado com `pipeline.py` e PyAppArq). A ordem dos
  símbolos preserva a ordenação espacial original (linha → coluna na
  foto), então os índices batem com os da imagem anotada.
* `annotated.png` → ortho com retângulo verde + rótulo `i:texto` para cada
  detecção, em **frame de foto** (sobre o ortho original, sem flip).
  Use os índices para correlacionar com o JSON.
* `candidates/` (opcional, com `--dump-candidates`) → regiões candidatas
  recortadas, antes da decodificação. Coordenadas em frame de foto.

### Debug

```bash
python pipeline_free.py --dump-candidates  # despeja todas as ROIs candidatas
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

As dependências Python do PyAppArq são as mesmas do restante do projeto e estão consolidadas no `requirements.txt` da raiz.

### Modos de operação

* **Interativo**: exibe a imagem ortorretificada com os símbolos sobrepostos (vermelho = incerto, azul = decodificado, verde = corrigido pelo usuário). O usuário pode clicar em qualquer célula para corrigir o valor.
* **Automático**: processa e salva diretamente sem verificação do usuário.

### Uso

Com a `.venv` da raiz ativa (ver seção **Instalação**):

```bash
cd PyAppArq
python main.py
```

### Saída

O projeto salvo contém:

* `maquete_objetos.json` → elementos arquitetônicos no formato Revit
* `maquete_imagem.png` → imagem original
* `maquete_grade.txt` → grade 37×37 de símbolos

### Frame de coordenadas

A foto é capturada da face inferior da maquete, então o eixo horizontal
sai espelhado em relação à vista de topo real. Antes da análise semântica,
o grid é espelhado horizontalmente (`grid = [row[::-1] for row in grid]`),
de modo que `maquete_objetos.json` e `maquete_grade.txt` saem em **frame
de topo** — compatível com o que o plugin do Revit espera e alinhado com
os outros pipelines do projeto. Apenas a `maquete_imagem.png` continua em
frame de foto, casando com a renderização do canvas da GUI (que mostra a
imagem como capturada). Para correlacionar célula da GUI com posição
no `maquete_grade.txt`, use `coluna_grade = COLS - 1 - coluna_GUI`.

---

# Validação contra um gabarito

`tests/validate_baseline.py` é o script canônico para verificar os pipelines
contra um gabarito. Ele lê `tests/expected_baseline.json` (que vive separado
do `symbols_config.json` justamente porque é específico de um lote de fotos
e não do projeto), roda os pipelines em subprocesso e compara o `Counter`
decodificado com o multiset esperado.

```bash
python tests/validate_baseline.py --pipeline free      # só pipeline_free.py
python tests/validate_baseline.py --pipeline grid      # só pipeline.py (grade)
python tests/validate_baseline.py --pipeline pyapparq  # só PyAppArq via API
python tests/validate_baseline.py --pipeline both      # grid + free (default)
python tests/validate_baseline.py --pipeline all       # todos os três
```

O script imprime uma linha por imagem com `count`, `missing`, `extra` e
`[OK]/[FAIL]`, e devolve exit code `0` se tudo bater. O `expected_baseline.json`
do lote atual cobre 8 imagens (4 fotos + 4 ortos rotacionados em 90°/180°/270°)
e exige 42 letras por imagem. Para um lote novo, basta criar outro
`expected_*.json` ao lado e passar com `--baseline`.

---

# Instalação

Todo o projeto — pipelines da raiz **e** PyAppArq — usa uma única venv e um único `requirements.txt`, ambos na raiz do diretório.

## 1. Dependências de sistema (Debian/Ubuntu/Mint)

O nome do pacote do `libdmtx` mudou entre versões. O comando abaixo escolhe
o que estiver disponível na sua distribuição (Debian 12+, Ubuntu 24.04+,
Mint 22+ usam `libdmtx0t64`; Ubuntu 22.04, Mint 21 usam `libdmtx0b`;
versões mais antigas usam `libdmtx0a`):

```bash
sudo apt update

# detecta o nome certo do libdmtx runtime na distro atual
LIBDMTX=$(apt-cache search '^libdmtx0' | awk '{print $1}' | head -1)

sudo apt install -y \
  python3 python3-venv python3-pip \
  "$LIBDMTX" libdmtx-dev \
  python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

Se preferir instalar manualmente, use um destes nomes para o runtime do libdmtx:
`libdmtx0t64` (Debian 12+, Ubuntu 24.04+, Mint 22+), `libdmtx0b` (Ubuntu 22.04, Mint 21),
ou `libdmtx0a` (versões mais antigas).

Significado dos pacotes:

* `libdmtx0t64` (ou `libdmtx0a`/`libdmtx0b`) e `libdmtx-dev`: biblioteca nativa
  usada pelo `pylibdmtx`. **Os dois** são necessários — o runtime para
  carregar a `.so` em tempo de execução e o `-dev` para o pip conseguir
  fazer linkagem ao instalar o `pylibdmtx`.
* `python3-gi` / `python3-gi-cairo` / `gir1.2-gtk-3.0`: GTK3 e PyGObject
  para a interface do PyAppArq (não instaláveis via pip).

## 2. Venv única na raiz do projeto

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --ignore-installed -r requirements.txt
```

Notas:

* A flag `--system-site-packages` permite que a venv enxergue o PyGObject/GTK3 do sistema (necessário para o PyAppArq).
* A flag `--ignore-installed` força a instalação das dependências Python *dentro* da venv mesmo quando já existirem em `~/.local`, mantendo a venv portável.
* Depois da ativação, tanto `python pipeline.py ...` na raiz quanto `cd PyAppArq && python main.py` usam o mesmo ambiente.

## requirements.txt

```
opencv-python>=4.5.0
numpy>=1.20.0
pylibdmtx>=0.1.10
Pillow
setuptools
```

---

# Problemas comuns

## libdmtx — `Unable to locate package libdmtx0t64`

Mensagem típica em Linux Mint 21 ou Ubuntu 22.04 (ambos baseados em
Ubuntu Jammy). O nome do pacote runtime nessas versões é `libdmtx0b`,
não `libdmtx0t64`. Descubra qual está disponível:

```bash
apt-cache search '^libdmtx0'
```

E instale o que for listado, junto com `libdmtx-dev`:

```bash
sudo apt install <nome-do-runtime> libdmtx-dev
```

## libdmtx — `pylibdmtx` instala mas trava na primeira chamada

Significa que o runtime (`libdmtx0t64`/`0b`/`0a`) está faltando — só
`libdmtx-dev` foi instalado. O `dev` traz só os headers, não a `.so`
carregada em runtime. Reinstale o runtime:

```bash
sudo apt install --reinstall <nome-do-runtime>
```

## OpenCV

```bash
pip install opencv-python
```

Em Mint, se `import cv2` falhar com `libGL.so.1: cannot open shared object`,
instale a dependência nativa:

```bash
sudo apt install libgl1
```

## GTK3 / PyGObject (para PyAppArq)

O PyGObject não é instalado via pip, e sim via sistema:

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

A venv é criada com `--system-site-packages` justamente para herdar esses pacotes do sistema. Se a GUI do PyAppArq falhar com `ModuleNotFoundError: No module named 'gi'`, recrie a venv com essa flag.

## Dependências Python "sumindo" na venv

Se você criou a venv e o `pip install` disse "Requirement already satisfied" para tudo (sem baixar nada), significa que o pip encontrou os pacotes em `~/.local/lib/python3.*/site-packages` por causa do `--system-site-packages`. A venv parece vazia e não é portável. Reinstale forçando:

```bash
pip install --ignore-installed -r requirements.txt
```

## Mint: `pip install` reclama de `externally-managed-environment`

Mint 22 herda do Ubuntu 24.04 a política PEP 668 que bloqueia `pip` fora
de venv. **Dentro** da venv (com `source .venv/bin/activate`) o erro não
deve aparecer. Se aparecer, confirme que o `python` ativo é o da venv:

```bash
which python    # deve apontar para .venv/bin/python
```

Se precisar instalar algo fora da venv, use `pipx` ou `--break-system-packages`
— mas pra este projeto, basta sempre estar dentro da venv.

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
