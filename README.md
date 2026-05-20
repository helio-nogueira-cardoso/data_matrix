# ECC200Decode

Pipeline para ortorretificação, detecção e decodificação de códigos DataMatrix ECC200 a partir de imagens da face inferior de uma maquete física modular. Base do projeto de TCC em Ciência da Computação (USP/ICMC).

## Contexto

A maquete é construída sobre uma base perfurada com grade regular de furos. Cada elemento arquitetônico (parede, porta, janela, móvel) é encaixado como pino na base e expõe, no lado de baixo, um código DataMatrix ECC200 colado à base. Uma foto da face inferior captura simultaneamente todos os códigos, e o pipeline aqui descrito os ortorretifica e decodifica.

A interpretação semântica dos códigos (tipo, orientação, relações espaciais) é responsabilidade do aplicativo PyAppArq (`PyAppArq/`), que envolve este pipeline e exporta um JSON consumido por um plugin externo no Revit.

## Escopo

O repositório contém:

1. **Pipelines de visão computacional** na raiz: `pipeline.py` (com grade fixa 37×37) e `pipeline_free.py` (sem grade, exploratório).
2. **Aplicação gráfica PyAppArq** em `PyAppArq/`: reimplementação em Python/GTK3 do AppArq original (C++/Qt), com decodificação DataMatrix no lugar do reconhecimento SIFT, interface de validação visual e exportação JSON para o Revit.

## Aquisição de imagem

As imagens são capturadas com smartphone, a partir da face inferior da maquete. O conjunto experimental utiliza um Samsung Galaxy S23 Ultra, mas qualquer câmera com resolução próxima a 50 MP e foco adequado deve funcionar.

Requisitos recomendados:

* Resolução próxima a 8160 × 6120 pixels (50 MP).
* Foco nítido sobre o plano dos códigos.
* Iluminação que preserve contraste local dos módulos.
* Distorção geométrica leve (corrigida na ortorretificação).
* Pinos firmemente encaixados, com o código em plano paralelo ao sensor.

### Presets de câmera por regime de iluminação

| Parâmetro | Interno diurno | Retroiluminado | Pouco iluminado |
|-----------|----------------|----------------|-----------------|
| Resolução | 50 MP | 50 MP | 50 MP |
| Formato | RAW | RAW | RAW |
| Lente | W (grande angular) | W (grande angular) | W (grande angular) |
| Abertura | f/1,7 | f/1,7 | f/1,7 |
| ISO | Automático | Automático | 12 |
| Velocidade | 1/8 s | 2 s | 8 s |
| Compensação (EV) | 0,0 | −0,6 | −0,5 |
| Foco | AF-M (múltiplo) | AF-M (múltiplo) | AF-M (múltiplo) |
| Balanço de branco | Automático | Automático | Automático |

O preset para ambiente pouco iluminado pressupõe câmera apoiada em superfície estável ou tripé, dada a exposição de 8 segundos.

## Vocabulário do projeto

A allowlist de letras válidas vive em `symbols_config.json` (artefato de produção, somente leitura em runtime). Os três pipelines carregam o arquivo na inicialização e rejeitam qualquer leitura fora do vocabulário. Esse filtro elimina corrupções típicas do `libdmtx` em tiles limítrofes, que ocasionalmente retornam letras cosméticas (`N`, `I`) ou sequências com caracteres de controle.

```json
{
  "version": 1,
  "max_length": 4,
  "vocabulary": [
    "A","B","C","E","F","G","H","J",
    "M","O","P","R","S","T","V",
    "h"
  ]
}
```

A edição manual desse arquivo é a única forma de alterar o conjunto aceito. Não há flag de linha de comando equivalente, por desenho: o vocabulário descreve o projeto, não a execução.

## Princípio: preservação da orientação original

Cada imagem é decodificada na orientação em que foi capturada. Os pipelines não rotacionam tile, ortho ou imagem como fallback. Apesar de o ECC200 ser nominalmente invariante a rotação, o `libdmtx` ocasionalmente retorna resultados distintos para versões rotacionadas do mesmo tile em casos limítrofes; explorar isso geraria leituras potencialmente erradas em produção, onde não há gabarito para validar. As recuperações vêm de variações legítimas:

* **Cinco pré-processamentos** (Otsu, Otsu sobre bilateral, limiar adaptativo gaussiano, realce unsharp, escala de cinza com borda) decididos por voto majoritário entre os candidatos.
* **Multi-crop central** (75 %, 65 %, 55 % do lado nominal) quando todas as variantes falham em um tile com indícios de conteúdo.
* **Allowlist do projeto** descartando leituras fora do vocabulário antes da contagem de votos.
* **Filtro anti-fantasma**: vencedores com um voto contra duas ou mais rejeições do vocabulário viram célula vazia.

## Pipelines

### `pipeline.py`: baseado em grade 37×37

A imagem é ortorretificada por homografia. Os quatro marcadores de canto em cruz são localizados por correlação cruzada normalizada (`cv2.matchTemplate` com `TM_CCOEFF_NORMED`) e validados por descritores de forma sobre a componente conexa escura central: razão de aspecto entre 0,85 e 1,15, extensão entre 0,35 e 0,46 e solidez abaixo de 0,70. Essa assinatura geométrica descarta os falsos positivos típicos do `matchTemplate` em imagens reais (constelações de quatro perfurações brilhantes, juntas perpendiculares de moldura, padrões lineares).

A imagem ortorretificada é segmentada em uma grade fixa de 37×37 células, totalizando 1369 posições. Antes do pré-processamento, cada caixa de célula é realinhada pela posição do componente conexo escuro mais próximo do centro nominal, com salvaguardas que impedem deslocamento em tiles efetivamente vazios. Cada célula gera as cinco variantes de pré-processamento, recebe borda branca artificial que simula a quiet zone exigida pelo padrão ISO/IEC 16022 e é decodificada pela `pylibdmtx`. O voto majoritário com filtro anti-fantasma consolida as leituras.

A margem do ortho é calculada automaticamente como o teto da metade do menor passo da grade, garantindo que toda célula, inclusive as de borda, receba recorte de tamanho nominal com o símbolo centralizado.

Parâmetros opcionais:

* `--skip-empty`: descarta tiles com baixo desvio-padrão ou poucos pixels escuros antes da decodificação (desligado por padrão; economiza ~10 % de tempo mas pode degradar o recall em pegs translúcidos).
* `--refine-fallback` (ligado por padrão): aplica o realinhamento da caixa apenas em células que falharam na primeira passada.
* `--heatmap-fallback` (desligado por padrão): para imagens com cruzes em condições adversas, aplica a proposição global do `pipeline_free.py` às células remanescentes.

Uso:

```bash
python pipeline.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --grid-output grid.txt
```

Saída:

* `grid.txt`: matriz 37×37 com os símbolos detectados (`_` em células vazias), em frame de topo (alinhado com o JSON do PyAppArq e a vista superior real da maquete).

Debug:

```bash
python pipeline.py --test-cell 5,6     # roda só uma célula
python pipeline.py --dump-elements     # salva todos os tiles em ./elements/
```

### `pipeline_free.py`: sem grade fixa

Localiza os códigos diretamente na imagem ortorretificada, sem assumir a estrutura discreta da grade. Combina três fontes de proposição de candidatos:

1. **Heatmap local de contraste**: soma ponderada do desvio-padrão local em janela e da transformação black top-hat morfológica sobre versão suavizada da imagem, em múltiplas escalas.
2. **Componentes conectados** sobre mapas binários, com filtro por área e razão de aspecto.
3. **Grade nominal 37×37** como rede de segurança, recuperando casos em que o heatmap propõe caixa levemente deslocada que decodifica para letra cosmética.

Os candidatos passam por supressão não-máxima por IoU e distância entre centros, e cada um é decodificado pelo mesmo voto majoritário das cinco variantes do pipeline com grade. Multi-crop central a 75 %, 65 % e 55 % atua como fallback quando o crop nominal não fecha. A deduplicação final usa `(n_votos, score)` como chave de desempate.

Uso:

```bash
python pipeline_free.py \
  --input imagem.png \
  --template template.png \
  --output ortho.png \
  --results-json symbols.json \
  --annotated-output annotated.png
```

Saída:

* `symbols.json`: lista de detecções com `text`, `box`, `center` e `n_votes`, em coordenadas do ortho e frame de topo.
* `annotated.png`: ortho com retângulos verdes e rótulos por detecção (frame de foto).
* `candidates/` com `--dump-candidates`: regiões candidatas antes da decodificação.

### `PyAppArq/`: aplicação gráfica

Reimplementação Python/GTK3 do AppArq original (C++/Qt), com decodificação DataMatrix no lugar do reconhecimento por SIFT. Encapsula o pipeline com grade e adiciona interpretação semântica e exportação JSON para o Revit.

| Arquivo | Função |
|---------|--------|
| `main.py` | Ponto de entrada |
| `gui.py` | Interface gráfica GTK3 |
| `pipeline.py` | Ortorretificação e decodificação DataMatrix |
| `objects_handler.py` | Parser semântico (paredes, janelas, portas, hospedados, mobiliário) |
| `objetos.json` | Catálogo dos elementos arquitetônicos |
| `template.png` | Gabarito dos marcadores de canto |

Modos de operação:

* **Interativo**: após o processamento, abre uma tela de validação visual com cada símbolo sobreposto à célula correspondente. O usuário pode clicar em qualquer célula para abrir uma janela modal de correção (zoom do recorte + campo de entrada). O projeto só é salvo após confirmação.
* **Automático**: aceita diretamente todas as leituras e salva o projeto sem verificação.

Uso (com a venv da raiz ativa):

```bash
cd PyAppArq
python main.py
```

Saída do projeto:

* `maquete_objetos.json`: elementos arquitetônicos no formato consumido pelo plugin Revit, com chaves `WallProperties`, `WindowProperties`, `DoorProperties`, `HostedProperties` e `FurnitureProperties`.
* `maquete_imagem.png`: imagem original.
* `maquete_grade.txt`: grade 37×37 de símbolos, em frame de topo.

## Validação contra gabarito

O script `tests/validate_baseline.py` é o ponto único de verificação dos pipelines contra um gabarito de referência. Lê `tests/expected_baseline.json`, roda cada pipeline em subprocesso e compara o multiset de letras decodificadas com o esperado.

```bash
python tests/validate_baseline.py --pipeline free      # só pipeline_free.py
python tests/validate_baseline.py --pipeline grid      # só pipeline.py
python tests/validate_baseline.py --pipeline pyapparq  # só PyAppArq via API
python tests/validate_baseline.py --pipeline both      # grid + free (default)
python tests/validate_baseline.py --pipeline all       # todos
```

O script imprime uma linha por imagem com `count`, `missing`, `extra` e marca `[OK]` ou `[FAIL]`, retornando `0` quando tudo bate.

## Instalação

Todo o projeto usa uma única venv e um único `requirements.txt`, ambos na raiz.

### Dependências de sistema (Debian/Ubuntu/Mint)

O nome do pacote runtime do `libdmtx` mudou entre versões: Debian 12+, Ubuntu 24.04+ e Mint 22+ usam `libdmtx0t64`; Ubuntu 22.04 e Mint 21 usam `libdmtx0b`; versões anteriores usam `libdmtx0a`. O comando abaixo detecta automaticamente o nome correto:

```bash
sudo apt update

LIBDMTX=$(apt-cache search '^libdmtx0' | awk '{print $1}' | head -1)

sudo apt install -y \
  python3 python3-venv python3-pip \
  "$LIBDMTX" libdmtx-dev \
  python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

Significado dos pacotes:

* `libdmtx0*` e `libdmtx-dev`: biblioteca nativa usada pelo `pylibdmtx`. Os dois são necessários: o runtime para carregar a `.so` em tempo de execução e o `-dev` para o pip linkar ao instalar o `pylibdmtx`.
* `python3-gi`, `python3-gi-cairo`, `gir1.2-gtk-3.0`: GTK3 e PyGObject para a interface do PyAppArq (não instaláveis via pip).

### Venv única

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --ignore-installed -r requirements.txt
```

* `--system-site-packages` permite que a venv enxergue o PyGObject/GTK3 do sistema.
* `--ignore-installed` força a instalação dentro da venv mesmo quando os pacotes já existirem em `~/.local`, mantendo a venv portável.

Depois da ativação, tanto `python pipeline.py ...` na raiz quanto `cd PyAppArq && python main.py` usam o mesmo ambiente.

### `requirements.txt`

```
opencv-python>=4.5.0
numpy>=1.20.0
pylibdmtx>=0.1.10
Pillow
setuptools
```

## Problemas comuns

### `libdmtx`: `Unable to locate package libdmtx0t64`

Mensagem típica em Linux Mint 21 ou Ubuntu 22.04 (ambos baseados em Ubuntu Jammy). O pacote runtime nessas versões é `libdmtx0b`. Descubra qual está disponível:

```bash
apt-cache search '^libdmtx0'
```

E instale o que for listado, junto com `libdmtx-dev`:

```bash
sudo apt install <nome-do-runtime> libdmtx-dev
```

### `pylibdmtx` instala mas trava na primeira chamada

Falta o runtime do `libdmtx`; apenas `libdmtx-dev` foi instalado. Reinstale o runtime:

```bash
sudo apt install --reinstall <nome-do-runtime>
```

### OpenCV

```bash
pip install opencv-python
```

Em Mint, se `import cv2` falhar com `libGL.so.1: cannot open shared object`:

```bash
sudo apt install libgl1
```

### GTK3 e PyGObject (para PyAppArq)

O PyGObject não é instalado via pip; deve vir do sistema:

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

A venv é criada com `--system-site-packages` justamente para herdar esses pacotes. Se a GUI do PyAppArq falhar com `ModuleNotFoundError: No module named 'gi'`, recrie a venv com essa flag.

### Dependências Python "sumindo" na venv

Se o `pip install` reportar "Requirement already satisfied" para tudo sem baixar nada, o pip encontrou os pacotes em `~/.local/lib/python3.*/site-packages` por causa do `--system-site-packages`. A venv parece vazia e não é portável. Reinstale forçando:

```bash
pip install --ignore-installed -r requirements.txt
```

### Mint: `pip install` reclama de `externally-managed-environment`

Mint 22 herda do Ubuntu 24.04 a política PEP 668 que bloqueia `pip` fora de venv. Dentro da venv (com `source .venv/bin/activate`) o erro não deve aparecer. Confirme que o `python` ativo é o da venv:

```bash
which python    # deve apontar para .venv/bin/python
```

## Síntese do fluxo

```
Maquete física
    → captura com smartphone (face inferior)
    → ortorretificação por homografia
    → decodificação DataMatrix (grade 37×37 ou abordagem sem grade)
    → interpretação semântica (PyAppArq)
    → JSON estruturado (maquete_objetos.json)
    → modelo BIM no Revit (plugin externo)
```

## Autor

Hélio Nogueira Cardoso
