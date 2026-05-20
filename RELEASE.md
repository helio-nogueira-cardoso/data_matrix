# Releases multiplataforma

Este repositório publica binários standalone via GitHub Actions sempre que
uma tag `vX.Y.Z` é criada. Os binários são gerados em quatro plataformas
(Linux x86_64, Windows x86_64, macOS ARM64, macOS x86_64) para os três
entrypoints do projeto:

| Binário | O que é |
|---|---|
| `pipeline` | CLI do pipeline com grade fixa (37×37). |
| `pipeline_free` | CLI do pipeline sem grade fixa, com proposição por heatmap. |
| `PyAppArq` | Aplicativo gráfico GTK3 que envolve o pipeline com grade, com modo interativo de validação visual e exportação JSON para o Revit. |

## Como publicar uma release

```bash
git tag v0.1.0
git push origin v0.1.0
```

O push da tag dispara o workflow `.github/workflows/release.yml`. Quando os
12 builds terminam (3 binários × 4 plataformas), o workflow cria
automaticamente uma Release no GitHub com todos os artefatos anexados.

Para disparar o workflow manualmente sem criar tag (apenas para testar a
matriz de build): aba **Actions → release → Run workflow**. Esse modo não
publica Release, só sobe os artefatos para inspeção.

## Artefatos gerados

Por plataforma, o release contém:

```
pipeline-<plataforma>[.exe]
pipeline_free-<plataforma>[.exe]
PyAppArq-<plataforma>.zip
```

Onde `<plataforma>` é uma das: `linux-x86_64`, `windows-x86_64`, `macos-arm64`,
`macos-x86_64`. O sufixo `.exe` aparece apenas em Windows. A GUI é sempre
distribuída como `.zip` porque seu conteúdo é um diretório (`PyAppArq/` em
Linux/Windows; `PyAppArq.app` em macOS), não um único arquivo.

## Como testar o build localmente

Antes de criar uma release, recomenda-se gerar os binários da plataforma
local e testá-los movidos para fora do repositório (para flagrar dependências
escondidas em caminhos do código fonte).

### Linux

```bash
sudo apt install libdmtx0t64 libdmtx-dev python3-gi gir1.2-gtk-3.0
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pyinstaller
pyinstaller --clean --noconfirm pipeline.spec
pyinstaller --clean --noconfirm pipeline_free.spec
( cd PyAppArq && pyinstaller --clean --noconfirm PyAppArq.spec )
```

### macOS

```bash
brew install libdmtx gtk+3 pygobject3 pkg-config
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pyinstaller
pyinstaller --clean --noconfirm pipeline.spec
pyinstaller --clean --noconfirm pipeline_free.spec
( cd PyAppArq && pyinstaller --clean --noconfirm PyAppArq.spec )
```

### Windows (via MSYS2 MinGW64)

Em PowerShell, instalar MSYS2 e abrir o shell MinGW64. Dentro do MinGW64:

```bash
pacman -S --noconfirm mingw-w64-x86_64-toolchain mingw-w64-x86_64-python \
  mingw-w64-x86_64-python-pip mingw-w64-x86_64-python-gobject \
  mingw-w64-x86_64-gtk3 mingw-w64-x86_64-python-pillow \
  mingw-w64-x86_64-python-numpy mingw-w64-x86_64-python-opencv

# Compilar libdmtx no proprio MinGW (a wheel do pylibdmtx assume libdmtx em /mingw64)
cd /tmp && git clone https://github.com/dmtx/libdmtx.git && cd libdmtx
autoreconf -fiv && ./configure --prefix=/mingw64 --enable-shared && make -j && make install

cd /caminho/do/repo
python -m pip install --upgrade pip pyinstaller && pip install -r requirements.txt
pyinstaller --clean --noconfirm pipeline.spec
pyinstaller --clean --noconfirm pipeline_free.spec
( cd PyAppArq && pyinstaller --clean --noconfirm PyAppArq.spec )
```

Após gerar, mova os binários para um diretório de teste limpo e tente
executar:

```bash
mkdir -p /tmp/release-test && cp dist/pipeline /tmp/release-test/
cd /tmp/release-test
./pipeline --input <imagem> --template <template>
```

Se o binário lê os assets corretamente fora da árvore do repositório, o
build está consistente.

## Alertas de segurança

Os binários **não são assinados nem notarizados** (escopo deliberadamente
deixado fora deste pipeline). Como consequência:

**Windows.** O SmartScreen do Windows pode exibir um aviso *"Windows
protegeu seu PC"* na primeira execução. Para autorizar:

1. Clicar em **Mais informações**.
2. Clicar em **Executar assim mesmo**.

**macOS.** O Gatekeeper bloqueia binários sem assinatura Apple Developer
ID, mostrando *"não pode ser aberto porque não é de um desenvolvedor
identificado"*. Para autorizar uma única vez:

1. Clicar com o botão direito no `.app` (ou no binário), depois **Abrir**.
2. Confirmar **Abrir** no diálogo de aviso.

Alternativamente, no Terminal:

```bash
xattr -dr com.apple.quarantine /Applications/PyAppArq.app
xattr -d com.apple.quarantine ./pipeline
```

**Linux.** Não há tela de bloqueio equivalente. Após o download, garantir
permissão de execução:

```bash
chmod +x pipeline pipeline_free
```

## Tamanho esperado

| Binário | Tamanho aproximado |
|---|---|
| `pipeline` | 70 a 90 MB (numpy + opencv + pylibdmtx) |
| `pipeline_free` | 70 a 90 MB |
| `PyAppArq.zip` | 150 a 250 MB (inclui runtime GTK3) |

A GUI é substancialmente maior porque carrega o runtime gráfico inteiro.

## Limitações conhecidas

- Apenas arquiteturas x86_64 e ARM64 (macOS); não há build para Linux
  ARM, Windows ARM, BSDs etc.
- O build Windows depende do ambiente MSYS2 MinGW64. Se o runner do GitHub
  Actions atualizar o pacote `mingw-w64-x86_64-gtk3` ou
  `python-gobject` para uma versão incompatível com PyInstaller, o build
  Windows pode quebrar antes dos outros. Nesse caso, ajustar versões no
  workflow (`pacman -S --noconfirm ...=<versao>`).
- Notebooks com OpenCV alternativos (Apple Silicon com `opencv-python-headless`
  em vez de `opencv-python`) podem precisar ajuste no `requirements.txt`.
- A heurística `--heatmap-fallback` do `pipeline` (opt-in) requer que
  `pipeline_free` esteja disponível como hidden import no bundle da GUI; o
  `PyAppArq.spec` já declara isso.
