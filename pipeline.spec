# PyInstaller spec — entrypoint CLI pipeline.py (baseado em grade fixa).
# Roda da raiz do repo: pyinstaller pipeline.spec
import os
import sys
from pathlib import Path

ROOT = Path(os.path.abspath(SPECPATH))
PLATFORM = sys.platform

# Recursos embutidos. Caminho destino e "." -> raiz do bundle (sys._MEIPASS).
datas = [
    (str(ROOT / "symbols_config.json"), "."),
]

# A biblioteca nativa libdmtx tem nome diferente por plataforma.
# O workflow garante que ela esteja disponivel no PATH/biblioteca padrao do
# runner antes de invocar este spec; aqui apenas declaramos hidden imports.
hiddenimports = [
    "pylibdmtx.pylibdmtx",
]

# No Windows (MSYS2/MinGW), libdmtx-0.dll depende de runtimes do MinGW
# que o PyInstaller nao detecta sozinho.
binaries = []
if sys.platform.startswith("win"):
    mingw_bin = Path(os.environ.get("MINGW_PREFIX", "/mingw64")) / "bin"
    for dll in (
        "libgcc_s_seh-1.dll",
        "libwinpthread-1.dll",
        "libstdc++-6.dll",
        "libdmtx-0.dll",
    ):
        src = mingw_bin / dll
        if src.exists():
            binaries.append((str(src), "."))

block_cipher = None

a = Analysis(
    ["pipeline.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "gi", "gtk"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="pipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
