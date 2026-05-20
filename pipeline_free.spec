# PyInstaller spec — entrypoint CLI pipeline_free.py (sem grade fixa).
# Roda da raiz do repo: pyinstaller pipeline_free.spec
import os
from pathlib import Path

ROOT = Path(os.path.abspath(SPECPATH))

# pipeline_free.py nao depende de symbols_config.json em runtime, mas
# inclui-lo no bundle nao machuca (alguns codepaths futuros podem usar).
datas = [
    (str(ROOT / "symbols_config.json"), "."),
]

hiddenimports = [
    "pylibdmtx.pylibdmtx",
]

block_cipher = None

a = Analysis(
    ["pipeline_free.py"],
    pathex=[str(ROOT)],
    binaries=[],
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
    name="pipeline_free",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
