# PyInstaller spec — entrypoint GUI main.py (GTK3, PyGObject).
# Roda de dentro de PyAppArq/: cd PyAppArq && pyinstaller PyAppArq.spec
#
# Modo --onedir (nao --onefile) porque GTK3 + PyGObject em onefile
# costuma falhar ao localizar schemas e typelibs.
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(SPECPATH))      # .../PyAppArq
ROOT = HERE.parent                            # raiz do repo

# Assets locais ao PyAppArq.
datas = [
    (str(HERE / "template.png"), "."),
    (str(HERE / "objetos.json"), "."),
]
# symbols_config.json vive na raiz; ainda assim e util no bundle para o
# load_symbols_config detectar o vocabulario de producao.
if (ROOT / "symbols_config.json").exists():
    datas.append((str(ROOT / "symbols_config.json"), "."))

# Hidden imports:
# - pylibdmtx.pylibdmtx: carregado via ctypes em runtime.
# - pipeline_free (na raiz do repo): importado dinamicamente pelo
#   heatmap_fallback do pipeline interno. Adicionamos ao pathex para que
#   o PyInstaller encontre.
hiddenimports = [
    "pylibdmtx.pylibdmtx",
    "pipeline_free",
    "gi",
    "gi.repository.Gtk",
    "gi.repository.Gdk",
    "gi.repository.GdkPixbuf",
    "gi.repository.GLib",
    "gi.repository.Pango",
]

# Hooks padroes do PyInstaller para gi cobrem o basico; em Linux e
# suficiente quando GTK3 esta instalado no sistema (gir1.2-gtk-3.0).
# No Windows (MSYS2) e no macOS (Homebrew), as instrucoes de instalacao
# do RELEASE.md exigem que GTK esteja instalado no ambiente de build.

block_cipher = None

# No Windows (MSYS2/MinGW), libdmtx-64.dll depende de runtimes do MinGW
# que o PyInstaller nao detecta sozinho. Sem isso, o exe falha com
# "Could not find module 'libdmtx-64.dll' (or one of its dependencies)".
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

a = Analysis(
    ["main.py"],
    pathex=[str(HERE), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PyAppArq",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # --windowed equivalent
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PyAppArq",
)

# No macOS, criar tambem o .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PyAppArq.app",
        icon=None,
        bundle_identifier="br.usp.icmc.pyapparq",
        info_plist={
            "CFBundleName": "PyAppArq",
            "CFBundleDisplayName": "PyAppArq",
            "CFBundleVersion": "1.0.0",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
