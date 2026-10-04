# lorebook.spec — PyInstaller build definition for the Windows app.
#
# Built by .github/workflows/release.yml on windows-latest:
#     pyinstaller lorebook.spec --noconfirm
# Output: dist/LoreBook/ (onedir — onefile with TensorFlow means a huge
# self-extract on every launch), wrapped by installer.iss into the installer.

from PyInstaller.utils.hooks import collect_all

# TensorFlow and Keras 3 load submodules dynamically; collect everything,
# including dist metadata (TF probes importlib.metadata internally).
tf_datas, tf_bins, tf_hidden = collect_all("tensorflow")
keras_datas, keras_bins, keras_hidden = collect_all("keras")

a = Analysis(
    ["UI.py"],
    pathex=[],
    binaries=tf_bins + keras_bins,
    datas=[("lorebook/ui/assets", "lorebook/ui/assets")] + tf_datas + keras_datas,
    hiddenimports=tf_hidden + keras_hidden,
    hookspath=[],
    runtime_hooks=[],
    # Trim modules the app never imports. If a CI build fails on one of
    # these, remove that exclude first — QtNetwork is the likeliest to be a
    # transitive Qt dependency.
    excludes=[
        "tkinter",
        "matplotlib",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtOpenGL",
        "PySide6.QtDBus",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="LoreBook",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # windowed app; crash story is %LOCALAPPDATA%\LoreBook\logs
    icon="lorebook/ui/assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LoreBook",
)
