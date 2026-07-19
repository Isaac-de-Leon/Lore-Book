#!/usr/bin/env python3
"""Generate the app icon (lorebook/ui/assets/icon.ico + icon.png).

Renders the existing "book" sidebar SVG (lorebook/ui/icons.py) in the app's
accent orange on a dark rounded-square tile — the same look as the sidebar
logo — at every size Windows wants in an .ico, plus a 256px PNG for the
runtime window icon.

Run once and commit the outputs (offscreen rendering, no display needed):

    QT_QPA_PLATFORM=offscreen python scripts/make_icon.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, Qt
from PySide6.QtGui import QGuiApplication, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from lorebook.ui.icons import _ICONS

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ACCENT = "#F97316"      # dark-theme accent (styles.py)
TILE = "#1C1917"        # dark tile behind the glyph so it reads on any taskbar
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lorebook", "ui", "assets"
)

# Full-tile SVG: rounded square + the book glyph scaled into its center.
_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    f'<rect x="0" y="0" width="32" height="32" rx="7" fill="{TILE}"/>'
    f'<g transform="translate(4 4)" fill="none" stroke="{ACCENT}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    f'{_ICONS["book"]}'
    "</g></svg>"
)


def render(size: int) -> Image.Image:
    renderer = QSvgRenderer(QByteArray(_SVG.encode()))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()

    buf = QBuffer()
    buf.open(QBuffer.ReadWrite)
    pm.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 — Qt needs an app for rendering

    images = {size: render(size) for size in ICO_SIZES}
    png_path = os.path.join(OUT_DIR, "icon.png")
    images[256].save(png_path)

    ico_path = os.path.join(OUT_DIR, "icon.ico")
    images[256].save(
        ico_path,
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=[images[s] for s in ICO_SIZES if s != 256],
    )
    print(f"Wrote {png_path} and {ico_path}")


if __name__ == "__main__":
    main()
