#!/usr/bin/env python3
"""
Turns a photo into the ASCII portrait shown on the left of the profile card.

Run this ONLY when you want to change the photo:
    pip install pillow numpy rembg onnxruntime
    python photo_to_ascii.py my_photo.jpg

It writes portrait.txt, which generate_profile.py reads. The daily GitHub
Action never runs this file, so the workflow stays dependency-free.

Tuning knobs:
  COLS      characters across (more = more detail, but widen the card)
  BUST      how far down the body to keep, as a fraction of the subject height
  DETAIL    local-contrast gain — raise it if the shirt looks like a solid blob
  WEIGHT    how much of the overall light/dark shape to keep (0 = pure edges)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from rembg import remove

SRC = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"
COLS = 90
# MUST match generate_profile.py ART_LH / ART_CW (currently 1.72).
# Using a smaller value creates extra rows that look vertically stretched in the SVG.
ASPECT = 1.72
BUST = 0.75
DETAIL = 3.0
WEIGHT = 0.4
MAX_ROWS = 60
RAMP = "@%#*+=-:. "  # darkest -> lightest


def main():
    cut = remove(Image.open(SRC))  # cut the subject out of the background
    rgba = np.asarray(cut)
    alpha = rgba[:, :, 3]

    ys, xs = np.nonzero(alpha > 60)
    x0, x1 = xs.min(), xs.max()
    y0 = ys.min()
    y1 = int(y0 + (ys.max() - y0) * BUST)  # head + torso only
    pad = 8
    box = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(rgba.shape[1], x1 + pad),
        min(rgba.shape[0], y1),
    )

    cut = cut.crop(box)
    a = np.asarray(cut)[:, :, 3].astype(float) / 255.0
    g = np.asarray(
        ImageOps.autocontrast(cut.convert("L"), cutoff=1), dtype=np.int16
    )
    h, w = g.shape

    # local contrast: pulls folds/edges out of the flat dark shirt
    blur = np.asarray(
        Image.fromarray(g.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(max(2, w // 55))
        ),
        dtype=np.int16,
    )
    ink = np.clip(150 + (g - blur) * DETAIL + (g - 128) * WEIGHT, 0, 255)

    inside = a > 0.5
    lo, hi = np.percentile(ink[inside], 2), np.percentile(ink[inside], 98)
    ink = np.clip((ink - lo) * 255.0 / max(1, hi - lo), 0, 255)

    rows = max(1, int(COLS * (h / w) / ASPECT))
    if rows > MAX_ROWS:
        rows = MAX_ROWS
    small = np.asarray(
        Image.fromarray(ink.astype(np.uint8)).resize(
            (COLS, rows), Image.LANCZOS
        ),
        dtype=float,
    )
    mask = np.asarray(
        Image.fromarray((a * 255).astype(np.uint8)).resize(
            (COLS, rows), Image.LANCZOS
        ),
        dtype=float,
    )

    n = len(RAMP) - 1
    lines = []
    for y in range(rows):
        line = "".join(
            RAMP[round(small[y, x] / 255 * n)] if mask[y, x] > 110 else " "
            for x in range(COLS)
        )
        # keep full width so the SVG art column stays consistent
        lines.append(line.ljust(COLS).rstrip() if line.strip() else line)

    # trim empty top/bottom, then left-pad so the bust is centered in COLS
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    Path(__file__).parent.joinpath("portrait.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\nwrote portrait.txt  ({COLS} cols x {len(lines)} rows)")


if __name__ == "__main__":
    main()
