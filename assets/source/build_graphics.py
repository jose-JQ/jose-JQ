#!/usr/bin/env python3
"""Build José Quiros's profile graphics from assets/source/avatar.png.

The banner is a small data figure: his portrait crossfades between a
photograph, a point cloud, contour lines, and a color mosaic. Practice and
attention charts use the same ink / gold / teal palette. Animations are SMIL.
Photographs are embedded so GitHub's image proxy can render them.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "assets" / "source" / "avatar.png"
OUT = ROOT / "assets"

FRAME = 640
CYCLE = 16.0

SERIF = "Palatino Linotype, Palatino, Liberation Serif, Georgia, serif"
SANS = "Liberation Sans, Segoe UI, Helvetica, Arial, sans-serif"

THEMES = {
    "dark": {
        "bg": "#12161C",
        "plot": "#0E1318",
        "grid": "#24303A",
        "line": "#3C4A56",
        "text": "#F4F0E6",
        "muted": "#9AA39C",
        "faint": "#6E7872",
        "gold": "#E0A106",
        "teal": "#3CABA0",
        "earth": "#E07A5F",
        "track": "#24303A",
        "paper_rgb": (14, 19, 24),
        "shadow_rgb": (28, 92, 88),
        "mid_rgb": (224, 161, 6),
        "hi_rgb": (244, 240, 230),
    },
    "light": {
        "bg": "#F6F3EC",
        "plot": "#FFFCF7",
        "grid": "#E6DFD2",
        "line": "#D4CBBA",
        "text": "#1C1915",
        "muted": "#6E675C",
        "faint": "#8A8174",
        "gold": "#A16207",
        "teal": "#0F6E66",
        "earth": "#C45C3E",
        "track": "#E7E0D4",
        "paper_rgb": (255, 252, 247),
        "shadow_rgb": (15, 110, 102),
        "mid_rgb": (161, 98, 7),
        "hi_rgb": (28, 25, 21),
    },
}


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def jpeg_b64(image: Image.Image, quality: int = 86) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_avatar() -> Image.Image:
    image = Image.open(SRC).convert("RGB")
    return image.resize((FRAME, FRAME), Image.Resampling.LANCZOS)


def luminance(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def stretch(gray: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a, b = np.percentile(gray, [lo, hi])
    return np.clip((gray - a) / (b - a + 1e-6), 0.0, 1.0)


def mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t[..., None]


def colormap(theme: str, t: np.ndarray) -> np.ndarray:
    pal = THEMES[theme]
    shadow = np.array(pal["shadow_rgb"], dtype=np.float32)
    mid = np.array(pal["mid_rgb"], dtype=np.float32)
    hi = np.array(pal["hi_rgb"], dtype=np.float32)
    t = np.clip(t, 0.0, 1.0)
    low = mix(shadow, mid, np.clip(t / 0.55, 0, 1))
    high = mix(mid, hi, np.clip((t - 0.55) / 0.45, 0, 1))
    return np.where(t[..., None] < 0.55, low, high)


def photograph(avatar: Image.Image) -> Image.Image:
    arr = np.asarray(avatar, dtype=np.float32) / 255.0
    arr = np.clip((arr - 0.5) * 1.06 + 0.5, 0.0, 1.0)
    # A little warmth in the highlights, without a neon grade.
    lum = luminance(Image.fromarray((arr * 255).astype(np.uint8)))
    warm = np.array([1.0, 0.94, 0.86], dtype=np.float32)
    arr = np.clip(arr * 0.94 + warm * (lum[..., None] * 0.06), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def point_cloud(avatar: Image.Image, theme: str) -> Image.Image:
    pal = THEMES[theme]
    gray = stretch(luminance(avatar), 4, 97)
    blur = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1)),
        dtype=np.float32,
    ) / 255.0
    gx = np.zeros_like(blur)
    gy = np.zeros_like(blur)
    gx[:, 1:] = np.abs(blur[:, 1:] - blur[:, :-1])
    gy[1:, :] = np.abs(blur[1:, :] - blur[:-1, :])
    edge = gx + gy
    paper = pal["paper_rgb"]
    image = Image.new("RGB", (FRAME, FRAME), paper)
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(7)
    step = 4
    for y in range(2, FRAME - 2, step):
        for x in range(2, FRAME - 2, step):
            e = float(edge[y, x])
            g = float(blur[y, x])
            # Keep the dark room quiet and spend the dots on the person.
            if e < 0.045 and not (0.22 < g < 0.9 and rng.random() < 0.22):
                continue
            if g < 0.08 and e < 0.08:
                continue
            tone = float(np.clip(g * 0.75 + min(e * 3.5, 1.0) * 0.25, 0, 1))
            color = tuple(int(c) for c in colormap(theme, np.array([tone]))[0])
            r = 1.15 + min(e * 9.0, 1.6)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return image


def contours(avatar: Image.Image, theme: str) -> Image.Image:
    pal = THEMES[theme]
    gray = stretch(luminance(avatar), 5, 96)
    soft = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6)),
        dtype=np.float32,
    ) / 255.0
    canvas = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    canvas[:] = pal["paper_rgb"]
    levels = [0.28, 0.42, 0.56, 0.70, 0.84]
    ink_cycle = [pal["shadow_rgb"], pal["mid_rgb"], pal["hi_rgb"], pal["mid_rgb"], pal["shadow_rgb"]]
    for level, color in zip(levels, ink_cycle):
        mask = soft >= level
        boundary = np.zeros_like(mask)
        boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
        boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        # Keep every other pixel so the lines stay fine, not a filled mass.
        boundary[::2, ::2] = False
        canvas[boundary] = color
    return Image.fromarray(canvas, "RGB")


def mosaic(avatar: Image.Image, theme: str) -> Image.Image:
    pal = THEMES[theme]
    arr = np.asarray(photograph(avatar), dtype=np.float32)
    cell = 16
    gap = 2
    image = Image.new("RGB", (FRAME, FRAME), pal["paper_rgb"])
    draw = ImageDraw.Draw(image)
    for y in range(0, FRAME, cell):
        for x in range(0, FRAME, cell):
            block = arr[y : y + cell, x : x + cell]
            if block.size == 0:
                continue
            color = tuple(int(c) for c in block.mean(axis=(0, 1)))
            draw.rectangle(
                (x + gap, y + gap, min(FRAME - 1, x + cell - gap), min(FRAME - 1, y + cell - gap)),
                fill=color,
            )
    return image


def build_frames(avatar: Image.Image, theme: str) -> list[tuple[str, str, str]]:
    photo = photograph(avatar)
    cloud = point_cloud(avatar, theme)
    lines = contours(avatar, theme)
    tiles = mosaic(avatar, theme)
    frames = [
        ("Photograph", "image/jpeg", jpeg_b64(photo, 86)),
        ("Point cloud", "image/png", png_b64(cloud)),
        ("Contours", "image/png", png_b64(lines)),
        ("Mosaic", "image/png", png_b64(tiles)),
    ]
    return frames


def frame_opacity(index: int) -> tuple[str, str]:
    fade = 0.5 / CYCLE
    hold = 0.25
    start = index * hold
    end = start + hold
    if index == 0:
        keys = [0.0, hold - fade, hold, 1.0 - fade, 1.0]
        vals = [1, 1, 0, 0, 1]
    else:
        keys = [0.0, start, start + fade, end - fade, end, 1.0]
        vals = [0, 0, 1, 1, 0, 0]
    return (
        ";".join(f"{k:.4f}" for k in keys),
        ";".join(str(v) for v in vals),
    )


def banner_svg(theme_name: str, frames: list[tuple[str, str, str]]) -> str:
    t = THEMES[theme_name]
    px, py, size = 56, 78, 400
    layers = []
    for i, (label, mime, data) in enumerate(frames):
        keys, vals = frame_opacity(i)
        layers.append(
            f'<g opacity="{"1" if i == 0 else "0"}">'
            f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>'
            f'<image x="{px}" y="{py}" width="{size}" height="{size}" '
            f'preserveAspectRatio="xMidYMid slice" '
            f'href="data:{mime};base64,{data}"/>'
            f"</g>"
        )
    # Shared plot grid sits over every frame.
    grid = []
    for i in range(1, 4):
        offset = px + i * size / 4
        grid.append(
            f'<line x1="{offset:.1f}" y1="{py}" x2="{offset:.1f}" y2="{py + size}" '
            f'stroke="{t["grid"]}" stroke-width="1"/>'
        )
        offset = py + i * size / 4
        grid.append(
            f'<line x1="{px}" y1="{offset:.1f}" x2="{px + size}" y2="{offset:.1f}" '
            f'stroke="{t["grid"]}" stroke-width="1"/>'
        )
    captions = []
    for i, (label, _mime, _data) in enumerate(frames):
        keys, vals = frame_opacity(i)
        captions.append(
            f'<text x="{px}" y="{py + size + 28}" fill="{t["text"]}" font-family="{SANS}" '
            f'font-size="13" letter-spacing="1.4" opacity="{"1" if i == 0 else "0"}">'
            f'0{i + 1}  {esc(label.upper())}'
            f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>'
            f"</text>"
        )
    marks = []
    for i in range(4):
        keys, vals = frame_opacity(i)
        x = px + 168 + i * 28
        marks.append(
            f'<rect x="{x}" y="{py + size + 18}" width="16" height="3" fill="{t["faint"]}"/>'
            f'<rect x="{x}" y="{py + size + 18}" width="16" height="3" fill="{t["gold"]}" '
            f'opacity="{"1" if i == 0 else "0"}">'
            f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>'
            f"</rect>"
        )
    # A few field points drift in the margin. Quiet, not confetti.
    dots = []
    seeds = [(980, 150, t["gold"], "7s"), (1040, 210, t["teal"], "9s"), (1100, 168, t["earth"], "8s"),
             (1010, 390, t["teal"], "11s"), (1124, 430, t["gold"], "10s")]
    for x, y, color, dur in seeds:
        dots.append(
            f'<circle cx="{x}" cy="{y}" r="3.2" fill="{color}">'
            f'<animate attributeName="cy" values="{y};{y - 10};{y}" dur="{dur}" '
            f'repeatCount="indefinite" calcMode="spline" keyTimes="0;0.5;1" '
            f'keySplines="0.4 0 0.6 1;0.4 0 0.6 1"/>'
            f'<animate attributeName="opacity" values="0.35;0.9;0.35" dur="{dur}" repeatCount="indefinite"/>'
            f"</circle>"
        )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560" role="img" aria-labelledby="title desc">
  <title id="title">José Quiros</title>
  <desc id="desc">A data figure of José Quiros. His portrait moves from a photograph to a point cloud, contour lines, and a color mosaic.</desc>
  <rect width="1200" height="560" fill="{t["bg"]}"/>
  <rect x="36" y="36" width="1128" height="488" fill="none" stroke="{t["line"]}"/>
  <text x="56" y="62" fill="{t["gold"]}" font-family="{SANS}" font-size="12" letter-spacing="2.2">FIG. 01  ·  PORTRAIT AS DATA</text>
  <text x="1144" y="62" text-anchor="end" fill="{t["muted"]}" font-family="{SANS}" font-size="12" letter-spacing="1.8">ECUADOR</text>

  <rect x="{px - 8}" y="{py - 8}" width="{size + 16}" height="{size + 16}" fill="{t["plot"]}" stroke="{t["line"]}"/>
  <clipPath id="plot"><rect x="{px}" y="{py}" width="{size}" height="{size}"/></clipPath>
  <g clip-path="url(#plot)">
    {''.join(layers)}
    <g opacity="0.28">{''.join(grid)}</g>
  </g>
  <path d="M{px} {py}h14M{px} {py}v14M{px + size} {py}h-14M{px + size} {py}v14M{px} {py + size}h14M{px} {py + size}v-14M{px + size} {py + size}h-14M{px + size} {py + size}v-14" fill="none" stroke="{t["gold"]}" stroke-width="1.4"/>
  {''.join(captions)}
  {''.join(marks)}

  <text x="520" y="168" fill="{t["text"]}" font-family="{SERIF}" font-size="64">José</text>
  <text x="520" y="242" fill="{t["text"]}" font-family="{SERIF}" font-size="64">Quiros</text>
  <rect x="520" y="268" width="64" height="3" fill="{t["gold"]}"/>
  <text x="520" y="312" fill="{t["muted"]}" font-family="{SANS}" font-size="18">Learning, cooking, swimming,</text>
  <text x="520" y="338" fill="{t["muted"]}" font-family="{SANS}" font-size="18">games, and the work in between.</text>
  <text x="520" y="392" fill="{t["teal"]}" font-family="{SANS}" font-size="15">Data science · machine learning</text>
  <text x="520" y="418" fill="{t["text"]}" font-family="{SANS}" font-size="15">Business intelligence · full-stack</text>
  <text x="520" y="444" fill="{t["earth"]}" font-family="{SANS}" font-size="15">Cybersecurity, practiced with care</text>
  <text x="520" y="500" fill="{t["faint"]}" font-family="{SANS}" font-size="13" letter-spacing="1.6">@jose-JQ</text>
  {''.join(dots)}
</svg>
'''


def practice_svg(theme_name: str) -> str:
    t = THEMES[theme_name]
    rows = [
        ("01", "Data and models", "Python, PyTorch, scikit-learn, pandas, NumPy", "manpac · notebooks · an acoustic model"),
        ("02", "Business intelligence", "ETL, Oracle, PostgreSQL, Power BI", "from raw tables to a report someone trusts"),
        ("03", "Products", "React, TypeScript, Node.js, FastAPI", "search, small web things, a game or two"),
        ("04", "Security", "Cybersecurity and ethical hacking", "curious, and careful about it"),
    ]
    body = []
    y = 108
    for num, title, tools, note in rows:
        body.append(
            f'<text x="48" y="{y}" fill="{t["gold"]}" font-family="{SERIF}" font-size="22">{num}</text>'
            f'<text x="108" y="{y}" fill="{t["text"]}" font-family="{SERIF}" font-size="22">{esc(title)}</text>'
            f'<text x="108" y="{y + 28}" fill="{t["teal"]}" font-family="{SANS}" font-size="15">{esc(tools)}</text>'
            f'<text x="108" y="{y + 50}" fill="{t["muted"]}" font-family="{SANS}" font-size="14">{esc(note)}</text>'
            f'<line x1="48" y1="{y + 68}" x2="912" y2="{y + 68}" stroke="{t["line"]}"/>'
        )
        y += 96
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520" role="img" aria-label="What José Quiros works on">
  <rect width="960" height="520" fill="{t["bg"]}"/>
  <text x="48" y="48" fill="{t["gold"]}" font-family="{SANS}" font-size="12" letter-spacing="2.2">FIG. 02  ·  PRACTICE</text>
  <text x="912" y="48" text-anchor="end" fill="{t["faint"]}" font-family="{SANS}" font-size="12" letter-spacing="1.4">FOUR THREADS</text>
  {''.join(body)}
</svg>
'''


def attention_svg(theme_name: str) -> str:
    t = THEMES[theme_name]
    rows = [
        ("Python", 90),
        ("Data science", 86),
        ("Machine learning", 82),
        ("Full-stack", 76),
        ("Business intelligence", 72),
        ("Databases", 66),
        ("Security", 58),
    ]
    label_x = 36
    bar_x = 250
    bar_w = 560
    body = []
    for i, (label, score) in enumerate(rows):
        y = 86 + i * 52
        width = bar_w * score / 100
        body.append(
            f'<text x="{label_x}" y="{y + 16}" fill="{t["text"]}" font-family="{SERIF}" font-size="18">{esc(label)}</text>'
            f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="10" fill="{t["track"]}"/>'
            f'<rect x="{bar_x}" y="{y}" width="0" height="10" fill="{t["gold"]}">'
            f'<animate attributeName="width" begin="{0.08 * i:.2f}s" dur="0.9s" '
            f'values="0;{width:.1f}" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines="0.2 0.8 0.2 1"/>'
            f"</rect>"
            f'<text x="{bar_x + bar_w + 16}" y="{y + 12}" fill="{t["muted"]}" font-family="{SANS}" font-size="13">{score}</text>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500" role="img" aria-label="Where José Quiros spends attention">
  <rect width="960" height="500" fill="{t["bg"]}"/>
  <text x="36" y="42" fill="{t["gold"]}" font-family="{SANS}" font-size="12" letter-spacing="2.2">FIG. 03  ·  ATTENTION</text>
  <text x="924" y="42" text-anchor="end" fill="{t["faint"]}" font-family="{SANS}" font-size="12">SELF-RATED · NOT A SCOREBOARD</text>
  {''.join(body)}
  <text x="36" y="470" fill="{t["muted"]}" font-family="{SANS}" font-size="13">A sketch of public work and stated interests. The numbers are a feeling, not a certificate.</text>
</svg>
'''


def main() -> None:
    avatar = load_avatar()
    for theme in ("dark", "light"):
        frames = build_frames(avatar, theme)
        path = OUT / f"banner-{theme}.svg"
        path.write_text(banner_svg(theme, frames), encoding="utf-8")
        print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")
        for name, builder in (("practice", practice_svg), ("attention", attention_svg)):
            out = OUT / f"{name}-{theme}.svg"
            out.write_text(builder(theme), encoding="utf-8")
            print(f"{out.name}: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
