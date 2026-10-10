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


def build_frames(avatar: Image.Image, theme: str) -> list[tuple[str, str]]:
    photo = photograph(avatar)
    cloud = point_cloud(avatar, theme)
    lines = contours(avatar, theme)
    tiles = mosaic(avatar, theme)
    return [
        ("image/jpeg", jpeg_b64(photo, 86)),
        ("image/png", png_b64(cloud)),
        ("image/png", png_b64(lines)),
        ("image/png", png_b64(tiles)),
    ]


def _stroke(color: str, body: str, width: str = "1.5") -> str:
    return (
        f'<g fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</g>'
    )


def icon_python(color: str, bg: str) -> str:
    # Two snakes, filled, in one ink. Eyes are punched back to the plate.
    return (
        f'<path fill="{color}" fill-rule="evenodd" d="'
        f"M8.6 11.2V7.4c0-2.2 1.5-3.8 4-3.8h3.2c2 0 3.4 1.3 3.4 3.2v3.2h-4.6"
        f"V8.2c0-.7-.5-1.2-1.2-1.2h-.8c-.8 0-1.3.5-1.3 1.3V11.2H8.6z"
        f"M5.4 9.2c-.8 0-1.4.6-1.4 1.4v1.6c0 1 .7 1.6 1.6 1.6h2.6V9.2H5.4z"
        f"M15.4 12.8v3.8c0 2.2-1.5 3.8-4 3.8H8.2c-2 0-3.4-1.3-3.4-3.2v-3.2h4.6"
        f"v1.8c0 .7.5 1.2 1.2 1.2h.8c.8 0 1.3-.5 1.3-1.3V12.8h2.7z"
        f"M18.6 14.8c.8 0 1.4-.6 1.4-1.4v-1.6c0-1-.7-1.6-1.6-1.6h-2.6v4.6h2.8z"
        f'"/>'
        f'<circle cx="11.6" cy="5.8" r="0.7" fill="{bg}"/>'
        f'<circle cx="12.4" cy="18.2" r="0.7" fill="{bg}"/>'
    )


def icon_sql(color: str, bg: str = "") -> str:
    return _stroke(
        color,
        '<rect x="3.4" y="4.2" width="17.2" height="15.6" rx="1.2"/>'
        '<path d="M3.4 8.8h17.2M9.2 8.8v11M14.9 8.8v11"/>',
    )


def icon_postgres(color: str, bg: str) -> str:
    # Elephant head and trunk, original geometry, one ink.
    return (
        f'<ellipse cx="8.2" cy="11.2" rx="3.2" ry="3.8" fill="{color}"/>'
        f'<ellipse cx="12.6" cy="10.4" rx="4.8" ry="4" fill="{color}"/>'
        f'<path fill="{color}" d="M15.4 11.6c2 .2 3.4 1.4 3.4 3.1 0 1.6-1 2.4-.8 3.6'
        f'.2 1.1 1 1.7 1.8 1.6-.2.8-1.5 1.2-2.3.4-1-.9-1.2-2.1-.9-3.3'
        f'.2-.8.7-1.3.6-2.1-.1-1.1-.8-1.8-2-2z"/>'
        f'<circle cx="13.8" cy="9.3" r="0.75" fill="{bg}"/>'
    )


def icon_oracle(color: str, bg: str = "") -> str:
    return _stroke(color, '<circle cx="12" cy="12" r="6.3"/>', "2.15")


def icon_powerbi(color: str, bg: str = "") -> str:
    return _stroke(
        color,
        '<path d="M6 17V12.2" stroke-width="2.15"/>'
        '<path d="M10.7 17V8.6" stroke-width="2.15"/>'
        '<path d="M15.4 17V6.4" stroke-width="2.15"/>'
        '<path d="M4.2 12.6c3.2-3.4 6-.6 11.4-6.2"/>'
        '<path d="M4 17.2h15.6"/>',
    )


def icon_react(color: str, bg: str = "") -> str:
    return _stroke(
        color,
        f'<circle cx="12" cy="12" r="1.55" fill="{color}" stroke="none"/>'
        '<ellipse cx="12" cy="12" rx="8.6" ry="3.35"/>'
        '<ellipse cx="12" cy="12" rx="8.6" ry="3.35" transform="rotate(60 12 12)"/>'
        '<ellipse cx="12" cy="12" rx="8.6" ry="3.35" transform="rotate(120 12 12)"/>',
    )


def icon_typescript(color: str, bg: str = "") -> str:
    return (
        f'<g fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round">'
        f'<rect x="3.3" y="3.3" width="17.4" height="17.4" rx="2"/>'
        f'</g>'
        f'<text x="12" y="15.8" text-anchor="middle" fill="{color}" '
        f'font-family="{SANS}" font-size="8.2" font-weight="700">TS</text>'
    )


def icon_node(color: str, bg: str = "") -> str:
    return _stroke(
        color,
        '<path d="M12 3.4 20 8v8L12 20.6 4 16V8z"/>'
        '<path d="M12 8.2 16.2 10.6v4.8L12 17.8 7.8 15.4v-4.8z"/>',
    )


def icon_shield(color: str, bg: str = "") -> str:
    return _stroke(
        color,
        f'<path d="M12 3.3 18.3 5.7v5.1c0 3.5-2.4 6-6.3 7.7-3.9-1.7-6.3-4.2-6.3-7.7V5.7z"/>'
        f'<circle cx="12" cy="10.6" r="1.35"/>'
        f'<path d="M12 11.9v2.5"/>',
    )


ICONS = (
    ("python", icon_python, "teal"),
    ("sql", icon_sql, "text"),
    ("postgres", icon_postgres, "teal"),
    ("oracle", icon_oracle, "earth"),
    ("powerbi", icon_powerbi, "gold"),
    ("react", icon_react, "teal"),
    ("typescript", icon_typescript, "text"),
    ("node", icon_node, "gold"),
    ("shield", icon_shield, "earth"),
)


def icon_plate(theme: dict, x: float, y: float, w: float, h: float) -> str:
    arm = 8
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{theme["plot"]}" stroke="{theme["line"]}"/>',
        (
            f'<path d="M{x} {y}h{arm}M{x} {y}v{arm}M{x + w} {y}h-{arm}M{x + w} {y}v{arm}'
            f'M{x} {y + h}h{arm}M{x} {y + h}v-{arm}M{x + w} {y + h}h-{arm}M{x + w} {y + h}v-{arm}" '
            f'fill="none" stroke="{theme["gold"]}" stroke-width="1.3"/>'
        ),
    ]
    inset = 6
    cell = (w - inset * 2) / len(ICONS)
    cy = y + h / 2
    for i, (_name, draw, key) in enumerate(ICONS):
        cx = x + inset + cell * i + cell / 2
        parts.append(
            f'<g transform="translate({cx - 12:.2f} {cy - 12:.2f})">{draw(theme[key], theme["plot"])}</g>'
        )
    return "".join(parts)


COPY = {
    "en": {
        "kicker": "FIG. 01  ·  PORTRAIT AS DATA",
        "frames": ["Photograph", "Point cloud", "Contours", "Mosaic"],
        "line1": "Learning, cooking, swimming,",
        "line2": "games, and the work in between.",
        "focus1": "Data science · machine learning",
        "focus2": "Business intelligence · full-stack",
        "focus3": "Cybersecurity, practiced with care",
        "desc": "A data figure of José Quiros. His portrait moves from a photograph to a point cloud, contour lines, and a color mosaic.",
        "practice_kicker": "FIG. 02  ·  PRACTICE",
        "practice_aside": "FOUR THREADS",
        "practice_label": "What José Quiros works on",
        "practice_rows": [
            ("01", "Data and models", "Python, PyTorch, scikit-learn, pandas, NumPy", "manpac · notebooks · an acoustic model"),
            ("02", "Business intelligence", "ETL, Oracle, PostgreSQL, Power BI", "from raw tables to a report someone trusts"),
            ("03", "Products", "React, TypeScript, Node.js, FastAPI", "search, small web things, a game or two"),
            ("04", "Security", "Cybersecurity and ethical hacking", "curious, and careful about it"),
        ],
        "attention_kicker": "FIG. 03  ·  ATTENTION",
        "attention_aside": "SELF-RATED · NOT A SCOREBOARD",
        "attention_label": "Where José Quiros spends attention",
        "attention_rows": [
            ("Python", 90),
            ("Data science", 86),
            ("Machine learning", 82),
            ("Full-stack", 76),
            ("Business intelligence", 72),
            ("Databases", 66),
            ("Security", 58),
        ],
        "attention_note": "A sketch of public work and stated interests. The numbers are a feeling, not a certificate.",
        "bar_x": 250,
        "bar_w": 560,
    },
    "es": {
        "kicker": "FIG. 01  ·  RETRATO EN DATOS",
        "frames": ["Fotografía", "Nube de puntos", "Contornos", "Mosaico"],
        "line1": "Aprender, cocinar, nadar,",
        "line2": "los juegos, y el trabajo de por medio.",
        "focus1": "Ciencia de datos · aprendizaje automático",
        "focus2": "Inteligencia de negocios · full-stack",
        "focus3": "Ciberseguridad, con cuidado",
        "desc": "Una figura de datos de José Quiros. El retrato pasa de una fotografía a una nube de puntos, curvas de nivel y un mosaico.",
        "practice_kicker": "FIG. 02  ·  PRÁCTICA",
        "practice_aside": "CUATRO HILOS",
        "practice_label": "En qué trabaja José Quiros",
        "practice_rows": [
            ("01", "Datos y modelos", "Python, PyTorch, scikit-learn, pandas, NumPy", "manpac · notebooks · un modelo acústico"),
            ("02", "Inteligencia de negocios", "ETL, Oracle, PostgreSQL, Power BI", "de la tabla cruda al reporte en el que alguien confía"),
            ("03", "Productos", "React, TypeScript, Node.js, FastAPI", "búsquedas, cosas pequeñas en la web, un juego o dos"),
            ("04", "Seguridad", "Ciberseguridad y hacking ético", "con curiosidad, y con cuidado"),
        ],
        "attention_kicker": "FIG. 03  ·  ATENCIÓN",
        "attention_aside": "A OJO · NO ES UN RANKING",
        "attention_label": "Dónde pone la atención José Quiros",
        "attention_rows": [
            ("Python", 90),
            ("Ciencia de datos", 86),
            ("Aprendizaje automático", 82),
            ("Full-stack", 76),
            ("Inteligencia de negocios", 72),
            ("Bases de datos", 66),
            ("Seguridad", 58),
        ],
        "attention_note": "Un apunte del trabajo público y de lo que me importa. Los números son una intuición, no un certificado.",
        "bar_x": 292,
        "bar_w": 518,
    },
}


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


def banner_svg(theme_name: str, frames: list[tuple[str, str]], lang: str) -> str:
    t = THEMES[theme_name]
    c = COPY[lang]
    px, py, size = 56, 76, 380
    layers = []
    for i, ((mime, data), label) in enumerate(zip(frames, c["frames"])):
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
    for i, label in enumerate(c["frames"]):
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
        # Sit the ticks on the right of the caption row so longer
        # Spanish labels ("NUBE DE PUNTOS") still clear them.
        step = 26
        x = px + size - (16 + 3 * step) + i * step
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
    caption_y = py + size + 28
    plate_y = caption_y + 20
    plate_h = 68
    plate_bottom = plate_y + plate_h
    frame_y = 32
    frame_bottom = plate_bottom + 22
    canvas_h = frame_bottom + 28
    frame_h = frame_bottom - frame_y
    plate = icon_plate(t, px, plate_y, size, plate_h)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{canvas_h}" viewBox="0 0 {1200} {canvas_h}" role="img" aria-labelledby="title desc">
  <title id="title">José Quiros</title>
  <desc id="desc">{esc(c["desc"])}</desc>
  <rect width="1200" height="{canvas_h}" fill="{t["bg"]}"/>
  <rect x="36" y="{frame_y}" width="1128" height="{frame_h}" fill="none" stroke="{t["line"]}"/>
  <text x="56" y="58" fill="{t["gold"]}" font-family="{SANS}" font-size="12" letter-spacing="2.2">{esc(c["kicker"])}</text>
  <text x="1144" y="58" text-anchor="end" fill="{t["muted"]}" font-family="{SANS}" font-size="12" letter-spacing="1.8">ECUADOR</text>

  <rect x="{px - 8}" y="{py - 8}" width="{size + 16}" height="{size + 16}" fill="{t["plot"]}" stroke="{t["line"]}"/>
  <clipPath id="plot"><rect x="{px}" y="{py}" width="{size}" height="{size}"/></clipPath>
  <g clip-path="url(#plot)">
    {''.join(layers)}
    <g opacity="0.28">{''.join(grid)}</g>
  </g>
  <path d="M{px} {py}h14M{px} {py}v14M{px + size} {py}h-14M{px + size} {py}v14M{px} {py + size}h14M{px} {py + size}v-14M{px + size} {py + size}h-14M{px + size} {py + size}v-14" fill="none" stroke="{t["gold"]}" stroke-width="1.4"/>
  {''.join(captions)}
  {''.join(marks)}
  {plate}

  <text x="500" y="188" fill="{t["text"]}" font-family="{SERIF}" font-size="64">José</text>
  <text x="500" y="262" fill="{t["text"]}" font-family="{SERIF}" font-size="64">Quiros</text>
  <rect x="500" y="288" width="64" height="3" fill="{t["gold"]}"/>
  <text x="500" y="332" fill="{t["muted"]}" font-family="{SANS}" font-size="18">{esc(c["line1"])}</text>
  <text x="500" y="358" fill="{t["muted"]}" font-family="{SANS}" font-size="18">{esc(c["line2"])}</text>
  <text x="500" y="408" fill="{t["teal"]}" font-family="{SANS}" font-size="15">{esc(c["focus1"])}</text>
  <text x="500" y="434" fill="{t["text"]}" font-family="{SANS}" font-size="15">{esc(c["focus2"])}</text>
  <text x="500" y="460" fill="{t["earth"]}" font-family="{SANS}" font-size="15">{esc(c["focus3"])}</text>
  <text x="500" y="536" fill="{t["faint"]}" font-family="{SANS}" font-size="13" letter-spacing="1.6">@jose-JQ</text>
  {''.join(dots)}
</svg>
'''


def practice_svg(theme_name: str, lang: str) -> str:
    t = THEMES[theme_name]
    c = COPY[lang]
    rows = c["practice_rows"]
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
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520" role="img" aria-label="{esc(c["practice_label"])}">
  <rect width="960" height="520" fill="{t["bg"]}"/>
  <text x="48" y="48" fill="{t["gold"]}" font-family="{SANS}" font-size="12" letter-spacing="2.2">{esc(c["practice_kicker"])}</text>
  <text x="912" y="48" text-anchor="end" fill="{t["faint"]}" font-family="{SANS}" font-size="12" letter-spacing="1.4">{esc(c["practice_aside"])}</text>
  {''.join(body)}
</svg>
'''


def attention_svg(theme_name: str, lang: str) -> str:
    t = THEMES[theme_name]
    c = COPY[lang]
    rows = c["attention_rows"]
    label_x = 36
    bar_x = c["bar_x"]
    bar_w = c["bar_w"]
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
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500" role="img" aria-label="{esc(c["attention_label"])}">
  <rect width="960" height="500" fill="{t["bg"]}"/>
  <text x="36" y="42" fill="{t["gold"]}" font-family="{SANS}" font-size="12" letter-spacing="2.2">{esc(c["attention_kicker"])}</text>
  <text x="924" y="42" text-anchor="end" fill="{t["faint"]}" font-family="{SANS}" font-size="12">{esc(c["attention_aside"])}</text>
  {''.join(body)}
  <text x="36" y="470" fill="{t["muted"]}" font-family="{SANS}" font-size="13">{esc(c["attention_note"])}</text>
</svg>
'''


def lang_badge(label: str, theme_name: str, active: bool) -> str:
    t = THEMES[theme_name]
    stroke = t["gold"] if active else t["line"]
    color = t["gold"] if active else t["muted"]
    rule = (
        f'<rect x="40" y="22" width="38" height="2" fill="{t["gold"]}"/>'
        if active
        else ""
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="118" height="28" viewBox="0 0 118 28" role="img" aria-label="{esc(label)}">
  <rect x="0.5" y="0.5" width="117" height="27" fill="{t["bg"]}" stroke="{stroke}"/>
  <text x="59" y="17.5" text-anchor="middle" fill="{color}" font-family="{SANS}" font-size="12" letter-spacing="0.4">{esc(label)}</text>
  {rule}
</svg>
'''


def main() -> None:
    avatar = load_avatar()
    for theme in ("dark", "light"):
        frames = build_frames(avatar, theme)
        for lang in ("en", "es"):
            suffix = "" if lang == "en" else "-es"
            path = OUT / f"banner{suffix}-{theme}.svg"
            path.write_text(banner_svg(theme, frames, lang), encoding="utf-8")
            print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")
            for name, builder in (("practice", practice_svg), ("attention", attention_svg)):
                out = OUT / f"{name}{suffix}-{theme}.svg"
                out.write_text(builder(theme, lang), encoding="utf-8")
                print(f"{out.name}: {out.stat().st_size / 1024:.1f} KB")
        for code, label in (("en", "English"), ("es", "Español")):
            for state, active in (("on", True), ("off", False)):
                out = OUT / f"lang-{code}-{state}-{theme}.svg"
                out.write_text(lang_badge(label, theme, active), encoding="utf-8")
                print(f"{out.name}: {out.stat().st_size} B")


if __name__ == "__main__":
    main()
