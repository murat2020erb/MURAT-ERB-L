# -*- coding: utf-8 -*-
"""Osmanlı Shorts video üretici.

Kullanım:  python3 render.py 1 2 3      (bölüm numaraları; boşsa hepsi)
Gerekenler: ffmpeg, piper-tts, basemap, pillow, numpy
"""
import os, sys, math, json, random, subprocess, wave, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from episodes import EPISODES

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
BUILD = os.environ.get("BUILD_DIR", os.path.join(HERE, "build"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "..", "videolar"))
VOICE = os.environ.get("PIPER_VOICE", os.path.join(ASSETS, "voice", "tr_TR-dfki-medium.onnx"))
W, H, FPS = 1080, 1920, 30
SR = 44100
XFADE = 0.35
OUTRO = 3.0

BORDO = (122, 31, 43)
GOLD = (232, 185, 80)
GOLD_L = (246, 214, 130)
INK = (12, 10, 14)

F = lambda n, s: ImageFont.truetype(os.path.join(ASSETS, "fonts", n), s)
FONT_CAP = F("Montserrat_800ExtraBold.ttf", 68)
FONT_LABEL = F("Cinzel_900Black.ttf", 40)
FONT_BADGE = F("Cinzel_900Black.ttf", 44)
FONT_TITLE = F("Montserrat_600SemiBold.ttf", 34)
FONT_BIG = F("Cinzel_900Black.ttf", 118)
FONT_MAPDOT = F("Montserrat_800ExtraBold.ttf", 40)
FONT_MAPTXT = F("Cinzel_900Black.ttf", 58)
FONT_SMALL = F("Montserrat_600SemiBold.ttf", 22)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


# ------------------------------------------------------------------ ses
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_wav(path):
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() == 2:
            a = a.reshape(-1, 2).mean(1)
    return a


def tts(text, out):
    if os.path.exists(out):
        return read_wav(out)
    raw = out + ".raw.wav"
    subprocess.run([sys.executable, "-m", "piper", "-m", VOICE, "-f", raw, "--length-scale", "0.97",
                    "--sentence-silence", "0.18"], input=text.encode("utf-8"), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["ffmpeg", "-y", "-i", raw, "-af",
         "highpass=f=70,equalizer=f=180:t=q:w=1:g=2,equalizer=f=3500:t=q:w=1.5:g=2.5,"
         "acompressor=threshold=-20dB:ratio=3:attack=5:release=80,"
         "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
         "silenceremove=start_periods=1:start_threshold=-45dB,areverse",
         "-ar", str(SR), "-ac", "1", out])
    os.remove(raw)
    return read_wav(out)


_WHISPER = None


def word_times(wav):
    """Whisper ile kelime zamanları (altyazı senkronu). Whisper yoksa None."""
    out = wav[:-4] + ".words.json"
    if os.path.exists(out):
        return json.load(open(out))
    global _WHISPER
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    if _WHISPER is None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8",
                                download_root=os.path.join(BUILD, "whisper"))
    a = np.frombuffer(subprocess.run(["ffmpeg", "-loglevel", "error", "-i", wav, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                                     capture_output=True).stdout, np.float32)
    segs, _ = _WHISPER.transcribe(a, language="tr", word_timestamps=True, beam_size=5)
    words = [dict(w=w.word.strip(), s=round(w.start, 3), e=round(w.end, 3)) for sg in segs for w in sg.words]
    json.dump(words, open(out, "w"), ensure_ascii=False)
    return words


def time_at_fraction(words, f, total):
    """Konuşulan metnin f oranındaki (karakter bazlı) anı bul."""
    if not words:
        return f * total
    lens = [len(w["w"]) + 1 for w in words]
    target = f * sum(lens)
    acc = 0
    for w, L in zip(words, lens):
        if acc + L >= target:
            return w["s"] + (w["e"] - w["s"]) * (target - acc) / L
        acc += L
    return words[-1]["e"]


def make_music(dur, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    music = np.zeros(n, np.float32)
    # dem (drone): Re + La, harmonikli
    for f0, amp in ((73.42, 0.22), (110.0, 0.14), (146.83, 0.07)):
        lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6))
        for h, ha in ((1, 1), (2, 0.45), (3, 0.25), (4, 0.12)):
            music += amp * ha * lfo * np.sin(2 * np.pi * f0 * h * t * (1 + rng.uniform(-0.002, 0.002)))
    # yavaş akorlar (Re minör - Si bemol - Do - La)
    chords = [(293.66, 349.23, 440.0), (233.08, 293.66, 349.23), (261.63, 329.63, 392.0), (220.0, 277.18, 329.63)]
    seg = 4.0
    pad = np.zeros(n, np.float32)
    for i in range(int(dur / seg) + 1):
        s, e = int(i * seg * SR), min(n, int((i + 1) * seg * SR + 0.8 * SR))
        if s >= n:
            break
        tt = np.arange(e - s) / SR
        env = np.minimum(1, tt / 1.2) * np.minimum(1, (tt[-1] - tt) / 1.0 + 1e-3)
        for f in chords[i % 4]:
            pad[s:e] += 0.05 * env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.003 * tt))
    music += pad
    # kudüm/def: düm - tek deseni, 84 bpm
    beat = 60 / 84
    pattern = [(0.0, "dum"), (1.0, "tek"), (1.5, "tek"), (2.0, "dum"), (3.0, "tek")]
    k = 0
    while True:
        base = k * 4 * beat
        if base > dur:
            break
        for off, kind in pattern:
            st = int((base + off * beat) * SR)
            if st >= n:
                continue
            L = int(0.35 * SR)
            tt = np.arange(L) / SR
            if kind == "dum":
                hit = 0.5 * np.sin(2 * np.pi * (55 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
            else:
                nz = rng.standard_normal(L).astype(np.float32)
                nz = nz - np.concatenate([[0], nz[:-1]])  # yüksek geçiren
                hit = 0.10 * nz * np.exp(-tt * 45) + 0.12 * np.sin(2 * np.pi * 420 * tt) * np.exp(-tt * 30)
            e = min(n, st + L)
            music[st:e] += hit[: e - st]
        k += 1
    music /= np.abs(music).max() + 1e-6
    fade = int(1.5 * SR)
    music[:fade] *= np.linspace(0, 1, fade)
    music[-fade:] *= np.linspace(1, 0, fade)
    return music


def sfx_whoosh(rng):
    L = int(0.6 * SR)
    t = np.arange(L) / SR
    nz = rng.standard_normal(L).astype(np.float32)
    env = np.sin(np.pi * t / t[-1]) ** 2
    # kayan bant: fark alarak parlaklaştır, zarfla şekillendir
    out = np.zeros(L, np.float32)
    acc = 0.0
    a = 0.02 + 0.25 * env
    for i in range(L):
        acc += a[i] * (nz[i] - acc)
        out[i] = acc
    return 0.6 * out * env


def sfx_boom():
    L = int(1.4 * SR)
    t = np.arange(L) / SR
    return (0.9 * np.sin(2 * np.pi * (38 + 30 * np.exp(-t * 6)) * t) * np.exp(-t * 3.2)).astype(np.float32)


# ------------------------------------------------------------------ harita
MAPS = {  # ad: (merkez enlem, merkez boylam, boylam genişliği, çözünürlük)
    "world": (22.0, 24.0, 58.0, "l"),
    "asia": (32.0, 45.0, 34.0, "l"),
    "nw_anatolia": (39.75, 29.6, 3.2, "h"),
    "bursa_close": (39.87, 29.15, 1.5, "h"),
    "marmara": (39.95, 27.6, 4.6, "i"),
    "balkans": (40.2, 24.0, 9.5, "i"),
}


def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def inv_merc(y):
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)


def map_extent(name):
    clat, clon, span, _ = MAPS[name]
    wr = math.radians(span)
    hr = wr * H / W
    cy = merc_y(clat)
    return clon - span / 2, inv_merc(cy - hr / 2), clon + span / 2, inv_merc(cy + hr / 2)


def to_px(name, lat, lon):
    x0, y0, x1, y1 = map_extent(name)
    px = (lon - x0) / (x1 - x0) * W
    my0, my1 = merc_y(y0), merc_y(y1)
    py = (1 - (merc_y(lat) - my0) / (my1 - my0)) * H
    return px, py


@functools.lru_cache(None)
def map_base(name):
    path = os.path.join(BUILD, "maps", name + ".png")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        x0, y0, x1, y1 = map_extent(name)
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        m = Basemap(projection="merc", llcrnrlon=x0, llcrnrlat=y0, urcrnrlon=x1, urcrnrlat=y1,
                    resolution=MAPS[name][3], ax=ax, fix_aspect=False)
        m.drawmapboundary(fill_color="#14212e", linewidth=0)
        m.fillcontinents(color="#d6bf93", lake_color="#14212e")
        m.drawcoastlines(color="#5b4426", linewidth=1.6)
        ax.set_xlim(m.llcrnrx, m.urcrnrx); ax.set_ylim(m.llcrnry, m.urcrnry); ax.set_aspect("auto")
        fig.savefig(path, dpi=100, facecolor="#14212e")
        plt.close(fig)
        # parşömen dokusu + kenar kararması
        im = Image.open(path).convert("RGB").resize((W, H))
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32)
        noise = np.array(Image.fromarray(((noise * 20) + 128).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
                         .filter(ImageFilter.GaussianBlur(3)), np.float32) - 128
        a = np.array(im, np.float32) + noise[..., None] * 0.6
        Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(path)
    return Image.open(path).convert("RGB")


def text_center(d, xy, s, font, fill, stroke=4, sfill=INK, anchor="mm"):
    d.text(xy, s, font=font, fill=fill, stroke_width=stroke, stroke_fill=sfill, anchor=anchor)


def draw_map_layer(name, layers, t):
    im = map_base(name).copy()
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for L in layers:
        kind = L[0]
        if kind == "blob":
            pts, t0, dur = L[1], L[2], L[3]
            p = ease_out((t - t0) / dur)
            if p <= 0:
                continue
            px = [to_px(name, la, lo) for la, lo in pts]
            cx = sum(x for x, _ in px) / len(px); cy = sum(y for _, y in px) / len(px)
            sc = 0.2 + 0.8 * p
            q = [(cx + (x - cx) * sc, cy + (y - cy) * sc) for x, y in px]
            d.polygon(q, fill=BORDO + (int(170 * p),), outline=GOLD + (255,), width=5)
        elif kind == "border":
            pts, t0 = L[1], L[2]
            p = ease((t - t0) / 1.5)
            px = [to_px(name, la, lo) for la, lo in pts]
            seglens = [math.dist(px[i], px[i + 1]) for i in range(len(px) - 1)]
            total = sum(seglens) * p
            for i, sl in enumerate(seglens):
                if total <= 0:
                    break
                f = min(1, total / sl)
                a, b = px[i], px[i + 1]
                e = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                nd = int(sl * f / 26)
                for k in range(nd + 1):
                    s0 = k / max(nd, 1) * f
                    s1 = min(f, s0 + 14 / sl)
                    d.line([(a[0] + (b[0] - a[0]) * s0, a[1] + (b[1] - a[1]) * s0),
                            (a[0] + (b[0] - a[0]) * s1, a[1] + (b[1] - a[1]) * s1)], fill=(178, 34, 34, 255), width=9)
                total -= sl
        elif kind in ("dot", "pulse"):
            (la, lo), label, t0 = L[1], L[2], L[3]
            a = ease((t - t0) / 0.5) if t0 > -5 else 1
            if a <= 0:
                continue
            x, y = to_px(name, la, lo)
            if kind == "pulse":
                for k in range(3):
                    ph = ((t - t0) * 0.8 + k / 3) % 1
                    r = 18 + ph * 90
                    d.ellipse([x - r, y - r, x + r, y + r], outline=GOLD + (int(255 * (1 - ph) * a),), width=6)
            r = 16
            d.ellipse([x - r, y - r, x + r, y + r], fill=GOLD + (int(255 * a),), outline=INK + (int(255 * a),), width=4)
            ty = y - 58 if y > 200 else y + 58
            d.text((x, ty), label, font=FONT_MAPDOT, fill=(255, 255, 255, int(255 * a)), stroke_width=5,
                   stroke_fill=INK + (int(255 * a),), anchor="mm")
        elif kind == "text":
            (la, lo), label, t0 = L[1], L[2], L[3]
            a = ease((t - t0) / 0.8) if t0 > -5 else 1
            x, y = to_px(name, la, lo)
            d.text((x, y), label, font=FONT_MAPTXT, fill=(90, 60, 30, int(200 * a)), anchor="mm")
        elif kind == "arrow":
            A, B, t0 = L[1], L[2], L[3]
            p = ease((t - t0) / 1.4)
            if p <= 0:
                continue
            ax_, ay = to_px(name, *A); bx, by = to_px(name, *B)
            mx, my = (ax_ + bx) / 2, (ay + by) / 2
            nx, ny = -(by - ay), (bx - ax_)
            ln = math.hypot(nx, ny) or 1
            cx, cy = mx + nx / ln * math.dist((ax_, ay), (bx, by)) * 0.18, my + ny / ln * math.dist((ax_, ay), (bx, by)) * 0.18
            pts = []
            for k in range(int(60 * p) + 1):
                s = k / 60
                pts.append(((1 - s) ** 2 * ax_ + 2 * (1 - s) * s * cx + s * s * bx,
                            (1 - s) ** 2 * ay + 2 * (1 - s) * s * cy + s * s * by))
            if len(pts) > 1:
                d.line(pts, fill=INK + (255,), width=20, joint="curve")
                d.line(pts, fill=BORDO + (255,), width=12, joint="curve")
                (x1, y1), (x2, y2) = pts[-2], pts[-1]
                ang = math.atan2(y2 - y1, x2 - x1)
                hs = 38
                head = [(x2 + math.cos(ang) * hs * 0.6, y2 + math.sin(ang) * hs * 0.6),
                        (x2 + math.cos(ang + 2.5) * hs, y2 + math.sin(ang + 2.5) * hs),
                        (x2 + math.cos(ang - 2.5) * hs, y2 + math.sin(ang - 2.5) * hs)]
                d.polygon(head, fill=BORDO + (255,), outline=INK + (255,), width=4)
        elif kind == "ring":
            (la, lo), label, t0 = L[1], L[2], L[3]
            x, y = to_px(name, la, lo)
            R = 300
            for k in range(8):
                tk = t0 + 0.35 + k * 0.45
                a = ease((t - tk) / 0.4)
                if a <= 0:
                    continue
                ang = -math.pi / 2 + k * 2 * math.pi / 8
                kx, ky = x + R * math.cos(ang), y + R * math.sin(ang)
                s = 26
                d.rectangle([kx - s, ky - s * 0.7, kx + s, ky + s], fill=BORDO + (int(255 * a),), outline=GOLD + (int(255 * a),), width=4)
                for j in range(3):
                    d.rectangle([kx - s + j * s * 0.8, ky - s * 1.15, kx - s + j * s * 0.8 + s * 0.45, ky - s * 0.7],
                                fill=BORDO + (int(255 * a),), outline=GOLD + (int(255 * a),), width=3)
            p = ease((t - t0 - 4.0) / 1.0)
            if p > 0:
                d.ellipse([x - R, y - R, x + R, y + R], outline=(178, 34, 34, int(255 * p)), width=8)
            r = 18
            d.ellipse([x - r, y - r, x + r, y + r], fill=GOLD + (255,), outline=INK + (255,), width=4)
            d.text((x, y - 62), label, font=FONT_MAPDOT, fill=(255, 255, 255), stroke_width=5, stroke_fill=INK, anchor="mm")
    im.paste(ov, (0, 0), ov)
    return im


# ------------------------------------------------------------------ prosedürel görseller
def night_bg(t, seed=3):
    rng = random.Random(seed)
    g = np.linspace(0, 1, H)[:, None]
    top, bot = np.array([8, 12, 30]), np.array([60, 18, 36])
    arr = (top * (1 - g) + bot * g)[:, None, :].repeat(W, 1)
    im = Image.fromarray(np.broadcast_to(arr, (H, W, 3)).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(220):
        x, y, r = rng.uniform(0, W), rng.uniform(0, H * 0.75), rng.choice([1, 1, 2, 2, 3])
        tw = 0.5 + 0.5 * math.sin(t * rng.uniform(1, 3) + rng.uniform(0, 6))
        c = int(120 + 135 * tw)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(c, c, int(c * 0.9)))
    return im


def crescent(im, x, y, R, glow=1.0):
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.ellipse([x - R * 1.9, y - R * 1.9, x + R * 1.9, y + R * 1.9], fill=GOLD + (int(70 * glow),))
    lay = lay.filter(ImageFilter.GaussianBlur(R * 0.6))
    d = ImageDraw.Draw(lay)
    d.ellipse([x - R, y - R, x + R, y + R], fill=GOLD_L + (255,))
    d.ellipse([x - R * 0.55, y - R * 1.05, x + R * 1.25, y + R * 0.75], fill=(0, 0, 0, 0))
    im.paste(lay, (0, 0), lay)


def draw_tree(d, x, y, ang, length, depth, maxd, grow, width, rnd):
    if depth > maxd:
        return
    p = min(1, max(0, grow * (maxd + 1) - depth))
    if p <= 0:
        return
    x2 = x + math.cos(ang) * length * p
    y2 = y + math.sin(ang) * length * p
    d.line([(x, y), (x2, y2)], fill=GOLD + (255,), width=max(2, int(width)))
    if depth == maxd and p >= 1:
        r = 7
        d.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=GOLD_L + (255,))
    if p >= 1:
        n = 3 if depth < 3 else 2
        for k in range(n):
            da = (k - (n - 1) / 2) * (0.62 if n == 3 else 0.7) + rnd.uniform(-0.12, 0.12)
            draw_tree(d, x2, y2, ang + da, length * rnd.uniform(0.66, 0.78), depth + 1, maxd, grow, width * 0.68, rnd)


def dream_frame(t, dur, tree=True):
    im = night_bg(t)
    if not tree:
        crescent(im, W * 0.5, H * 0.36, 150, 0.8 + 0.2 * math.sin(t * 2))
        return im
    p = ease(t / (dur * 0.35))
    cx = W * 0.25 + (W * 0.5 - W * 0.25) * p
    cy = H * 0.18 + (H * 0.62 - H * 0.18) * p
    R = 110 * (1 - 0.5 * p)
    g = ease((t - dur * 0.33) / (dur * 0.55))
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    if g > 0:
        draw_tree(d, W / 2, H * 0.66, -math.pi / 2, 300, 0, 7, g, 40, random.Random(7))
    glow = lay.filter(ImageFilter.GaussianBlur(14))
    im.paste(glow, (0, 0), glow)
    im.paste(lay, (0, 0), lay)
    if p < 1 or g < 0.15:
        crescent(im, cx, cy, R, 1.0)
    # zemin silueti
    d2 = ImageDraw.Draw(im)
    d2.ellipse([-300, H * 0.64, W + 300, H * 1.1], fill=(14, 8, 14))
    return im


def kayi_damga():
    im = Image.new("RGB", (W, H), BORDO)
    d = ImageDraw.Draw(im)
    for k in range(-H, W + H, 90):
        d.line([(k, 0), (k + H, H)], fill=(110, 26, 38), width=3)
    cx, cy, s = W / 2, H * 0.34, 210
    wd = 46
    d.line([(cx - s * 0.95, cy - s), (cx - s * 0.95, cy + s)], fill=GOLD, width=wd)
    d.line([(cx + s * 0.95, cy - s), (cx + s * 0.95, cy + s)], fill=GOLD, width=wd)
    d.line([(cx - s * 0.5, cy - s), (cx, cy)], fill=GOLD, width=wd)
    d.line([(cx + s * 0.5, cy - s), (cx, cy)], fill=GOLD, width=wd)
    d.line([(cx, cy - 20), (cx, cy + s)], fill=GOLD, width=wd)
    d.ellipse([cx - 330, cy - 330, cx + 330, cy + 330], outline=GOLD, width=8)
    text_center(d, (cx, cy + 410), "KAYI BOYU DAMGASI", FONT_LABEL, GOLD_L)
    return im



# ------------------------------------------------------------------ motifli kartlar
def star8(d, cx, cy, r, **kw):
    pts = []
    for k in range(16):
        a = math.pi / 8 * k - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.72
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, **kw)


@functools.lru_cache(None)
def card_bg(tone):
    top, bot = ((40, 10, 18), (110, 28, 40)) if tone == "bordo" else ((8, 16, 34), (30, 44, 70))
    g = np.linspace(0, 1, H)[:, None]
    arr = (np.array(top) * (1 - g) + np.array(bot) * g)[:, None, :].repeat(W, 1).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for y in range(-60, H + 120, 150):
        for x in range(-60 + (75 if (y // 150) % 2 else 0), W + 120, 150):
            star8(d, x, y, 46, outline=GOLD + (34,), width=3)
    im.alpha_composite(lay)
    return im.convert("RGB")


def motif(d, kind, cx, cy, t):
    G = GOLD + (255,)
    if kind == "tent":
        d.rectangle([0, cy + 120, W, H], fill=(14, 8, 12, 255))
        for k, (x, s) in enumerate(((cx - 300, 0.8), (cx, 1.15), (cx + 300, 0.9))):
            w, h = 220 * s, 150 * s
            b = cy + 120
            d.rectangle([x - w / 2, b - h * 0.55, x + w / 2, b], fill=G)
            d.pieslice([x - w / 2, b - h * 1.25, x + w / 2, b + h * 0.05 - h * 0.55 + h * 0.1], 180, 360, fill=G)
            d.rectangle([x - w * 0.12, b - h * 0.4, x + w * 0.12, b], fill=(40, 12, 20, 255))
    elif kind == "dome":
        b = cy + 220
        d.rectangle([cx - 230, b - 230, cx + 230, b], fill=G)
        d.rectangle([cx - 250, b - 250, cx + 250, b - 220], fill=G)
        d.pieslice([cx - 210, b - 470, cx + 210, b - 50], 180, 360, fill=G)
        d.line([(cx, b - 260), (cx, b - 330)], fill=G, width=1)
        d.rectangle([cx - 6, b - 330, cx + 6, b - 260], fill=G)
        crescent_small(d, cx, b - 365, 30)
        for k in (-1, 0, 1):
            d.rounded_rectangle([cx + k * 130 - 38, b - 170, cx + k * 130 + 38, b - 40], 36, fill=(40, 12, 20, 255))
        d.rectangle([0, b, W, H], fill=(14, 8, 12, 255))
    elif kind == "castle":
        b = cy + 230
        d.rectangle([cx - 420, b - 230, cx + 420, b], fill=G)
        for x in range(int(cx - 420), int(cx + 420), 60):
            d.rectangle([x, b - 270, x + 34, b - 230], fill=G)
        for tx in (cx - 380, cx, cx + 380):
            d.rectangle([tx - 75, b - 400, tx + 75, b], fill=G)
            for x in range(int(tx - 75), int(tx + 75), 44):
                d.rectangle([x, b - 440, x + 26, b - 400], fill=G)
        d.pieslice([cx - 60, b - 150, cx + 60, b - 30], 180, 360, fill=(40, 12, 20, 255))
        d.rectangle([cx - 60, b - 90, cx + 60, b], fill=(40, 12, 20, 255))
        d.rectangle([0, b, W, H], fill=(14, 8, 12, 255))
    elif kind == "spears":
        b = cy + 260
        rnd = random.Random(5)
        for k in range(26):
            x = 30 + k * 40 + rnd.uniform(-8, 8)
            top = b - 420 - rnd.uniform(0, 110) + 8 * math.sin(t * 2 + k)
            d.line([(x, b), (x, top)], fill=G, width=6)
            d.polygon([(x - 13, top), (x + 13, top), (x, top - 46)], fill=G)
            if k % 6 == 2:
                wv = 10 * math.sin(t * 3 + k)
                d.polygon([(x, top + 10), (x + 110, top + 30 + wv), (x, top + 75)], fill=BORDO + (255,), outline=G)
        d.rectangle([0, b, W, H], fill=(14, 8, 12, 255))
    elif kind == "star":
        r = 260 + 8 * math.sin(t * 1.5)
        star8(d, cx, cy, r, outline=G, width=14)
        star8(d, cx, cy, r * 0.62, fill=BORDO + (255,), outline=G, width=8)
        crescent_small(d, cx, cy, 70)
    elif kind == "coin":
        r = 270
        cy += 60
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(196, 196, 204, 255), outline=(120, 120, 130, 255), width=10)
        d.ellipse([cx - r * 0.78, cy - r * 0.78, cx + r * 0.78, cy + r * 0.78], outline=(130, 130, 140, 255), width=6)
        for k in range(48):
            a = k * math.pi / 24
            x, y = cx + r * 0.89 * math.cos(a), cy + r * 0.89 * math.sin(a)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(130, 130, 140, 255))
        d.text((cx, cy), "AKÇE", font=F("Cinzel_900Black.ttf", 110), fill=(110, 110, 120, 255), anchor="mm")
    elif kind == "lightning":
        cy += 70
        pts = [(cx + 40, cy - 330), (cx - 140, cy + 20), (cx - 10, cy + 20), (cx - 90, cy + 330), (cx + 160, cy - 60), (cx + 20, cy - 60), (cx + 120, cy - 330)]
        d.polygon(pts, fill=GOLD_L + (255,))


def crescent_small(d, x, y, R):
    d.ellipse([x - R, y - R, x + R, y + R], fill=GOLD_L + (255,))
    d.ellipse([x - R * 0.45, y - R * 1.0, x + R * 1.3, y + R * 0.8], fill=BORDO + (255,))


def card_frame(title, sub, kind, t, u):
    im = card_bg("navy" if kind in ("tent", "spears", "lightning", "castle") else "bordo").copy().convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    motif(d, kind, W / 2, 760, t)
    glow = lay.filter(ImageFilter.GaussianBlur(22))
    im.alpha_composite(glow)
    im.alpha_composite(lay)
    d = ImageDraw.Draw(im)
    a = ease((t - 0.1) / 0.6)
    fs = 92 if FONT_BIG.getlength(title) * 92 / 118 < W - 100 else int(92 * (W - 100) / (FONT_BIG.getlength(title) * 92 / 118))
    d.text((W / 2, 390), title, font=F("Cinzel_900Black.ttf", fs), fill=GOLD_L + (int(255 * a),),
           stroke_width=6, stroke_fill=INK + (int(255 * a),), anchor="mm")
    d.text((W / 2, 475), sub, font=F("Montserrat_600SemiBold.ttf", 42), fill=(255, 255, 255, int(230 * a)),
           stroke_width=3, stroke_fill=INK + (int(255 * a),), anchor="mm")
    z = 1 + 0.06 * u
    return im.convert("RGB").resize((W, H), Image.BILINEAR, box=_zoombox(z))


# ------------------------------------------------------------------ görüntü (Ken Burns)
@functools.lru_cache(None)
def load_img(key):
    for ext in (".jpg", ".png"):
        p = os.path.join(ASSETS, "img", key + ext)
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    print("UYARI: görsel yok ->", key, file=sys.stderr)
    im = Image.new("RGB", (900, 1400), (60, 20, 28))
    ImageDraw.Draw(im).text((450, 700), key, font=FONT_LABEL, fill=GOLD, anchor="mm")
    return im


@functools.lru_cache(None)
def blurred_bg(key):
    im = load_img(key)
    sc = max(W / im.width, H / im.height)
    bg = im.resize((int(im.width * sc) + 1, int(im.height * sc) + 1)).crop((0, 0, W, H))
    bg = bg.filter(ImageFilter.GaussianBlur(40))
    return Image.blend(bg, Image.new("RGB", (W, H), (0, 0, 0)), 0.45)


def img_frame(key, opts, u):
    """u: çekim içindeki ilerleme (0..1+)"""
    im = load_img(key)
    zoom = opts.get("zoom", "in")
    z = 1.0 + 0.13 * (u if zoom == "in" else 1 - u)
    fx, fy = opts.get("focus", (0.5, 0.5))
    aspect = im.width / im.height
    if aspect > 0.78 or opts.get("fit"):
        bg = blurred_bg(key).copy()
        fw = W * 0.98 * z
        fh = fw / aspect
        if fh > H * 0.62 * z:
            fh = H * 0.62 * z
            fw = fh * aspect
        fg = im.resize((int(fw), int(fh)), Image.BILINEAR)
        x = int((W - fw) / 2 + (0.5 - fx) * (fw - W * 0.98) * 0.5)
        y = int(H * 0.40 - fh / 2)
        sh = Image.new("RGBA", (int(fw) + 60, int(fh) + 60), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle([30, 30, int(fw) + 30, int(fh) + 30], fill=(0, 0, 0, 160))
        sh = sh.filter(ImageFilter.GaussianBlur(18))
        bg.paste(sh, (x - 30, y - 20), sh)
        bg.paste(fg, (x, y))
        ImageDraw.Draw(bg).rectangle([x - 2, y - 2, x + int(fw) + 1, y + int(fh) + 1], outline=GOLD, width=4)
        return bg
    sc = max(W / im.width, H / im.height) * z
    cw, ch = W / sc, H / sc
    cx = min(max(fx * im.width, cw / 2), im.width - cw / 2)
    cy = min(max(fy * im.height, ch / 2), im.height - ch / 2)
    # odağa doğru hafif kayma
    cx = im.width / 2 + (cx - im.width / 2) * (0.6 + 0.4 * u)
    cy = im.height / 2 + (cy - im.height / 2) * (0.6 + 0.4 * u)
    cx = min(max(cx, cw / 2), im.width - cw / 2)
    cy = min(max(cy, ch / 2), im.height - ch / 2)
    return im.resize((W, H), Image.BILINEAR, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))


def map_frame(name, layers, opts, t, u):
    im = draw_map_layer(name, layers, t)
    z = 1.0 + 0.06 * u + (0.05 if opts.get("shake") else 0)
    cw, ch = W / z, H / z
    ox = oy = 0.0
    if opts.get("shake") and t < opts.get("shake_until", 99):
        amp = 22 * (1 - min(1, t / opts.get("shake_until", 3.0)) * 0.7)
        ox, oy = amp * math.sin(t * 53), amp * math.cos(t * 41)
    ox = max(-(W - cw) / 2, min((W - cw) / 2, ox / z))
    oy = max(-(H - ch) / 2, min((H - ch) / 2, oy / z))
    box = ((W - cw) / 2 + ox, (H - ch) / 2 + oy, (W + cw) / 2 + ox, (H + ch) / 2 + oy)
    return im.resize((W, H), Image.BILINEAR, box=box)


def shot_frame(shot, t, dur):
    u = min(1.0, max(0.0, t / dur))
    kind = shot[0]
    if kind == "img":
        if shot[1] == "kayi_damga":
            return kayi_cached().resize((W, H), Image.BILINEAR, box=_zoombox(1 + 0.08 * u))
        return img_frame(shot[1], shot[2] if len(shot) > 2 else {}, u)
    if kind == "map":
        return map_frame(shot[1], shot[2], shot[3] if len(shot) > 3 else {}, t, u)
    if kind == "card":
        return card_frame(shot[1], shot[2], shot[3], t, u)
    if kind == "dream":
        return dream_frame(t, dur)
    if kind == "dream_night":
        return dream_frame(t, dur, tree=False)
    raise ValueError(kind)


@functools.lru_cache(None)
def kayi_cached():
    return kayi_damga()


def _zoombox(z):
    cw, ch = W / z, H / z
    return ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)


# ------------------------------------------------------------------ yazı katmanları
def static_overlay(ep):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = np.zeros((H, W, 4), np.uint8)
    top = np.clip(1 - np.arange(H) / 420, 0, 1) ** 1.5
    bot = np.clip((np.arange(H) - 950) / 970, 0, 1) ** 1.2
    g[..., 3] = (np.maximum(top * 190, bot * 225))[:, None].astype(np.uint8)
    ov = Image.fromarray(g, "RGBA")
    # vinyet
    vig = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vig).ellipse([-W * 0.35, -H * 0.2, W * 1.35, H * 1.2], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(160))
    va = np.array(vig, np.float32) / 255
    vlay = np.zeros((H, W, 4), np.uint8)
    vlay[..., 3] = ((1 - va) * 150).astype(np.uint8)
    vl = Image.fromarray(vlay, "RGBA")
    ov = Image.alpha_composite(ov, vl)
    d = ImageDraw.Draw(ov)
    badge = f"OSMANLI #{ep['no']:02d}"
    bw = d.textlength(badge, font=FONT_BADGE) + 60
    x0, y0 = 60, 110
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + 76], 14, fill=BORDO + (240,), outline=GOLD + (255,), width=4)
    d.text((x0 + bw / 2, y0 + 40), badge, font=FONT_BADGE, fill=GOLD_L, anchor="mm")
    d.text((x0 + 4, y0 + 112), ep["title"], font=FONT_TITLE, fill=(255, 255, 255), stroke_width=3, stroke_fill=INK, anchor="lm")
    d.text((W - 40, H - 60), "Haritalar ve çizimler temsilidir", font=FONT_SMALL,
           fill=(255, 255, 255, 150), anchor="rm")
    return ov


def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        s = (cur + " " + w_).strip()
        if font.getlength(s) <= maxw or not cur:
            cur = s
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines


def chunk_caption(text, maxw=900):
    """Altyazıyı en fazla 2 satırlık parçalara böl."""
    words = text.split()
    chunks, cur = [], []
    for w_ in words:
        trial = " ".join(cur + [w_])
        if len(wrap(trial, FONT_CAP, maxw)) <= 2 and len(cur) < 6:
            cur.append(w_)
            if w_[-1] in ".?!" and len(cur) >= 2:
                chunks.append(" ".join(cur)); cur = []
        else:
            chunks.append(" ".join(cur)); cur = [w_]
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@functools.lru_cache(None)
def caption_img(text):
    lines = wrap(text, FONT_CAP, 900)
    lh = 86
    im = Image.new("RGBA", (W, lh * len(lines) + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        words = ln.split()
        total = FONT_CAP.getlength(ln)
        x = (W - total) / 2
        y = 20 + i * lh + lh / 2
        for w_ in words:
            hl = any(c.isdigit() for c in w_) or w_.strip(".,!?").isupper() and len(w_) > 2
            col = GOLD_L if hl else (255, 255, 255)
            d.text((x, y), w_, font=FONT_CAP, fill=col, stroke_width=7, stroke_fill=INK, anchor="lm")
            x += FONT_CAP.getlength(w_ + " ")
    return im


@functools.lru_cache(None)
def label_img(text):
    tw = FONT_LABEL.getlength(text)
    im = Image.new("RGBA", (int(tw) + 90, 84), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, tw + 70, 83], fill=(20, 10, 14, 215))
    d.rectangle([0, 0, 12, 83], fill=GOLD)
    d.line([(12, 82), (tw + 70, 82)], fill=GOLD, width=3)
    d.text((40, 44), text, font=FONT_LABEL, fill=GOLD_L, anchor="lm")
    return im


@functools.lru_cache(None)
def big_img(text):
    lines = text.split("\n")
    lh = 150
    im = Image.new("RGBA", (W, lh * len(lines) + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        font = FONT_BIG if FONT_BIG.getlength(ln) < W - 80 else F("Cinzel_900Black.ttf", int(118 * (W - 80) / FONT_BIG.getlength(ln)))
        d.text((W / 2, 30 + i * lh + lh / 2), ln, font=font, fill=GOLD_L, stroke_width=8, stroke_fill=INK, anchor="mm")
    return im


def paste_scaled(base, lay, cx, cy, s, alpha=1.0):
    if s != 1:
        lay = lay.resize((max(1, int(lay.width * s)), max(1, int(lay.height * s))), Image.BILINEAR)
    if alpha < 1:
        a = lay.getchannel("A").point(lambda v: int(v * alpha))
        lay = lay.copy(); lay.putalpha(a)
    base.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))


def outro_frame(ep, t, last):
    im = Image.blend(last, Image.new("RGB", (W, H), BORDO), min(0.93, ease(t / 0.5) * 0.93)).convert("RGBA")
    d = ImageDraw.Draw(im)
    a = ease((t - 0.2) / 0.5)
    nxt_no, nxt_title = ep["next"].split(" — ")
    if a > 0:
        A = int(255 * a)
        d.text((W / 2, 640), "DEVAMI", font=FONT_LABEL, fill=(255, 255, 255, A), anchor="mm")
        d.text((W / 2, 760), nxt_no.upper().replace("OSMANLI", "OSMANLI"), font=F("Cinzel_900Black.ttf", 96),
               fill=GOLD_L + (A,), stroke_width=6, stroke_fill=INK + (A,), anchor="mm")
        for i, ln in enumerate(wrap(nxt_title, F("Montserrat_800ExtraBold.ttf", 60), 940)):
            d.text((W / 2, 880 + i * 76), ln, font=F("Montserrat_800ExtraBold.ttf", 60), fill=(255, 255, 255, A),
                   stroke_width=5, stroke_fill=INK + (A,), anchor="mm")
    b = ease((t - 0.7) / 0.4)
    if b > 0:
        s = 1 + 0.05 * math.sin(t * 7)
        bw, bh = 420 * s, 110 * s
        cx, cy = W / 2, 1150
        d.rounded_rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 20, fill=(204, 0, 0, int(255 * b)))
        d.text((cx, cy), "ABONE OL", font=F("Montserrat_800ExtraBold.ttf", int(56 * s)), fill=(255, 255, 255, int(255 * b)), anchor="mm")
    return im.convert("RGB")


# ------------------------------------------------------------------ açıklama
TAGS = "#osmanlı #tarih #shorts #osmanlıtarihi #türktarihi"


def write_description(ep, out):
    cred_path = os.path.join(ASSETS, "credits.json")
    credits = json.load(open(cred_path)) if os.path.exists(cred_path) else {}
    keys = []
    for sc in ep["scenes"]:
        for sh in sc["shots"]:
            if sh[0] == "img" and sh[1] in credits and sh[1] not in keys:
                keys.append(sh[1])
    lines = [f"Osmanlı #{ep['no']} | {ep['title']}", "",
             " ".join(sc["cap"] for sc in ep["scenes"][1:3]), "",
             f"Bir sonraki bölüm: {ep['next']}",
             "60 bölümde Osmanlı tarihini baştan sona anlatıyoruz. Abone olmayı unutma!", "",
             "Görsel kaynakları (Wikimedia Commons):"]
    for k in keys:
        c = credits[k]
        artist = f" — {c['artist']}" if c.get("artist") else ""
        lines.append(f"• {c['file']}{artist} ({c['license']}) {c['page']}")
    lines += ["• Haritalar ve grafikler kanal tarafından hazırlanmıştır; sınırlar temsilidir.",
              "• Seslendirme: yapay zekâ ile üretilmiştir.", "", TAGS]
    open(out[:-4] + ".txt", "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ montaj
def build(ep):
    wd = os.path.join(BUILD, ep["slug"])
    os.makedirs(wd, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(ep["no"])
    # 1) seslendirme
    timeline, t = [], 0.0
    voices = []
    for i, sc in enumerate(ep["scenes"]):
        wav = os.path.join(wd, f"scene{i}.wav")
        v = tts(sc.get("tts", sc["cap"]), wav)
        lead = 0.25 if i == 0 else 0.12
        dur = lead + len(v) / SR + (0.55 if sc.get("style") == "hook" else 0.3)
        timeline.append(dict(scene=sc, start=t, dur=dur, vstart=t + lead, vlen=len(v) / SR, words=word_times(wav)))
        voices.append((t + lead, v))
        t += dur
    total = t + OUTRO
    # 2) çekimler
    shots = []
    for T in timeline:
        n = len(T["scene"]["shots"])
        for k, sh in enumerate(T["scene"]["shots"]):
            shots.append(dict(shot=sh, start=T["start"] + T["dur"] * k / n, dur=T["dur"] / n))
    # 3) altyazı zamanlaması (karakter uzunluğuna orantılı)
    caps = []
    for T in timeline:
        chunks = chunk_caption(T["scene"]["cap"])
        lens = [len(c) + 1 for c in chunks]
        tot, acc = sum(lens), 0
        bounds = [0.0]
        for L in lens[:-1]:
            acc += L
            bounds.append(time_at_fraction(T["words"], acc / tot, T["vlen"]) - 0.05)
        bounds.append(T["vlen"] + 0.25)
        for k, c in enumerate(chunks):
            caps.append((T["vstart"] + max(0, bounds[k]), T["vstart"] + bounds[k + 1], c))
    # 4) ses miksajı
    n = int(total * SR) + SR
    voice = np.zeros(n, np.float32)
    for st, v in voices:
        i = int(st * SR); voice[i:i + len(v)] += v
    music = make_music(total + 1, ep["no"])[:n]
    music = np.pad(music, (0, n - len(music)))
    # konuşma sırasında müziği kıs (ducking)
    env = np.convolve((np.abs(voice) > 0.02).astype(np.float32), np.ones(int(0.3 * SR)) / int(0.3 * SR), "same")
    duck = 0.16 - 0.09 * np.clip(env * 3, 0, 1)
    tail = int((total - OUTRO) * SR)
    duck[tail:] = np.linspace(duck[tail - 1] if tail > 0 else 0.16, 0.35, len(duck) - tail)
    mix = voice * 1.0 + music * duck
    boom = sfx_boom(); mix[: len(boom)] += boom * 0.5
    for sh in shots[1:]:
        wsh = sfx_whoosh(rng); i = max(0, int((sh["start"] - 0.3) * SR))
        mix[i:i + len(wsh)] += wsh[: len(mix) - i] * 0.25
    mix = mix[: int(total * SR)]
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    apath = os.path.join(wd, "mix.wav")
    with wave.open(apath, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    # 5) görüntü
    out = os.path.join(OUT, f"osmanli-{ep['slug']}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", apath,
                           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out],
                          stdin=subprocess.PIPE)
    static = static_overlay(ep)
    nframes = int(total * FPS)
    main_end = total - OUTRO
    last_main = None
    for fi in range(nframes):
        t = fi / FPS
        if t >= main_end:
            if last_main is None:
                last_main = frame.convert("RGB")
            ff.stdin.write(outro_frame(ep, t - main_end, last_main).tobytes())
            continue
        idx = max(i for i, s in enumerate(shots) if s["start"] <= t + 1e-9)
        S = shots[idx]
        lt = t - S["start"]
        frame = shot_frame(S["shot"], lt, S["dur"])
        if idx > 0 and lt < XFADE:
            P = shots[idx - 1]
            prev = shot_frame(P["shot"], t - P["start"], P["dur"])
            frame = Image.blend(prev, frame, ease(lt / XFADE))
        frame = frame.convert("RGBA")
        frame.alpha_composite(static)
        T = max((x for x in timeline if x["start"] <= t + 1e-9), key=lambda x: x["start"])
        sc, st = T["scene"], t - T["start"]
        if sc.get("style") == "hook":
            bi = big_img(sc["big"])
            p = ease_out(st / 0.6)
            paste_scaled(frame, bi, W / 2, H * 0.43, 1.35 - 0.35 * p, p)
        elif sc.get("label"):
            li = label_img(sc["label"])
            p = ease_out((st - 0.15) / 0.45)
            if p > 0:
                x = int(-li.width + (li.width + 60) * p)
                frame.alpha_composite(li, (max(x, -li.width + 1), 1090))
        for (a, b, c) in caps:
            if a <= t < b:
                ci = caption_img(c)
                p = ease_out((t - a) / 0.14)
                paste_scaled(frame, ci, W / 2, 1290, 0.88 + 0.12 * p, min(1, p * 1.5))
                break
        d = ImageDraw.Draw(frame)
        d.rectangle([0, H - 10, int(W * t / main_end), H], fill=GOLD + (255,))
        ff.stdin.write(frame.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()
    write_description(ep, out)
    print(f"{out}  ({total:.1f} sn)", flush=True)
    return out, total


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] or [e["no"] for e in EPISODES]
    for e in EPISODES:
        if e["no"] in nums:
            build(e)
