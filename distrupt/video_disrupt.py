from pathlib import Path
import json
import math
import re
import shutil
import subprocess
import tempfile
import time

from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_cdt

POLA_PER_DETIK = 30              # 30, 60, 120
PENUTUPAN = 70                # 0-100

REF_RESOLUTION = 720          # resolusi acuan
REF_BASE_PX = 80              # ukuran dasar kotak di 720px
REF_STEP_PX = 10.0            # interval garis di 720px

WARNA_POLA = ["#87A3A3", "#9F90AD", "#B3A089"] 
TEKS_HURUF = "saya ingin makan nasi padang"          
UKURAN_HURUF_PERSEN = 55.0     # 0-100
CRF = 18                       
PRESET = "fast"
SEED = None                    
MUTE = True                  

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "output"
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def color_palette():
    if len(WARNA_POLA) != 3:
        raise ValueError("WARNA_POLA harus berisi tiga warna HEX.")
    colors = []
    for color in WARNA_POLA:
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError(f"Warna tidak valid: {color!r}. Gunakan HEX seperti '#919191'.")
        colors.append([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    return np.asarray(colors, dtype=np.uint8)


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    width, height = video["width"], video["height"]
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for item in video.get("side_data_list", []):
        rotation = float(item.get("rotation", rotation))
    if round(rotation) % 180 == 90:
        width, height = height, width
    duration = float(video.get("duration", data.get("format", {}).get("duration", 0)))
    return width, height, duration


GLYPHS = {
    "O": ["11111", "10001", "10001", "10001", "10001", "10001", "11111"],
    "R": ["11111", "10001", "10001", "11111", "10110", "10011", "10001"],
    "A": ["11111", "10001", "10001", "11111", "10001", "10001", "10001"],
    "N": ["10001", "11001", "11101", "10111", "10011", "10011", "10001"],
    "G": ["11111", "10000", "10000", "10111", "10001", "10001", "11111"],
    'B': ['11110', '10011', '10011', '11110', '10011', '10011', '11110'],
    'C': ['11111', '10000', '10000', '10000', '10000', '10000', '11111'],
    'D': ['11110', '10011', '10001', '10001', '10001', '10011', '11110'],
    'E': ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
    'F': ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
    'H': ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
    'I': ['11111', '00100', '00100', '00100', '00100', '00100', '11111'],
    'J': ['11111', '00010', '00010', '00010', '10010', '10010', '11110'],
    'K': ['10011', '10110', '11100', '11000', '11100', '10110', '10011'],
    'L': ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
    'M': ['11011', '11111', '10101', '10101', '10001', '10001', '10001'],
    'P': ['11111', '10001', '10001', '11111', '10000', '10000', '10000'],
    'Q': ['11111', '10001', '10001', '10001', '10101', '10011', '11111'],
    'S': ['11111', '10000', '10000', '11111', '00001', '00001', '11111'],
    'T': ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
    'U': ['10001', '10001', '10001', '10001', '10001', '10001', '11111'],
    'V': ['10001', '10001', '10001', '11011', '01010', '01110', '00100'],
    'W': ['10001', '10001', '10001', '10101', '10101', '11111', '11011'],
    'X': ['10001', '11011', '01110', '00100', '01110', '11011', '10001'],
    'Y': ['10001', '11011', '01110', '00100', '00100', '00100', '00100'],
    'Z': ['11111', '00011', '00110', '01100', '11000', '10000', '11111'],
    '0': ['11111', '10011', '10111', '10101', '11101', '11001', '11111'],
    '1': ['00100', '01100', '00100', '00100', '00100', '00100', '11111'],
    '2': ['11111', '00001', '00001', '11111', '10000', '10000', '11111'],
    '3': ['11111', '00001', '00001', '01111', '00001', '00001', '11111'],
    '4': ['10001', '10001', '10001', '11111', '00001', '00001', '00001'],
    '5': ['11111', '10000', '10000', '11111', '00001', '00001', '11111'],
    '6': ['11111', '10000', '10000', '11111', '10001', '10001', '11111'],
    '7': ['11111', '00001', '00011', '00110', '00100', '00100', '00100'],
    '8': ['11111', '10001', '10001', '11111', '10001', '10001', '11111'],
    '9': ['11111', '10001', '10001', '11111', '00001', '00001', '11111'],
}


def letter_sequence():
    if not isinstance(TEKS_HURUF, str):
        raise ValueError('TEKS_HURUF harus berupa teks dalam tanda kutip.')
    text = "".join(ch for ch in TEKS_HURUF if not ch.isspace())
    if not text:
        raise ValueError("TEKS_HURUF tidak boleh kosong atau hanya spasi.")
    invalid = sorted(set(ch for ch in text if ch not in
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))
    if invalid:
        raise ValueError(f"Karakter belum didukung: {', '.join(repr(ch) for ch in invalid)}. "
                         "Gunakan huruf A–Z, a–z, angka 0–9, dan spasi.")
    return text.upper()


@lru_cache(maxsize=64)
def letter_distance(character, cell):
    glyph = np.array([[v == "1" for v in row] for row in GLYPHS[character]])
    mask = np.repeat(np.repeat(glyph, cell, axis=0), cell, axis=1)
    distance = distance_transform_cdt(np.pad(mask, 1), metric="chessboard")[1:-1, 1:-1]
    return glyph, mask, distance.astype(np.float32) - 0.5


def ring_mask(distance, step):
    thickness = step * PENUTUPAN / 100
    margin = (step - thickness) / 2
    return (distance >= margin) & ((distance - margin) % step < thickness)


def layered_pattern(width, height, rng, frame_index=0, position=None):
    short = min(width, height)
    long_side = max(width, height)

    # Scale berdasarkan sisi terpanjang (referensi 720px)
    scale = long_side / REF_RESOLUTION
    pattern_base_px = max(8, round(REF_BASE_PX * scale))
    pattern_step_px = max(2.0, REF_STEP_PX * scale)

    cell = max(1, round(short * UKURAN_HURUF_PERSEN / 700))
    cell = min(cell, max(1, width // 7), max(1, height // 9))
    rows, cols = math.ceil(height / cell), math.ceil(width / cell)
    if rows < 9 or cols < 7:
        raise ValueError("Resolusi video terlalu kecil untuk menampilkan huruf.")
    character = letter_sequence()[frame_index % len(letter_sequence())]
    glyph, letter_mask, distance = letter_distance(character, cell)
    pos = int(rng.integers(9)) if position is None else position
    max_col = max(1, width // cell - 6)
    max_row = max(1, height // cell - 8)
    col = (1, (1 + max_col) // 2, max_col)[pos % 3]
    row = (1, (1 + max_row) // 2, max_row)[pos // 3]
    used = np.zeros((rows, cols), dtype=bool)
    used[row:row + 7, col:col + 5] = glyph
    result = np.full((height, width), -1, dtype=np.int16)
    letter_color = int(rng.integers(3))
    other_colors = [i for i in range(3) if i != letter_color]

    step = pattern_step_px
    letter_step = max(2.0, min(step, cell / 6))
    rings = letter_mask & ring_mask(distance, letter_step)
    result[row * cell:(row + 7) * cell, col * cell:(col + 5) * cell][rings] = letter_color
    base = max(1, round(pattern_base_px / cell))
    choices = list(dict.fromkeys([
        (base, base), (base, 2 * base), (2 * base, base),
        (1, 3), (3, 1), (2, 2), (1, 2), (2, 1), (1, 1),
    ]))
    for r in range(rows):
        for c in range(cols):
            if used[r, c]:
                continue
            valid = [(rh, cw) for rh, cw in choices
                     if r + rh <= rows and c + cw <= cols
                     and not used[r:r + rh, c:c + cw].any()]
            rh, cw = valid[int(rng.integers(len(valid)))]
            used[r:r + rh, c:c + cw] = True
            x0, x1 = c * cell, min(width, (c + cw) * cell)
            y0, y1 = r * cell, min(height, (r + rh) * cell)
            w, h = x1 - x0, y1 - y0
            dx = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5)
            dy = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5)
            d = np.minimum(dy[:, None], dx[None, :])
            local_step = max(2.0, min(step, min(w, h) / 4))
            color = other_colors[int(rng.integers(2))]
            result[y0:y1, x0:x1][ring_mask(d, local_step)] = color
    return result


def read_frame(pipe, size):
    data = bytearray()
    while len(data) < size:
        chunk = pipe.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    if data and len(data) != size:
        raise RuntimeError("Frame tidak lengkap dari decoder.")
    return data


def destination(source):
    mute_tag = "_mute" if MUTE else ""
    name = f"{source.stem}_{source.suffix[1:]}_berlapis_{POLA_PER_DETIK}fps_{PENUTUPAN}pct{mute_tag}"
    path = OUTPUT / f"{name}.mp4"
    number = 2
    while path.exists():
        path = OUTPUT / f"{name}_{number}.mp4"
        number += 1
    return path


def process(source):
    width, height, duration = probe(source)
    target = destination(source)
    palette = color_palette()
    rng = np.random.default_rng(SEED)
    count = 0
    start = last_log = time.monotonic()
    decoder = encoder = None
    with tempfile.TemporaryDirectory(prefix="disrupt_", dir=OUTPUT) as work:
        work = Path(work)
        silent = work / "silent.mp4"
        final = work / "final.mp4"
        with (work / "decode.log").open("w+b") as dlog, (work / "encode.log").open("w+b") as elog:
            try:
                decoder = subprocess.Popen([
                    "ffmpeg", "-v", "error", "-nostdin", "-i", str(source),
                    "-map", "0:v:0", "-an", "-sn", "-vf", f"fps={POLA_PER_DETIK}",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                ], stdout=subprocess.PIPE, stderr=dlog)
                encoder = subprocess.Popen([
                    "ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(POLA_PER_DETIK),
                    "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
                    "-pix_fmt", "yuv420p", str(silent),
                ], stdin=subprocess.PIPE, stderr=elog)
                while True:
                    raw = read_frame(decoder.stdout, width * height * 3)
                    if not raw:
                        break
                    frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
                    pattern = layered_pattern(width, height, rng, count)
                    mask = pattern >= 0
                    frame[mask] = palette[pattern[mask]]
                    encoder.stdin.write(frame.tobytes())
                    count += 1
                    now = time.monotonic()
                    if now - last_log >= 0.5:
                        percent = min(99.9, count / (duration * POLA_PER_DETIK) * 100) if duration else 0
                        print(f"\r{percent:5.1f}% | {count} frame | {count / (now-start):.1f} frame/s", end="", flush=True)
                        last_log = now
                encoder.stdin.close()
                if decoder.wait() != 0 or encoder.wait() != 0 or not count:
                    raise RuntimeError("FFmpeg gagal memproses video.")
            except BaseException as exc:
                for proc in (decoder, encoder):
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                        proc.wait()
                dlog.seek(0)
                elog.seek(0)
                detail = (dlog.read() + elog.read()).decode(errors="replace")
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise RuntimeError(f"{exc}\n{detail[-4000:]}") from exc
            finally:
                if decoder is not None and decoder.stdout:
                    decoder.stdout.close()
        if MUTE:
            print("\rVideo selesai; tanpa audio (mute)...                 ", flush=True)
            result = subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(silent),
                "-c:v", "copy", "-an",
                "-t", str(count / POLA_PER_DETIK), "-movflags", "+faststart", str(final),
            ], capture_output=True, text=True)
        else:
            print("\rVideo selesai; memasukkan audio...                 ", flush=True)
            result = subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(silent), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-t", str(count / POLA_PER_DETIK), "-movflags", "+faststart", str(final),
            ], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        final.rename(target)
    print(f"100% | {count / POLA_PER_DETIK:.3f} detik | {target.name}\n")
    return target


def main():
    if POLA_PER_DETIK not in (30, 60, 120):
        raise ValueError("POLA_PER_DETIK harus 30, 60, atau 120.")
    if not 0 <= PENUTUPAN <= 100:
        raise ValueError("Penutupan harus 0–100 persen.")
    if not 20 <= REF_BASE_PX <= 200:
        raise ValueError("REF_BASE_PX harus 20–200.")
    if not 4.0 <= REF_STEP_PX <= 30.0:
        raise ValueError("REF_STEP_PX harus 4.0–30.0.")
    letter_sequence()
    if not 15 <= UKURAN_HURUF_PERSEN <= 75:
        raise ValueError("Ukuran huruf harus 15–75 persen.")
    color_palette()
    if not 0 <= CRF <= 51:
        raise ValueError("CRF harus 0–51.")
    if not isinstance(MUTE, bool):
        raise ValueError("MUTE harus True atau False.")
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("Install FFmpeg beserta ffprobe, lalu tambahkan ke PATH.")
    files = sorted(p for p in FOLDER.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        print("Tidak ada video di folder script.")
        return
    OUTPUT.mkdir(exist_ok=True)
    failed = 0
    for i, source in enumerate(files, 1):
        mute_info = "mute" if MUTE else "dengan audio"
        print(f"[{i}/{len(files)}] {source.name} | berlapis | {POLA_PER_DETIK} pola/detik | {mute_info}")
        try:
            process(source)
        except Exception as exc:
            failed += 1
            print(f"\nGAGAL: {exc}\n")
    print(f"Selesai: {len(files)-failed} berhasil, {failed} gagal. Video asli tidak diubah.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDibatalkan.")
    except Exception as exc:
        raise SystemExit(str(exc))