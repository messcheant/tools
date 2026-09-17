import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy import signal

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "blip_output"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma", ".aiff", ".aif", ".m4b", ".ac3", ".mp4"}
VARIANT_COUNT = 5
OUTPUT_DURATION = 5.0
BLIP_MS = 80.0
GAP_MS = 55.0
PITCH_SEMITONES = 2.0
SAMPLE_RATE = 44100


def read_audio(path):
    try:
        if path.suffix.lower() != ".wav":
            raise ValueError("Gunakan decoder FFmpeg.")
        rate, audio = wavfile.read(path)
        if audio.dtype == np.uint8:
            audio = audio.astype(np.float64) - 128
        else:
            audio = audio.astype(np.float64)
        if audio.ndim == 2:
            audio = audio[:, np.argmax(np.mean(audio ** 2, axis=0))]
    except (RuntimeError, ValueError):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                raise ValueError(
                    "Input MP3/M4A/FLAC/OGG/AAC/format lain membutuhkan FFmpeg. "
                    "Jalankan: python -m pip install imageio-ffmpeg"
                ) from None
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
             "-map", "0:a:0", "-vn", "-f", "f64le",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            capture_output=True,
        )
        if result.returncode:
            raise ValueError("FFmpeg gagal membaca file: " + result.stderr.decode(errors="replace"))
        audio = np.frombuffer(result.stdout, dtype="<f8").copy()
        rate = SAMPLE_RATE
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio kosong atau mengandung data tidak valid.")
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = signal.resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    audio = audio - np.mean(audio)
    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError("Audio hanya berisi hening atau terlalu pelan.")
    return audio / np.max(np.abs(audio))


def select_bases(audio, length, count):
    if len(audio) < length * count:
        raise ValueError(
            f"Input terlalu pendek untuk {count} potongan berbeda. "
            f"Gunakan rekaman bersuara minimal {length * count / SAMPLE_RATE:.2f} detik."
        )
    candidates = []
    energy_floor = max(1e-5, float(np.mean(audio ** 2)) * 0.05)
    for start in range(0, len(audio) - length + 1, length):
        frame = audio[start:start + length]
        energy = np.mean(frame ** 2)
        if energy < energy_floor:
            continue
        small = frame[::4]
        corr = signal.correlate(small, small, mode="full", method="fft")[len(small)-1:]
        low = max(1, SAMPLE_RATE // 4 // 500)
        high = min(len(corr), SAMPLE_RATE // 4 // 65)
        periodicity = np.max(corr[low:high]) / (corr[0] + 1e-12) if high > low else 0
        score = np.sqrt(energy) * (0.2 + max(0, periodicity))
        candidates.append((start, score))
    if len(candidates) < count:
        raise ValueError(
            f"Hanya ditemukan {len(candidates)} potongan bersuara yang terpisah; "
            f"dibutuhkan {count}. Gunakan rekaman suara lebih panjang/jelas."
        )
    groups = np.array_split(np.arange(len(candidates)), count)
    selected = [max((candidates[i] for i in group), key=lambda item: item[1])[0]
                for group in groups]
    return [(start, audio[start:start + length].copy()) for start in selected]


def make_blip(base, length, semitones):
    ratio = 2 ** (semitones / 12)
    positions = np.arange(length) * ratio
    blip = np.interp(positions, np.arange(len(base)), base)
    sos = signal.butter(2, [100, 6500], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    blip = signal.sosfilt(sos, blip)
    blip = np.tanh(blip * 1.5)
    fade = min(round(0.008 * SAMPLE_RATE), length // 3)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    blip[:fade] *= ramp
    blip[-fade:] *= ramp[::-1]
    rms = np.sqrt(np.mean(blip ** 2))
    blip *= min(0.18 / max(rms, 1e-12), 0.85 / max(np.max(np.abs(blip)), 1e-12))
    return blip


def main():
    parser = argparse.ArgumentParser(description="Suara biasa menjadi 5 blip dari waktu berbeda dan preview WAV.")
    parser.add_argument("input", nargs="?", default=str(INPUT_DIR),
                        help="Folder audio atau satu file audio; default folder input di sebelah script")
    parser.add_argument("--duration", type=float, default=OUTPUT_DURATION, help="Durasi preview dalam detik")
    parser.add_argument("--blip-ms", type=float, default=BLIP_MS)
    parser.add_argument("--gap-ms", type=float, default=GAP_MS)
    parser.add_argument("--pitch", type=float, default=PITCH_SEMITONES, help="Perubahan pitch dalam semitone")
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    values = [args.duration, args.blip_ms, args.gap_ms, args.pitch]
    if not all(math.isfinite(v) for v in values):
        parser.error("Pengaturan harus berupa angka finite.")
    if not 20 <= args.blip_ms <= 300 or not 0 <= args.gap_ms <= 2000:
        parser.error("Panjang blip harus 20–300 ms; jeda 0–2000 ms.")
    if not args.blip_ms / 1000 <= args.duration <= 600 or not -12 <= args.pitch <= 12:
        parser.error("Durasi harus minimal sepanjang blip dan maksimal 600 detik; pitch -12 sampai 12.")
    input_path = Path(args.input).expanduser().resolve()
    if input_path == INPUT_DIR:
        input_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        sources = sorted(
            (p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
        if not sources:
            print(f"Belum ada file audio. Masukkan suara ke: {input_path}")
            return
    elif input_path.is_file():
        sources = [input_path]
    else:
        parser.error(f"File/folder tidak ditemukan: {input_path}")
    failed = 0
    for index, source in enumerate(sources, 1):
        print(f"\n[{index}/{len(sources)}] {source.name}", flush=True)
        try:
            process_audio(source, args)
        except (ValueError, OSError) as error:
            failed += 1
            print(f"Gagal: {source.name}: {error}", file=sys.stderr)
    print(f"\nSelesai: {len(sources) - failed} berhasil, {failed} gagal.")
    if failed:
        sys.exit(1)


def process_audio(source, args):
    print("[1/4] Membaca suara input...", flush=True)
    audio = read_audio(source)
    length = round(args.blip_ms * SAMPLE_RATE / 1000)
    needed = max(length, math.ceil((length - 1) * 2 ** (args.pitch / 12)) + 2)
    print(f"[2/4] Memilih {VARIANT_COUNT} bagian suara dari waktu berbeda...", flush=True)
    bases = select_bases(audio, needed, VARIANT_COUNT)
    blips = []
    for index, (start, base) in enumerate(bases, 1):
        blips.append(make_blip(base, length, args.pitch))
        print(f"  blip_{index:02d}: sumber {start / SAMPLE_RATE:.3f}–{(start + needed) / SAMPLE_RATE:.3f} detik")
    print("[3/4] Menyusun preview...", flush=True)
    rng = np.random.default_rng(args.seed)
    preview = np.zeros(round(args.duration * SAMPLE_RATE))
    position, count = 0, 0
    order = []
    while position + length <= len(preview):
        if not order:
            order = rng.permutation(len(blips)).tolist()
        preview[position:position + length] = blips[order.pop()]
        gap = args.gap_ms * rng.uniform(0.8, 1.2) / 1000
        count += 1
        if count % 7 == 0:
            gap += 0.16
        position += length + round(gap * SAMPLE_RATE)
    output = Path(args.output).expanduser().resolve() / source.name
    names = [f"blip_{i + 1:02d}.wav" for i in range(len(blips))] + [f"preview_{args.duration:g}s.wav"]
    if source in [output / name for name in names]:
        raise ValueError("Folder output tidak boleh menimpa suara input.")
    output.mkdir(parents=True, exist_ok=True)
    # Each run gets a new folder if output names already exist.
    if any((output / name).exists() for name in names):
        index = 1
        while (output / f"run_{index:03d}").exists():
            index += 1
        output = output / f"run_{index:03d}"
        output.mkdir()
    for name, data in zip(names, [*blips, preview]):
        wavfile.write(output / name, SAMPLE_RATE, np.round(np.clip(data, -1, 1) * 32767).astype(np.int16))
    print(f"[4/4] Selesai: {count} blip, preview {args.duration:g} detik.")
    for name in names:
        print(output / name)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)