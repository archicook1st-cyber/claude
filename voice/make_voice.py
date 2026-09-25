#!/usr/bin/env python3
"""Subtitles -> narration in the BUDA-CAT reference voice style.

Voice = Kokoro-82M stock voice -> pitch/formant shift + wider intonation (Praat)
-> match-EQ towards the reference. Settings live in a profile JSON (profiles/).

  python3 make_voice.py subs.srt                       # -> out/subs.wav + .mp3
  python3 make_voice.py subs.srt --video short.mp4     # also muxes out/short_voiced.mp4
  python3 make_voice.py --text "Wait, did he just say yes?"   # quick preview
"""
import argparse, json, os, re, subprocess, sys, tempfile
import numpy as np, soundfile as sf, parselmouth
from parselmouth.praat import call

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000
MAX_SPEEDUP = 1.35  # beyond this the voice starts to sound rushed


def load_profile(path):
    with open(path) as f:
        return json.load(f)


# ---------- subtitles ----------

def _ts(s):
    h, m, rest = s.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def parse_subs(path):
    """Returns [(start, end, text)]; plain .txt gives one untimed cue per line."""
    raw = open(path, encoding="utf-8-sig").read()
    if path.lower().endswith(".txt"):
        return [(None, None, l.strip()) for l in raw.splitlines() if l.strip()]
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for i, l in enumerate(lines):
            m = re.match(r"\s*([\d:.,]+)\s*-->\s*([\d:.,]+)", l)
            if m:
                a, b = m.group(1), m.group(2)
                a = a if a.count(":") == 2 else "0:" + a
                b = b if b.count(":") == 2 else "0:" + b
                text = " ".join(lines[i + 1:])
                text = re.sub(r"<[^>]+>|\{[^}]+\}", "", text).strip()
                if text:
                    cues.append((_ts(a), _ts(b), text))
                break
    return cues


def group_sentences(cues):
    """Merge short subtitle chunks into sentences so the intonation stays natural."""
    groups, cur = [], []
    for c in cues:
        cur.append(c)
        if c[0] is None or re.search(r"[.!?…]['\")\]]*$", c[2]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        out.append((g[0][0], g[-1][1], " ".join(c[2] for c in g)))
    return out


# ---------- voice colouring ----------

EQ_FREQS = np.geomspace(60, 11000, 48)  # 1/3-octave band centres used by the match-EQ


def shift_voice(a, formant_ratio, pitch_median, expr, lo=110, hi=620):
    """Formant shift + new pitch median (Praat "Change gender"), then widen the
    intonation by `expr` in the semitone domain, clamped so it can't go undefined."""
    snd = parselmouth.Sound(np.asarray(a, dtype=np.float64), SR)
    s = call(snd, "Change gender", 75, 600, formant_ratio, pitch_median, 1.0, 1.0)
    if abs(expr - 1.0) > 1e-3:
        manip = call(s, "To Manipulation", 0.01, 75, 700)
        pt = call(manip, "Extract pitch tier")
        call(pt, "Formula", f"min(max({pitch_median} * (self/{pitch_median})^{expr}, {lo}), {hi})")
        call([pt, manip], "Replace pitch tier")
        s = call(manip, "Get resynthesis (overlap-add)")
    return s.values[0].astype(np.float32)


def band_db(a):
    import librosa
    S = np.abs(librosa.stft(np.asarray(a, dtype=np.float32), n_fft=2048, hop_length=512)) ** 2
    e = S.sum(0)
    S = S[:, e > np.percentile(e, 40)]  # speech frames only
    f, pw = librosa.fft_frequencies(sr=SR, n_fft=2048), S.mean(1)
    return np.array([10 * np.log10(pw[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].mean() + 1e-14)
                     for c in EQ_FREQS])


def eq_curve(a, ref, max_db=12):
    g = band_db(ref) - band_db(a)
    g -= g[(EQ_FREQS > 200) & (EQ_FREQS < 5000)].mean()
    g = np.convolve(np.pad(g, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    return np.clip(g, -max_db, max_db)


def apply_eq(a, gains_db, taps=1025):
    from scipy.signal import firwin2, fftconvolve
    g = np.asarray(gains_db, dtype=np.float64)
    f = np.concatenate([[0], EQ_FREQS, [SR / 2]]) / (SR / 2)
    h = firwin2(taps, f, 10 ** (np.concatenate([[g[0]], g, [g[-1]]]) / 20))
    return fftconvolve(np.asarray(a, dtype=np.float64), h, mode="same").astype(np.float32)


# ---------- synthesis ----------

class Voice:
    def __init__(self, profile):
        from kokoro_onnx import Kokoro
        self.p = profile
        m = os.path.join(HERE, "models")
        self.k = Kokoro(os.path.join(m, "kokoro-quantized.onnx"), os.path.join(m, "voices.npz"))
        mix = profile["voice_mix"]
        self.style = sum(w * self.k.voices[n] for n, w in mix.items()) / sum(mix.values())

    def raw(self, text, speed):
        a, sr = self.k.create(text, voice=self.style, speed=speed, lang=self.p.get("lang", "en-us"))
        assert sr == SR
        return a

    def color(self, a):
        p = self.p
        a = shift_voice(a, p["formant_shift"], p["pitch_median_hz"], p["expressiveness"])
        if p.get("eq_gains_db"):
            a = apply_eq(a, p["eq_gains_db"])
        return a

    def say(self, text, speed=None):
        return trim(self.color(self.raw(text, speed or self.p["speed"])))


def trim(a, thresh_db=-45, pad=0.03):
    env = np.abs(a)
    thr = env.max() * 10 ** (thresh_db / 20) if env.size else 0
    idx = np.where(env > thr)[0]
    if idx.size == 0:
        return a
    s = max(0, idx[0] - int(pad * SR))
    e = min(len(a), idx[-1] + int(pad * SR))
    return a[s:e]


def atempo(a, factor):
    if abs(factor - 1) < 0.01:
        return a
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "i.wav"), os.path.join(d, "o.wav")
        sf.write(i, a, SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-filter:a",
                        f"atempo={factor:.4f}", o], check=True)
        b, _ = sf.read(o, dtype="float32")
    return b


def fit(voice, text, slot):
    """Synthesize text so it fits in `slot` seconds (None = no limit)."""
    base = voice.p["speed"]
    a = voice.say(text, base)
    if slot is None or len(a) / SR <= slot:
        return a, 1.0
    need = len(a) / SR / slot
    # re-synthesize faster first (sounds more natural than stretching)
    sp = min(base * need * 1.02, base * MAX_SPEEDUP)
    a = voice.say(text, sp)
    ratio = sp / base
    rest = len(a) / SR / slot
    if rest > 1:
        # Kokoro's speed isn't exactly linear; close the gap by stretching,
        # allowing ~10% past the cap before giving up and overflowing
        extra = min(rest, MAX_SPEEDUP / ratio * 1.1)
        a = atempo(a, extra)
        ratio *= extra
    return a, ratio


def render(items, voice, gap):
    """items: [(start, end, text)] -> (audio, report)"""
    timed = items and items[0][0] is not None
    out = np.zeros(int(SR * ((items[-1][1] + 2) if timed else 1)), np.float32)
    report, cursor = [], 0.0
    for n, (s, e, text) in enumerate(items):
        if timed:
            nxt = items[n + 1][0] if n + 1 < len(items) else None
            slot = (nxt - s - gap) if nxt is not None else max(e - s, 0.1) + 1.5
            slot = max(slot, e - s)
            start = max(s, cursor)
        else:
            slot, start = None, cursor
        a, ratio = fit(voice, text, slot)
        i = int(start * SR)
        if i + len(a) > len(out):
            out = np.concatenate([out, np.zeros(i + len(a) - len(out) + SR, np.float32)])
        out[i:i + len(a)] += a
        cursor = start + len(a) / SR + (gap if timed else voice.p.get("sentence_pause", 0.25))
        over = (start + len(a) / SR) - (s + slot) if timed else 0
        report.append((start, len(a) / SR, ratio, over, text))
        print(f"[{start:6.2f}s] {len(a)/SR:5.2f}s x{ratio:.2f}"
              + (f"  OVERFLOW {over:.2f}s" if over > 0.05 else "") + f"  {text}", flush=True)
    end = int((cursor + 0.3) * SR)
    if timed:
        end = max(end, int((items[-1][1] + 0.3) * SR))
    return out[:end], report


def master(a, lufs, path):
    """Loudness-normalize with a true-peak limit and write a 48 kHz WAV."""
    with tempfile.TemporaryDirectory() as d:
        i = os.path.join(d, "i.wav")
        sf.write(i, a, SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-af",
                        f"loudnorm=I={lufs}:TP=-1.5:LRA=11", "-ar", "48000", path], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="?", help=".srt / .vtt / .txt (one sentence per line)")
    ap.add_argument("--text", help="speak this text instead of a subtitle file")
    ap.add_argument("--video", help="mux the narration into this video")
    ap.add_argument("--keep-original", type=float, default=0.0,
                    help="volume of the video's original audio under the voice (0 = drop it)")
    ap.add_argument("--profile", default=os.path.join(HERE, "profiles", "A_river.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--name")
    ap.add_argument("--no-group", action="store_true", help="speak each subtitle cue separately")
    args = ap.parse_args()
    if not args.subs and not args.text:
        ap.error("give a subtitle file or --text")

    prof = load_profile(args.profile)
    voice = Voice(prof)
    if args.text:
        items = [(None, None, t.strip()) for t in re.split(r"(?<=[.!?])\s+", args.text) if t.strip()]
        name = args.name or "preview"
    else:
        items = parse_subs(args.subs)
        if not args.no_group:
            items = group_sentences(items)
        name = args.name or os.path.splitext(os.path.basename(args.subs))[0]

    audio, report = render(items, voice, prof.get("min_gap", 0.08))
    os.makedirs(args.out, exist_ok=True)
    wav = os.path.join(args.out, name + ".wav")
    master(audio, prof.get("target_lufs", -16), wav)
    mp3 = wav[:-4] + ".mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-ar", "44100",
                    "-b:a", "192k", mp3], check=True)
    print("->", wav, "\n->", mp3)

    if args.video:
        vid_out = os.path.join(args.out, os.path.splitext(os.path.basename(args.video))[0] + "_voiced.mp4")
        if args.keep_original > 0:
            filt = (f"[0:a]volume={args.keep_original}[bg];[1:a]aresample=48000[v];"
                    "[bg][v]amix=inputs=2:duration=first:normalize=0[a]")
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", args.video, "-i", wav,
                   "-filter_complex", filt, "-map", "0:v", "-map", "[a]"]
        else:
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", args.video, "-i", wav,
                   "-map", "0:v", "-map", "1:a", "-af", "apad", "-shortest"]
        subprocess.run(cmd + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", vid_out], check=True)
        print("->", vid_out)

    overs = [r for r in report if r[3] > 0.05]
    if overs:
        print(f"\n{len(overs)} line(s) could not fully fit their subtitle slot (see OVERFLOW above).")


if __name__ == "__main__":
    main()
