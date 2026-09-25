#!/usr/bin/env python3
"""Subtitles -> narration in the BUDA-CAT reference voice style.

Engines (picked by the profile JSON in profiles/):
  chatterbox - zero-shot clone of the reference narrator (default, profiles/clone_narrator.json)
  kokoro     - Kokoro-82M stock voice coloured towards the reference (A/B/C profiles, offline)
Both then get a pitch/formant move (Praat) and optional match-EQ.

  python3 make_voice.py subs.srt                       # -> out/subs.wav + .mp3
  python3 make_voice.py subs.srt --video short.mp4     # -> out/subs_voice.*, subs_voice+bgm.*, subs_preview.mp4
  python3 make_voice.py --text "Wait, did he just say yes?"   # quick preview
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile
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


# ---------- text prep ----------

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 1_000_000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def _money(m):
    d, c = int(m.group(1).replace(",", "")), int(m.group(2) or 0)
    if d == 1 and c:
        return "a dollar " + num_words(c)  # $1.50 -> a dollar fifty
    dollars = f"{num_words(d)} dollar{'' if d == 1 else 's'}"
    return f"{dollars} {num_words(c)}" if c else dollars


def normalize_for_tts(text):
    """Spell out money/numbers and drop symbols the TTS would stumble on."""
    t = text.replace("’", "'").replace("‘", "'")
    t = re.sub(r"\$(\d[\d,]*)(?:\.(\d\d))?\b", _money, t)
    t = re.sub(r"\b\d{1,6}\b", lambda m: num_words(m.group()), t)
    t = re.sub(r"[\"“”]", "", t)
    t = t.replace("~", "!").replace("…", "...")
    t = re.sub(r"\b[A-Z]{2,}\b", lambda m: m.group() if m.group() in ("OK", "ID", "VIP") else m.group().lower(), t)
    t = re.sub(r"\s+", " ", t).strip()
    # capitalize after . ! ? (but not after "..." which reads as a trailing-off continuation)
    t = re.sub(r"(?<!\.)([.!?])(\s+)([a-z])", lambda m: m.group(1) + m.group(2) + m.group(3).upper(), t)
    return t[:1].upper() + t[1:]


def split_sentences(text, min_words=4):
    """Split a cue into sentences; very short ones ("Huh?") ride along with the next."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    out, carry = [], ""
    for p in parts:
        cur = (carry + " " + p).strip() if carry else p
        if len(cur.split()) < min_words:
            carry = cur
        else:
            out.append(cur[:1].upper() + cur[1:])
            carry = ""
    if carry:
        if out:
            out[-1] += " " + carry
        else:
            out.append(carry[:1].upper() + carry[1:])
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

def color(a, p):
    """Apply the profile's pitch/formant move and match-EQ to raw TTS audio."""
    target = p.get("pitch_median_hz")
    if target:
        strength = p.get("pitch_normalize", 1.0)
        pitch = p.get("pitch_median_hz")
        if strength < 1.0:  # pull each line only part of the way to the target
            f0 = parselmouth.Sound(np.asarray(a, dtype=np.float64), SR).to_pitch_ac(0.01, 75, 650)
            v = f0.selected_array["frequency"]
            v = v[v > 0]
            if v.size:
                m = float(np.median(v))
                pitch = m * (target / m) ** strength
        a = shift_voice(a, p.get("formant_shift", 1.0), pitch, p.get("expressiveness", 1.0))
    if p.get("eq_gains_db"):
        a = apply_eq(a, p["eq_gains_db"])
    return a


class KokoroVoice:
    """Kokoro-82M stock voice (optionally a blend), coloured towards the reference."""
    resynth = True  # can re-synthesize faster when a line doesn't fit

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

    def say(self, text, speed=None):
        return trim(color(self.raw(text, speed or self.p["speed"]), self.p))

    def synth_all(self, texts):
        return [self.say(t) for t in texts]


class CloneVoice:
    """Zero-shot clone of the reference narrator with Chatterbox (runs in cb_venv)."""
    resynth = False

    def __init__(self, profile):
        self.p = profile
        self.python = os.path.join(HERE, "cb_venv", "bin", "python")
        if not os.path.exists(self.python):
            sys.exit("cb_venv missing - run ./setup.sh first")

    def _key(self, text):
        p, ref = self.p, os.path.join(HERE, self.p["ref_audio"])
        with open(ref, "rb") as f:
            ref_hash = hashlib.sha1(f.read()).hexdigest()
        parts = [text, ref_hash, p["exaggeration"], p["cfg_weight"], p.get("seed", 7)]
        return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]

    def synth_all(self, texts):
        """Raw clone audio is cached per line, so re-runs only generate what changed."""
        p = self.p
        cache = os.path.join(HERE, ".cache", "clone")
        os.makedirs(cache, exist_ok=True)
        keys = [self._key(t) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if not os.path.exists(os.path.join(cache, k + ".wav"))}
        print(f"{len(texts) - len(todo)} of {len(texts)} lines cached, generating {len(todo)}", flush=True)
        if todo:
            with tempfile.TemporaryDirectory() as d:
                job = {"ref": os.path.join(HERE, p["ref_audio"]), "exaggeration": p["exaggeration"],
                       "cfg_weight": p["cfg_weight"], "seed": p.get("seed", 7),
                       "max_tries": p.get("max_tries", 3), "max_wer": p.get("max_wer", 0.1), "out_dir": d,
                       "items": [{"id": k, "text": t} for k, t in todo.items()]}
                jp = os.path.join(d, "job.json")
                with open(jp, "w") as f:
                    json.dump(job, f)
                subprocess.run([self.python, os.path.join(HERE, "cb_worker.py"), jp], check=True)
                for r in json.load(open(os.path.join(d, "results.json"))):
                    if r["wer"] > p.get("max_wer", 0.1):
                        print(f"  ! still differs after retries: {r['text']!r} -> heard {r['heard']!r}", flush=True)
                    os.replace(r["path"], os.path.join(cache, r["id"] + ".wav"))
        out = []
        for k in keys:
            a, sr = sf.read(os.path.join(cache, k + ".wav"), dtype="float32")
            assert sr == SR, sr
            out.append(trim(color(a, p)))
        return out


def make_voice(profile):
    return CloneVoice(profile) if profile.get("engine") == "chatterbox" else KokoroVoice(profile)


def trim(a, thresh_db=-45, pad=0.03):
    env = np.abs(a)
    thr = env.max() * 10 ** (thresh_db / 20) if env.size else 0
    idx = np.where(env > thr)[0]
    if idx.size == 0:
        return a
    s = max(0, idx[0] - int(pad * SR))
    e = min(len(a), idx[-1] + int(pad * SR))
    return a[s:e]


def stretch(a, factor):
    """Speed speech up by `factor` without changing pitch (Rubber Band, else atempo)."""
    if abs(factor - 1) < 0.01:
        return a
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "i.wav"), os.path.join(d, "o.wav")
        sf.write(i, a, SR)
        for filt in (f"rubberband=tempo={factor:.4f}", f"atempo={factor:.4f}"):
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-filter:a", filt, o])
            if r.returncode == 0:
                break
        else:
            raise RuntimeError("ffmpeg time-stretch failed")
        b, _ = sf.read(o, dtype="float32")
    return b


def fit(voice, text, slot, a):
    """Make pre-synthesized line `a` fit in `slot` seconds (None = no limit)."""
    if slot is None or len(a) / SR <= slot:
        return a, 1.0
    if not voice.resynth:
        extra = min(len(a) / SR / slot, MAX_SPEEDUP * 1.1)
        return stretch(a, extra), extra
    base = voice.p["speed"]
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
        a = stretch(a, extra)
        ratio *= extra
    return a, ratio


def join_sentences(parts, sentences, pause):
    """Concatenate one cue's sentences with a natural pause (longer after '...')."""
    out = []
    for i, (a, s) in enumerate(zip(parts, sentences)):
        out.append(a)
        if i < len(parts) - 1:
            out.append(np.zeros(int(SR * (pause * 1.6 if s.endswith("...") else pause)), np.float32))
    return np.concatenate(out)


def render(items, voice, gap, duration=None):
    """items: [(start, end, text)] -> (audio, report). `duration` caps the timeline (e.g. video length)."""
    timed = items and items[0][0] is not None
    out = np.zeros(int(SR * ((items[-1][1] + 2) if timed else 1)), np.float32)
    report, cursor = [], 0.0
    pause = voice.p.get("sentence_pause", 0.25)
    sents = [split_sentences(normalize_for_tts(t)) for _, _, t in items]
    flat = voice.synth_all([s for ss in sents for s in ss])
    lines, k = [], 0
    for ss in sents:
        lines.append(join_sentences(flat[k:k + len(ss)], ss, pause))
        k += len(ss)
    for n, (s, e, text) in enumerate(items):
        if timed:
            nxt = items[n + 1][0] if n + 1 < len(items) else None
            if nxt is not None:
                slot = max(nxt - s - gap, e - s)
            elif duration:  # last line must finish before the video ends
                slot = duration - s - 0.25
            else:
                slot = max(e - s, 0.1) + 1.5
            start = max(s, cursor)
        else:
            slot, start = None, cursor
        a, ratio = fit(voice, " ".join(sents[n]), slot, lines[n])
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
    if duration:
        end = int(duration * SR)
        out = np.concatenate([out, np.zeros(max(0, end - len(out)), np.float32)])
    return out[:end], report


def master(a, lufs, path, duration=None):
    """Loudness-normalize with a true-peak limit and write a 48 kHz WAV (exactly `duration` s if given)."""
    af = f"loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample=48000"
    if duration:
        af += f",apad,atrim=0:{duration:.3f}"
    with tempfile.TemporaryDirectory() as d:
        i = os.path.join(d, "i.wav")
        sf.write(i, a, SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-af", af, "-ar", "48000", path],
                       check=True)


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(r.stdout.strip())


def to_mp3(wav):
    mp3 = wav[:-4] + ".mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", "192k", mp3], check=True)
    return mp3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="?", help=".srt / .vtt / .txt (one sentence per line)")
    ap.add_argument("--text", help="speak this text instead of a subtitle file")
    ap.add_argument("--video", help="match this video's length; also writes a voice+BGM mix and a preview mp4")
    ap.add_argument("--bgm-volume", type=float, default=1.0,
                    help="level of the video's own audio in the voice+BGM mix (0 = skip the mix)")
    ap.add_argument("--profile", default=os.path.join(HERE, "profiles", "clone_narrator.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--name")
    ap.add_argument("--no-group", action="store_true", help="speak each subtitle cue separately")
    args = ap.parse_args()
    if not args.subs and not args.text:
        ap.error("give a subtitle file or --text")

    prof = load_profile(args.profile)
    voice = make_voice(prof)
    if args.text:
        items = [(None, None, args.text)]
        name = args.name or "preview"
    else:
        items = parse_subs(args.subs)
        if not args.no_group:
            items = group_sentences(items)
        name = args.name or os.path.splitext(os.path.basename(args.subs))[0]
    duration = media_duration(args.video) if args.video else None

    audio, report = render(items, voice, prof.get("min_gap", 0.08), duration)
    os.makedirs(args.out, exist_ok=True)
    wav = os.path.join(args.out, name + ("_voice.wav" if args.video else ".wav"))
    master(audio, prof.get("target_lufs", -16), wav, duration)
    print("->", wav, "\n->", to_mp3(wav))

    if args.video:
        track = wav
        if args.bgm_volume > 0 and has_audio(args.video):
            track = os.path.join(args.out, name + "_voice+bgm.wav")
            filt = (f"[0:a]aresample=48000,volume={args.bgm_volume}[bg];"
                    "[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0[v];"
                    "[bg][v]amix=inputs=2:duration=longest:normalize=0,"
                    f"alimiter=limit=0.89:level=false,apad,atrim=0:{duration:.3f}[a]")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.video, "-i", wav,
                            "-filter_complex", filt, "-map", "[a]", "-ar", "48000", track], check=True)
            print("->", track, "\n->", to_mp3(track))
        preview = os.path.join(args.out, name + "_preview.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.video, "-i", track,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", preview], check=True)
        print("->", preview)

    overs = [r for r in report if r[3] > 0.05]
    if overs:
        print(f"\n{len(overs)} line(s) could not fully fit their subtitle slot (see OVERFLOW above).")


if __name__ == "__main__":
    main()
