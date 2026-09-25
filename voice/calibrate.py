#!/usr/bin/env python3
"""Fit a profile's match-EQ to a reference recording and report how close it gets.

  python3 calibrate.py profiles/A_river.json reference.m4a [--skip START-END ...]

--skip removes spans (seconds) of the reference that are not the narrator.
Similarity uses Resemblyzer if it is installed (pip install torch; pip install --no-deps resemblyzer).
"""
import argparse, json, os, subprocess, tempfile
import numpy as np, soundfile as sf, parselmouth
from make_voice import SR, KokoroVoice, shift_voice, eq_curve, apply_eq, band_db

CALIBRATION_TEXT = (
    "I was just riding along, heading out from my home, when a motorbike came flying at me the wrong way! "
    "Boom. Huge bruise on my leg, and my back was so sore. Wait, did he just say yes? "
    "So I contacted support myself, and I explained everything, step by step, with photos. "
    "Their reply was basically: contact the police yourself. Seriously? "
    "The next day, my leg still hurt, and nobody called me back. "
    "Should I go to the police? What would you do? Tell me in the comments!"
)


def load_ref(path, skips):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.wav")
        af = []
        if skips:
            cond = "*".join(f"not(between(t,{a},{b}))" for a, b in skips)
            af = ["-af", f"aselect='{cond}',asetpts=N/SR/TB"]
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar", str(SR)]
                       + af + [out], check=True)
        y, _ = sf.read(out, dtype="float32")
    return y


def pitch_stats(a):
    p = parselmouth.Sound(np.asarray(a, dtype=np.float64), SR).to_pitch_ac(0.01, 75, 650)
    f = p.selected_array["frequency"]
    v = f[f > 0]
    return float(np.median(v)), float(np.std(12 * np.log2(v / np.median(v))))


def similarity(a, ref):
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return None
    enc = VoiceEncoder("cpu", verbose=False)
    e = lambda x: enc.embed_utterance(preprocess_wav(x, source_sr=SR))
    return float(np.dot(e(a), e(ref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("reference")
    ap.add_argument("--skip", nargs="*", default=[], help="spans like 7.5-9.8")
    ap.add_argument("--save-sample", help="write the calibrated sample here")
    args = ap.parse_args()

    skips = [tuple(map(float, s.split("-"))) for s in args.skip]
    ref = load_ref(args.reference, skips)
    prof = json.load(open(args.profile))
    prof.pop("eq_gains_db", None)
    if prof.get("engine") == "chatterbox":
        raise SystemExit("calibrate.py tunes Kokoro profiles; the clone profile takes its timbre from ref_audio")
    v = KokoroVoice(prof)
    raw = v.raw(CALIBRATION_TEXT, prof["speed"])
    shifted = shift_voice(raw, prof["formant_shift"], prof["pitch_median_hz"], prof["expressiveness"])
    gains = eq_curve(shifted, ref)
    out = apply_eq(shifted, gains)

    prof["eq_gains_db"] = [round(float(g), 2) for g in gains]
    with open(args.profile, "w") as f:
        json.dump(prof, f, indent=2)

    def band_rmse(x):
        d = band_db(x) - band_db(ref)
        return float(np.sqrt(np.mean((d - d.mean()) ** 2)))

    rm, rs = pitch_stats(ref)
    om, os_ = pitch_stats(out)
    print(f"reference : pitch median {rm:.0f} Hz, intonation spread {rs:.2f} st")
    print(f"voice     : pitch median {om:.0f} Hz, intonation spread {os_:.2f} st")
    print(f"spectrum mismatch: {band_rmse(shifted):.1f} dB before EQ -> {band_rmse(out):.1f} dB after")
    s = similarity(out, ref)
    if s is not None:
        print(f"speaker similarity (Resemblyzer cosine): {s:.3f}")
    if args.save_sample:
        sf.write(args.save_sample, out / max(1e-6, np.abs(out).max()) * 0.9, SR)
    print("saved EQ ->", args.profile)


if __name__ == "__main__":
    main()
