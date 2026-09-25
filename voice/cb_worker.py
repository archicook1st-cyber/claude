#!/usr/bin/env python3
"""Chatterbox voice-cloning worker. Runs inside cb_venv (torch 2.6), called by make_voice.py.

  cb_venv/bin/python cb_worker.py job.json

job.json: {"ref": "refs/narrator.wav", "exaggeration": 0.7, "cfg_weight": 0.4, "seed": 7,
           "max_tries": 3, "out_dir": "...", "items": [{"id": "0", "text": "..."}]}
Each line is checked with Whisper and regenerated with a new seed when words go missing.
Writes <out_dir>/<id>.wav and <out_dir>/results.json.
"""
import json, os, re, sys, time, difflib, warnings, zlib
warnings.filterwarnings("ignore")
import torch, torchaudio

torch.set_num_threads(os.cpu_count() or 4)


# Whisper may write "$1.50" where we said "a dollar fifty"; don't count number wording as errors
NUMBERISH = set(("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                 "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                 "hundred thousand dollar dollars cent cents point a").split())


def words(s):
    ws = re.findall(r"[a-z0-9']+", s.lower().replace("’", "'").replace("-", " "))
    return [w for w in ws if w not in NUMBERISH and not any(c.isdigit() for c in w)]


def trim_tail(path, text, asr):
    """Cut sounds the model tacked on after the final word (e.g. a stray "WHAA-").
    Returns what Whisper heard, up to the cut. Only trims when the final word is clearly found."""
    segs, _ = asr.transcribe(path, language="en", word_timestamps=True)
    ws = [w for s in segs for w in s.words]
    raw = lambda s: re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))
    last = (raw(text) or [""])[-1]
    idx = max((i for i, w in enumerate(ws) if last in raw(w.word)), default=None)
    if idx is None or idx == len(ws) - 1:
        return " ".join(w.word.strip() for w in ws)
    wav, sr = torchaudio.load(path)
    cut = min(wav.shape[-1], int((ws[idx].end + 0.12) * sr))
    fade = torch.linspace(1, 0, min(int(0.03 * sr), cut))
    wav = wav[..., :cut].clone()
    wav[..., cut - fade.numel():] *= fade
    torchaudio.save(path, wav, sr)
    extra = " ".join(w.word.strip() for w in ws[idx + 1:])
    print(f"    trimmed trailing {extra!r} after {ws[idx].end:.2f}s", flush=True)
    return " ".join(w.word.strip() for w in ws[:idx + 1])


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return 1 - matched / max(len(r), len(h))


def main():
    job = json.load(open(sys.argv[1]))
    from chatterbox.tts import ChatterboxTTS
    from faster_whisper import WhisperModel
    t = time.time()
    model = ChatterboxTTS.from_pretrained(device="cpu")
    asr = WhisperModel(job.get("asr_model", "large-v3-turbo"), device="cpu", compute_type="int8")
    print(f"models loaded in {time.time() - t:.0f}s", flush=True)
    os.makedirs(job["out_dir"], exist_ok=True)
    results = []
    for it in job["items"]:
        best = None
        for attempt in range(job.get("max_tries", 3)):
            torch.manual_seed(job.get("seed", 7) + attempt * 1000 + zlib.crc32(it["text"].encode()) % 100000)
            t = time.time()
            wav = model.generate(it["text"], audio_prompt_path=job["ref"],
                                 exaggeration=job["exaggeration"], cfg_weight=job["cfg_weight"])
            path = os.path.join(job["out_dir"], f"{it['id']}_try{attempt}.wav")
            torchaudio.save(path, wav, model.sr)
            heard = trim_tail(path, it["text"], asr)
            err = wer(it["text"], heard)
            print(f"[{it['id']}] try {attempt}: {wav.shape[-1] / model.sr:.1f}s in {time.time() - t:.0f}s, "
                  f"WER {err:.2f} | {heard}", flush=True)
            if best is None or err < best[0]:
                best = (err, path, heard)
            if err <= job.get("max_wer", 0.1):
                break
        final = os.path.join(job["out_dir"], f"{it['id']}.wav")
        os.replace(best[1], final)
        results.append({"id": it["id"], "text": it["text"], "heard": best[2], "wer": round(best[0], 3),
                        "path": final, "sr": model.sr})
        for f in os.listdir(job["out_dir"]):
            if f.startswith(f"{it['id']}_try"):
                os.remove(os.path.join(job["out_dir"], f))
    json.dump(results, open(os.path.join(job["out_dir"], "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
