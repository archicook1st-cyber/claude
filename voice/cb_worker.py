#!/usr/bin/env python3
"""Chatterbox voice-cloning worker. Runs inside cb_venv (torch 2.6), called by make_voice.py.

  cb_venv/bin/python cb_worker.py job.json

job.json: {"ref": "refs/narrator.wav", "exaggeration": 0.7, "cfg_weight": 0.4, "seed": 7,
           "max_tries": 3, "out_dir": "...", "items": [{"id": "0", "text": "..."}]}
Each line is checked with Whisper and regenerated with a new seed when words go missing.
Writes <out_dir>/<id>.wav and <out_dir>/results.json.
"""
import json, os, re, sys, time, difflib, warnings
warnings.filterwarnings("ignore")
import torch, torchaudio

torch.set_num_threads(os.cpu_count() or 4)


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


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
            torch.manual_seed(job.get("seed", 7) + attempt * 1000 + int(it["id"]))
            t = time.time()
            wav = model.generate(it["text"], audio_prompt_path=job["ref"],
                                 exaggeration=job["exaggeration"], cfg_weight=job["cfg_weight"])
            path = os.path.join(job["out_dir"], f"{it['id']}_try{attempt}.wav")
            torchaudio.save(path, wav, model.sr)
            heard = " ".join(s.text.strip() for s in asr.transcribe(path, language="en")[0])
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
