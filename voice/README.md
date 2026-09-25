# BUDA-CAT 스타일 내레이션 생성기

자막(SRT/VTT/TXT)을 받아 레퍼런스 음성(“Hit by a Motorbike in Vietnam | BUDA-CAT”)과
비슷한 목소리·말투로 내레이션을 만들고, 원하면 쇼츠 영상에 입혀줍니다.

## 레퍼런스 음성 분석 결과

| 항목 | 레퍼런스 | 비고 |
|---|---|---|
| 평균 음높이(F0 중앙값) | ≈ 300–320 Hz | 성인 여성 평균(≈200 Hz)보다 한참 높음. 귀엽고 만화 캐릭터 같은 톤 |
| 억양 폭 | 5–6 반음(표준편차) | 일반 TTS(2–4 반음)보다 훨씬 표현력이 큼. 감탄·의문문에서 크게 오르내림 |
| 말하기 속도 | 발화 구간 기준 초당 약 4.6음절 | 보통~약간 빠른 이야기체 |
| 쉼 | 문장 사이 약 0.24초, 최대 1초 | 짧고 경쾌하게 이어짐 |
| 음색 | 3.5 kHz까지 배음이 강하고, 4 kHz 위는 약 20 dB 낮음 | 밝지만 치찰음은 부드러움 |
| 녹음 상태 | 배경음악 없음, 쉼 구간이 깨끗함, 약 −19 LUFS | 깨끗한 스튜디오/TTS 음질 |
| 스타일 | 1인칭 이야기체, 수사 의문문(“Wait, did he just say yes?”), 시청자에게 질문으로 마무리 | |

## 목소리 (profiles/)

### 기본값: `clone_narrator` — 원본 내레이터 음성 복제

- **엔진:** [Chatterbox](https://github.com/resemble-ai/chatterbox)(MIT 라이선스, 상업적 사용 가능)를 씁니다. 원본 영상 10.0~21.4초의 내레이터 음성(`refs/narrator_ref.wav`)을 참고해 목소리를 복제합니다.
- **설정:** 감정 표현(`exaggeration`)은 1.0, 가이던스(`cfg_weight`)는 0.3입니다. 문장마다 음높이를 295Hz 쪽으로 85%만큼 당겨 캐릭터 톤을 일정하게 유지합니다.
- **자동 검사:** 문장을 하나 만들 때마다 Whisper로 받아쓰기해 단어가 빠지거나 틀렸는지 확인합니다. 문제가 있으면 다른 시드로 최대 3번까지 다시 만듭니다.

| | 원본과 유사도* | 음높이 | 억양 폭 |
|---|---|---|---|
| 원본(복제에 쓰지 않은 구간) | 0.981 | 297 Hz | 4.5 반음 |
| **clone_narrator** | **0.963–0.966** | 294 Hz | 4.0–4.2 반음 |
| Kokoro A_river | 0.747 | 298 Hz | 3.4 반음 |

\* Resemblyzer 화자 임베딩의 코사인 유사도입니다. 원본 내레이터 구간 전체와 비교했습니다.

- **속도:** CPU만 쓰면 음성 1초를 만드는 데 약 10초 걸립니다(받아쓰기 검사 포함). 60초짜리 쇼츠는 10~15분쯤 걸립니다.
- **최초 실행:** 모델을 huggingface.co에서 내려받습니다. 그래서 클라우드 환경의 Network access 허용 목록에 `huggingface.co`, `*.hf.co`, `us.aws.cdn.hf.co` 등이 있어야 합니다.

### 오프라인 대안: Kokoro 프로필 `A_river` / `B_alloy` / `C_sarah`

- **방식:** Kokoro-82M(Apache-2.0)의 기본 목소리에 음높이·포먼트 이동과 매칭 EQ를 적용합니다. 인터넷 없이 빠르게 만들 수 있지만 원본과는 덜 비슷합니다(유사도 0.69–0.79).
- **EQ 재계산:** `calibrate.py`로 다시 맞출 수 있습니다.

`samples/`에 목소리별 샘플이 있습니다.

## 사용법

```bash
./setup.sh                                  # 최초 1회: ffmpeg, 복제 엔진(cb_venv), Kokoro 모델 설치
python3 make_voice.py subs.srt              # -> out/subs.wav, out/subs.mp3
python3 make_voice.py subs.srt --video short.mp4   # 영상 길이에 맞춘 목소리 + (목소리+배경음악) 믹스 + 미리보기 mp4
python3 make_voice.py subs.srt --video short.mp4 --bgm-volume 0.6   # 믹스에서 배경음악을 60%로
python3 make_voice.py subs.srt --profile profiles/C_sarah.json        # Kokoro 목소리 사용(오프라인, 빠름)
python3 make_voice.py --text "Wait, did he just say yes?"             # 빠른 미리듣기
```

- 잘게 쪼개진 자막은 문장 부호(. ! ?) 기준으로 합쳐 문장 단위로 만듭니다. 그래서 억양이 자연스럽습니다. 자막 한 줄씩 따로 읽히려면 `--no-group`을 붙이세요.
  "Huh?"처럼 아주 짧은 문장은 다음 문장에 붙여서 만듭니다.
- `$1.50` → "a dollar fifty", `3` → "three", `~` → `!`, `ALWAYS` → "always"처럼 숫자와 기호는 읽기 좋게 바꿔서 합성합니다.
- `--video`를 주면 영상과 길이가 정확히 같은 파일이 세 가지 나옵니다.
  - `*_voice.wav/mp3`: 목소리만
  - `*_voice+bgm.wav/mp3`: 영상의 원래 오디오(배경음악)와 목소리를 섞은 것
  - `*_preview.mp4`: 싱크 확인용 미리보기

  YouTube 다국어 오디오 트랙은 원래 오디오를 통째로 바꾸기 때문에, 배경음악이 있는 영상이면 `*_voice+bgm` 파일을 올리세요.
- 복제 음성은 문장별로 `.cache/`에 저장됩니다. 자막 일부만 고쳐서 다시 돌리면 바뀐 문장만 새로 만듭니다.
- 문장은 해당 자막 시작 시각에 배치됩니다. 다음 자막 전까지 다 못 읽으면 음높이는 그대로 두고 최대 약 1.5배까지 빠르게 만듭니다(Rubber Band). 그래도 넘치면 `OVERFLOW`로 알려줍니다.
- 출력은 −16 LUFS, 트루피크 −1.5 dBTP로 맞춰집니다(`target_lufs`로 조정).
- Kokoro 프로필의 EQ를 새 레퍼런스에 맞춰 다시 계산하려면
  `python3 calibrate.py profiles/A_river.json ref.m4a --skip 7.5-9.8 67.5-81` 를 실행하세요.
  `--skip`은 내레이터가 아닌 구간을 빼는 옵션입니다.
- 다른 캐릭터 목소리가 필요하면 `clone_narrator.json`을 복사해 `ref_audio`만 그 인물의 깨끗한 음성 10초로 바꾸면 됩니다.

## 조절 가능한 값 (프로필 JSON)

- **공통:** `pitch_median_hz`, `formant_shift`(1보다 작으면 더 굵은 음색), `expressiveness`(억양 확대 배율), `eq_gains_db`, `target_lufs`
- **clone:** `ref_audio`, `exaggeration`(클수록 감정이 과장됨), `cfg_weight`(작을수록 느긋하게 말함), `pitch_normalize`(0~1, 문장별 음높이를 목표값으로 당기는 강도), `seed`, `max_tries`, `max_wer`
- **Kokoro:** `voice_mix`(여러 목소리를 섞는 비율), `speed`
