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

## 목소리 후보 (profiles/)

엔진: **Kokoro-82M**(오픈소스 TTS) 기본 목소리 → 음높이/포먼트 이동 + 억양 확대(Praat) → 레퍼런스에 맞춘 매칭 EQ.

| 프로필 | 기본 목소리 | 화자 유사도* | EQ 후 스펙트럼 차이 | 특징 |
|---|---|---|---|---|
| `A_river` (기본값) | af_river | **0.79** | 7.4 dB** | 수치상 가장 비슷 |
| `B_alloy` | af_alloy | 0.76 | 2.4 dB | 균형형 |
| `C_sarah` | af_sarah | 0.69 | **1.6 dB** | 억양 폭이 가장 비슷하고 가장 자연스러움 |

\* Resemblyzer 화자 임베딩 코사인 유사도. 참고로 레퍼런스의 앞/뒤 절반끼리는 0.98이고, Kokoro 기본 목소리를 그대로 쓰면 0.48–0.63입니다.
\*\* A는 포먼트 이동 때문에 9.5 kHz 이상 대역이 비어 있어 이 수치가 커 보입니다. 들리는 음역에서는 차이가 작습니다.

`samples/`에 세 목소리가 같은 문장을 읽은 샘플이 있습니다.

## 사용법

```bash
./setup.sh                                  # 최초 1회: ffmpeg, espeak-ng, 파이썬 패키지, 모델 설치
python3 make_voice.py subs.srt              # -> out/subs.wav, out/subs.mp3
python3 make_voice.py subs.srt --video short.mp4                      # 원본 오디오를 빼고 내레이션만 입힘
python3 make_voice.py subs.srt --video short.mp4 --keep-original 0.2  # 원본 오디오를 20% 볼륨으로 깔기
python3 make_voice.py subs.srt --profile profiles/C_sarah.json        # 다른 목소리 사용
python3 make_voice.py --text "Wait, did he just say yes?"             # 빠른 미리듣기
```

- 잘게 쪼개진 자막은 문장 부호(. ! ?) 기준으로 합쳐서 읽기 때문에 억양이 자연스럽습니다. 자막 한 줄씩 따로 읽히려면 `--no-group`을 붙이세요.
- 문장은 해당 자막 시작 시각에 배치됩니다. 다음 자막 전까지 다 못 읽으면 먼저 합성 속도를 높이고, 그래도 넘치면 최대 약 1.5배까지 빠르게 만든 뒤 `OVERFLOW`로 알려줍니다.
- 출력은 −16 LUFS, 트루피크 −1.5 dBTP로 맞춰집니다(`target_lufs`로 조정).
- 새 레퍼런스에 맞춰 EQ를 다시 계산하려면
  `python3 calibrate.py profiles/A_river.json ref.m4a --skip 7.5-9.8 67.5-81` 를 실행하세요.
  `--skip`은 내레이터가 아닌 구간을 빼는 옵션입니다.

## 조절 가능한 값 (프로필 JSON)

`voice_mix`(여러 Kokoro 목소리를 섞을 때 비율), `speed`, `pitch_median_hz`, `formant_shift`(<1이면 더 굵은 음색),
`expressiveness`(억양 확대 배율), `eq_gains_db`(calibrate.py가 채움), `target_lufs`.
