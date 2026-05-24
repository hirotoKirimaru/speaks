## Why

`speaks` の現状 (`faster-whisper large-v3` + `pyannote 3.1` + `Ollama llama3.1`) は動くが、近年の論文で示された次の弱点が残っている。

- Whisper 内蔵の `word_timestamps` は forced alignment より精度が低く、pyannote の話者区間と word 単位でマージすると境界がずれる (WhisperX, Bain+ 2023)。
- VAD を通しても非発話区間で繰り返し句や定型フレーズの幻覚が残る (Barański+ 2025)。
- `--initial-prompt` はトークン長に制約され、固有名詞の保証が弱い。trie ベースの contextual biasing で改善余地がある (Zero-shot Context Biasing, 2025)。
- 単発プロンプトで「議事録 + TODO + 決定事項」を一気に出させる現設計は、長尺だと欠落が起きる。ASR 補正と要約を多段化する手法が標準になりつつある (GenSEC, AutoMin, 2024–2025)。

これらを論文の知見で潰し、ローカル運用のまま品質を底上げする。

## What Changes

- ASR 後段に **wav2vec2 ベースの forced alignment** を追加して word timestamps を置換 (任意フラグ、デフォルト有効)
- diarization に **NVIDIA Sortformer (オフライン)** を選択肢として追加 (`--diarizer pyannote|sortformer`、デフォルト pyannote 維持)
- `--vocab-file` を新設し、語彙リストから (a) `initial_prompt` への自動注入 と (b) decoder への trie ベース biasing を行う
- VAD 通過後の出力に対し **幻覚パターン後処理** (繰り返しトークン除去、低 `avg_logprob` セグメント破棄、定型フレーズ辞書マッチ) を追加
- 要約パイプラインを単発 → **多段** に分解: `(1) ASR 補正 → (2) トピック抽出 → (3) 議事録合成 (TODO/決定/議論)`
- すべて CLI フラグでオプトアウト可能。既存ユーザの挙動を壊さない (BREAKING ではない)

## Capabilities

### New Capabilities
- `audio-preprocessing`: 入力音声を 16kHz mono + ピーク正規化した WAV に変換するパイプライン
- `transcription-pipeline`: faster-whisper による日本語 ASR と forced alignment、幻覚後処理を含む文字起こし生成
- `speaker-diarization`: pyannote / Sortformer から選択可能な話者分離と word-level 話者マージ
- `vocabulary-biasing`: 固有名詞辞書による Whisper への文脈バイアス (initial_prompt + trie decoding)
- `transcript-postcorrection`: LLM (Ollama) を用いた文字起こしの誤り訂正後処理
- `meeting-summary`: 多段プロンプトによる議事録 (トピック / 決定事項 / TODO / 議論メモ) 生成

### Modified Capabilities
(なし — 既存 spec は無いので全て新規)

## Impact

**コード:**
- `main.py` をモジュール分割 (`speaks/preprocess.py`, `speaks/transcribe.py`, `speaks/align.py`, `speaks/diarize.py`, `speaks/biasing.py`, `speaks/postprocess.py`, `speaks/correct.py`, `speaks/summarize.py`, `speaks/cli.py`)
- CLI に `--align/--no-align`, `--diarizer`, `--vocab-file`, `--hallucination-filter/--no-hallucination-filter`, `--correct/--no-correct`, `--summary-pipeline single|multi` を追加

**依存:**
- 追加: `transformers` (wav2vec2 alignment)、Sortformer は `nemo_toolkit[asr]` を **オプション extra** にして必須化しない
- 既存 (`faster-whisper`, `pyannote-audio`, `torch`, `torchaudio`, `httpx`, `typer`) は据え置き

**外部:**
- HuggingFace モデル `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` を追加でダウンロード
- Ollama モデル要件は変わらず (補正・要約とも既存モデルで動作)

**互換性:**
- 既定動作は現行と同等になるよう、新機能はオプトインまたは効果が等価な範囲のみデフォルト有効
- 出力ファイル名 / フォーマットは維持
