## ADDED Requirements

### Requirement: faster-whisper による日本語 ASR
システムは faster-whisper の `WhisperModel` を用いて、日本語固定 (`language="ja"`)、VAD フィルタ有効、word timestamps 有効で音声を文字起こしする SHALL。デフォルトモデルは `large-v3` とし、`--whisper-model` で上書きできる SHALL。

#### Scenario: 既定モデルでの文字起こし
- **WHEN** ユーザが追加オプションなしで `speaks <wav>` を実行する
- **THEN** `large-v3` モデルで日本語 ASR が実行され、各 segment に `start`, `end`, `text`, `words[]` が含まれる結果が得られる

#### Scenario: VAD パラメータ
- **WHEN** ASR を実行する
- **THEN** Silero VAD が有効化され、`min_silence_duration_ms=500` で短時間の無音が結合される

#### Scenario: 幻覚閾値
- **WHEN** ASR を実行する
- **THEN** `no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `condition_on_previous_text=False` が適用される

### Requirement: wav2vec2 による forced alignment
システムは Whisper の word timestamps を、wav2vec2 ベースの forced alignment で得た word 境界に置換できる SHALL。アライナのデフォルトモデルは日本語向け `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` とする SHALL。

#### Scenario: アライメント有効 (既定)
- **WHEN** `--align` 既定の状態で文字起こしを実行する
- **THEN** 各 word の `start` / `end` が wav2vec2 のアライメント結果で置換され、Whisper 内蔵タイムスタンプより細粒度になる

#### Scenario: アライメント無効化
- **WHEN** `--no-align` を指定する
- **THEN** wav2vec2 ロードは行われず、Whisper の word timestamps がそのまま使われる

#### Scenario: アライメント失敗時のフォールバック
- **WHEN** wav2vec2 アライメントが特定 segment で失敗 (例: 文字が辞書外) する
- **THEN** その segment のみ Whisper 内蔵タイムスタンプを保持し、パイプライン全体は失敗しない

### Requirement: 幻覚パターン後処理
システムは ASR 出力の幻覚を後処理で抑制する SHALL。具体的には:
- 同一 segment 内で同じ token 列が 3 回以上連続反復する箇所を 1 回に縮約する SHALL
- segment の `avg_logprob` が `-1.5` 未満かつ `no_speech_prob` が `0.5` 以上の segment を除外する SHALL
- 設定可能な定型フレーズ辞書 (例: `ご視聴ありがとうございました`) に完全一致する segment を除外する SHALL

#### Scenario: 反復幻覚の除去
- **WHEN** ASR 出力に「はい はい はい はい はい」が単一 segment で含まれる
- **THEN** 後処理後は「はい」1 回 (または連続パターン1組) に縮約される

#### Scenario: 低信頼セグメント除去
- **WHEN** ある segment の `avg_logprob=-1.8`, `no_speech_prob=0.7`
- **THEN** その segment は出力から除外される

#### Scenario: 後処理の無効化
- **WHEN** `--no-hallucination-filter` を指定する
- **THEN** 反復縮約 / 低信頼除去 / 定型フレーズ除去 のいずれも適用されない

### Requirement: 文字起こし出力フォーマット
システムは `[start.s - end.s] {speaker}: {text}` (話者分離あり) または `[start.s - end.s] {text}` (話者分離なし) の行ベースフォーマットで文字起こしを `<prefix>_transcript.txt` に保存する SHALL。

#### Scenario: 話者分離なし
- **WHEN** `--no-diarize` で実行する
- **THEN** 各行が `[1.2s - 3.4s] こんにちは` 形式で出力される
