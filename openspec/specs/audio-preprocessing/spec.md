# audio-preprocessing Specification

## Purpose
TBD - created by archiving change transcription-research-upgrade. Update Purpose after archive.
## Requirements
### Requirement: 16kHz モノラル正規化
システムは入力音声を 16kHz サンプリングレート / モノラル / ピーク 0.95 への振幅正規化 PCM に変換した WAV ファイルを生成し、後続パイプラインに渡す SHALL。

#### Scenario: 44.1kHz ステレオ WAV を入力
- **WHEN** 44100 Hz / 2ch / int16 の WAV ファイルを `speaks` に渡す
- **THEN** 16000 Hz / 1ch / int16 の WAV ファイルが `<output_dir>/<stem>_16k.wav` に生成される
- **AND** 波形の絶対値の最大値が 0.95 ± 0.01 (正規化後の振幅) になる

#### Scenario: 既に 16kHz モノラルの入力
- **WHEN** 16000 Hz / 1ch の WAV を入力する
- **THEN** リサンプルは行われず、振幅正規化のみが適用された WAV が生成される

### Requirement: 前処理スキップ
ユーザは `--skip-preprocess` フラグを指定して前処理を無効化できる SHALL。その場合、システムは入力 WAV をそのまま後続パイプラインに渡す SHALL。

#### Scenario: スキップ指定
- **WHEN** `--skip-preprocess` を付けて任意の WAV を入力する
- **THEN** リサンプル / 正規化は行われず、元のファイルパスが後続に渡る

### Requirement: 入力 dtype の安全な変換
システムは int16 / int32 / uint8 / float32 の WAV を float32 [-1.0, 1.0] に正規化して扱う SHALL。

#### Scenario: int32 WAV の入力
- **WHEN** dtype が int32 の WAV を入力する
- **THEN** 値域 [-2147483648, 2147483647] が [-1.0, 1.0] にスケーリングされ、後続処理でクリッピングや無音化が起きない

