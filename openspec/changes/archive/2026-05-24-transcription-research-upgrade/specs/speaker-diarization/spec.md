## ADDED Requirements

### Requirement: 話者分離バックエンドの選択
システムは `--diarizer pyannote|sortformer` で話者分離バックエンドを選択できる SHALL。既定は `pyannote` とする SHALL。

#### Scenario: 既定の pyannote 利用
- **WHEN** `--diarizer` を指定せず話者分離を有効で実行する
- **THEN** `pyannote/speaker-diarization-3.1` パイプラインがロードされ実行される

#### Scenario: Sortformer 選択
- **WHEN** `--diarizer sortformer` を指定する
- **THEN** NVIDIA Sortformer (オフラインモード) で話者分離が実行される
- **AND** `nemo_toolkit` が未インストールの場合は明示的なエラーメッセージで対処方法を案内する

### Requirement: HuggingFace トークンの自動解決
システムは pyannote 使用時、HF トークンを以下の優先順位で解決する SHALL:
1. CLI オプション `--hf-token`
2. 環境変数 `HF_TOKEN`
3. `~/.cache/huggingface/token` ファイル

#### Scenario: 環境変数からの解決
- **WHEN** `HF_TOKEN=hf_xxx speaks <wav>` を実行する
- **AND** `--hf-token` を指定しない
- **THEN** 環境変数の値が pyannote へ渡される

#### Scenario: トークン未設定でのエラー
- **WHEN** いずれの場所にもトークンが存在せず、`--no-diarize` も指定しない
- **THEN** トークン設定方法と `--no-diarize` フラグを案内するエラーで終了 (exit code 1)

### Requirement: 話者分離の無効化
ユーザは `--no-diarize` で話者分離をスキップできる SHALL。スキップ時はトークン要求やパイプラインロードを行わない SHALL。

#### Scenario: 無効化
- **WHEN** `--no-diarize` を指定する
- **THEN** diarizer のモデルロードは発生せず、文字起こしには話者ラベルが付与されない

### Requirement: word-level 話者割当
システムは話者分離が有効な場合、各 word の中点タイムスタンプを話者ターンに照合し、segment 内で連続する同一話者を 1 ライン (`[start - end] SPEAKER_XX: text`) にマージする SHALL。

#### Scenario: segment 境界をまたぐ話者切替
- **WHEN** ある Whisper segment 内で word 単位に話者が `SPEAKER_00` から `SPEAKER_01` へ切り替わる
- **THEN** 出力ではその segment が話者ごとに 2 行に分割される

#### Scenario: 話者ターン外の word
- **WHEN** ある word の中点がいずれの話者ターンにも入らない
- **THEN** 最も近い話者ターンの話者ラベルが採用される
