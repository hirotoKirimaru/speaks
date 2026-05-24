## ADDED Requirements

### Requirement: LLM による ASR 誤り訂正
システムは `--correct` (既定: 有効) フラグで、文字起こし全文を Ollama LLM に渡し、表記揺れ / 助詞抜け / 固有名詞誤りを修正したテキストを取得する SHALL。タイムスタンプと話者ラベルの構造は保持する SHALL。

#### Scenario: 補正の既定動作
- **WHEN** `--correct` 既定で実行する
- **THEN** ASR 出力 → 補正 LLM → 補正済み transcript の順で処理され、`<prefix>_transcript.txt` には補正後テキストが保存される
- **AND** 補正前のテキストは `<prefix>_transcript.raw.txt` として併せて保存される

#### Scenario: 補正の無効化
- **WHEN** `--no-correct` を指定する
- **THEN** LLM 補正は行われず、`<prefix>_transcript.txt` には ASR 生出力が保存される
- **AND** `<prefix>_transcript.raw.txt` は生成されない

#### Scenario: 補正モデルの差し替え
- **WHEN** `--correction-model qwen2.5:7b` を指定する
- **THEN** 補正フェーズではそのモデルが使用される (要約モデルは別途指定)

### Requirement: 構造保持プロンプト
補正プロンプトは LLM に対し、次を厳守させる SHALL:
- 各行の `[start - end] SPEAKER: text` 形式を保持する
- 行を増減させない
- 内容の創作 (発言追加) を禁止する
- 表記揺れ / 助詞 / 同音異義の選択に限定して修正する

#### Scenario: 行数保持
- **WHEN** 100 行の transcript を補正する
- **THEN** 出力も 100 行であり、タイムスタンプと話者ラベルは元と一致する

### Requirement: 語彙ファイルの併用
`--vocab-file` が指定されている場合、補正プロンプトに語彙リストを「正しい表記の固有名詞」として埋め込み、LLM がそれらに揃えるよう促す SHALL。

#### Scenario: 語彙併用
- **WHEN** 語彙に `Anthropic` を含み、ASR 出力に `アンソロピック` が含まれる
- **THEN** 補正後の transcript では `Anthropic` (または辞書通りの表記) に揃えられる
