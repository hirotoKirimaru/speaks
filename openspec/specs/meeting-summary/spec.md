# meeting-summary Specification

## Purpose
TBD - created by archiving change transcription-research-upgrade. Update Purpose after archive.
## Requirements
### Requirement: 要約パイプライン選択
システムは `--summary-pipeline single|multi` を提供する SHALL。既定は `multi`。

#### Scenario: 単段パイプライン (既存互換)
- **WHEN** `--summary-pipeline single` を指定する
- **THEN** 既存の単一プロンプト (議事録テンプレート一発出力) で要約され、出力は `<prefix>_minutes.md` のみ

#### Scenario: 多段パイプライン (既定)
- **WHEN** `--summary-pipeline` を指定しない (既定 `multi`)
- **THEN** 後述する (1) トピック抽出 → (2) 議事録合成 の 2 段で要約が行われ、`<prefix>_minutes.md` に最終出力が保存される

### Requirement: トピック抽出ステージ
多段パイプラインの第 1 段でシステムは、補正済み transcript からトピックの箇条書き (5 件以内) を抽出する SHALL。中間結果は `<prefix>_topics.md` として保存する SHALL。

#### Scenario: トピックの粒度
- **WHEN** 30 分相当の transcript を入力する
- **THEN** 1 件あたり 1 行 (1 文以内) のトピックが最大 5 件、`<prefix>_topics.md` に出力される

### Requirement: 議事録合成ステージ
多段パイプラインの第 2 段でシステムは、補正済み transcript と抽出されたトピックを合わせて LLM に渡し、以下を含む Markdown を生成する SHALL:
- `# 議事録`
- `## トピック` (第 1 段の結果)
- `## 決定事項` (なければ「特になし」)
- `## TODO` (チェックボックス形式、なければ「特になし」)
- `## 議論メモ`

#### Scenario: TODO の形式
- **WHEN** transcript 中に「来週までに Y をやる」という発言がある
- **THEN** `## TODO` セクションに `- [ ] Y を来週までに行う (発言者: ...)` のような行が現れる

#### Scenario: 話者ラベルの保持
- **WHEN** transcript に複数の `SPEAKER_XX` が登場する
- **THEN** 議事録の `## 議論メモ` で話者を推定 (発言内容から発話主を推定) しつつ要約し、不明な場合は「発言者不明」と記す

#### Scenario: 出力ファイル
- **WHEN** 多段要約が成功する
- **THEN** `<prefix>_minutes.md` に最終議事録が保存される
- **AND** `<prefix>_topics.md` (第 1 段の中間結果) も保持される

### Requirement: 要約スキップ
ユーザは `--transcript-only` で要約フェーズ全体をスキップできる SHALL。

#### Scenario: 文字起こしのみ
- **WHEN** `--transcript-only` を指定する
- **THEN** `<prefix>_transcript.txt` (および `--correct` 有効時は `<prefix>_transcript.raw.txt`) のみが生成され、`<prefix>_minutes.md` / `<prefix>_topics.md` は作られない

### Requirement: 要約モデルの差し替え
ユーザは `--ollama-model` で要約に使う Ollama モデルを上書きできる SHALL。既定は `llama3.1:8b` を維持する SHALL。

#### Scenario: 異なるモデルの指定
- **WHEN** `--ollama-model qwen2.5:14b` を指定する
- **THEN** 要約 (多段の場合は両ステージ) は qwen2.5:14b で実行される

