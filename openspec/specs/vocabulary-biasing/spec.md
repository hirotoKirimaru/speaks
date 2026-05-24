# vocabulary-biasing Specification

## Purpose
TBD - created by archiving change transcription-research-upgrade. Update Purpose after archive.
## Requirements
### Requirement: 語彙ファイルからの biasing
ユーザは `--vocab-file <path>` で 1 行 1 語の固有名詞 / 専門用語リストを与えることができる SHALL。空行と `#` で始まるコメント行は無視する SHALL。

#### Scenario: ファイル読み込み
- **WHEN** `--vocab-file vocab.txt` を指定し、ファイル内容が以下である:
  ```
  # 会議参加者
  山田太郎
  Anthropic
  ```
- **THEN** 語彙リスト `["山田太郎", "Anthropic"]` がパイプラインに渡される

#### Scenario: ファイルが存在しない
- **WHEN** 存在しないパスを `--vocab-file` に渡す
- **THEN** 明示的なエラーで exit code 1 で終了する

### Requirement: initial_prompt への自動注入
語彙ファイルが与えられた場合、システムは Whisper の `initial_prompt` に語彙を、ユーザ指定の `--initial-prompt` と結合して渡す SHALL。Whisper の prompt トークン上限 (224 token 相当 ≒ 日本語 100 文字程度) を超える場合は、語彙を切り詰める SHALL。

#### Scenario: ユーザプロンプトと結合
- **WHEN** `--initial-prompt "技術会議"` と `--vocab-file` (語彙: `["山田", "Anthropic"]`) を併用する
- **THEN** Whisper には `技術会議 用語: 山田、Anthropic` のような結合プロンプトが渡される

#### Scenario: 上限超過時の切り詰め
- **WHEN** 語彙数が prompt 上限を超える
- **THEN** 先頭から優先して切り詰めた上で警告ログを stderr に出す

### Requirement: trie ベース contextual biasing (オプション)
システムは `--biasing-mode prompt|trie|both` を提供する SHALL。既定は `prompt`。`trie` および `both` は decoder のロジット段で、語彙の prefix trie に沿った token 系列のスコアを加算してバイアスする SHALL。

#### Scenario: prompt のみ (既定)
- **WHEN** `--biasing-mode` を指定しない
- **THEN** `initial_prompt` のみが使われ、decoder ロジット改変は行われない

#### Scenario: trie 有効時の固有名詞ヒット
- **WHEN** `--biasing-mode trie` で語彙に `山田太郎` を含み、その発話が音声に存在する
- **THEN** prompt なしでも語彙からの token 系列が選好され、認識結果に `山田太郎` が現れる確率が上がる

#### Scenario: trie 未実装の段階的扱い
- **WHEN** 初回リリースで trie 実装が未完成の場合
- **THEN** `--biasing-mode trie` 指定時に「未実装、`prompt` を使用」と警告を出して `prompt` モードで動作する

