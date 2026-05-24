# speaks

ローカル音声文字起こし + 議事録要約ツール (日本語向け)。

`faster-whisper` で文字起こし、`pyannote` / `Sortformer` で話者分離、`wav2vec2`
で word 単位の forced alignment、Ollama で ASR 補正と多段要約を行う。
すべてローカルで完結 (HuggingFace モデルのダウンロードと Ollama 起動を除く)。

## インストール

```bash
uv sync
# Sortformer (任意、NeMo が必要)
# uv pip install 'speaks[sortformer]'
```

事前に Ollama を起動し、要約モデル (既定: `llama3.1:8b`) を pull しておく:

```bash
ollama pull llama3.1:8b
```

話者分離に pyannote を使う場合は HuggingFace トークンが必要。
`huggingface-cli login` または `HF_TOKEN` 環境変数で設定する。

## 使い方

```bash
# 既定 (forced alignment + 幻覚抑制 + LLM 補正 + 多段要約)
speaks input.wav

# 文字起こしのみ
speaks input.wav --transcript-only

# 旧パイプラインで実行 (既存ユーザ向け互換モード)
speaks input.wav \
  --no-align --no-hallucination-filter --no-correct \
  --summary-pipeline single
```

## 主なオプション

| フラグ | 既定 | 説明 |
|---|---|---|
| `--whisper-model` | `large-v3` | faster-whisper モデル |
| `--ollama-model` | `llama3.1:8b` | 要約に使う Ollama モデル |
| `--correction-model` | (= ollama-model) | ASR 補正に使う Ollama モデル |
| `--align / --no-align` | `--align` | wav2vec2 forced alignment |
| `--align-model` | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` | アライナの HF モデル |
| `--diarizer` | `pyannote` | `pyannote` または `sortformer` |
| `--no-diarize` | off | 話者分離を無効化 |
| `--vocab-file` | なし | 固有名詞リスト (1 行 1 語、`#` でコメント) |
| `--biasing-mode` | `prompt` | `prompt`/`trie`/`both` (Phase 1 は prompt のみ実装、trie/both は warning で fallback) |
| `--hallucination-filter / --no-hallucination-filter` | on | 反復縮約 / 低信頼除去 / 定型句除外 |
| `--hallucination-phrases` | 同梱の日本語辞書 | 定型句リスト |
| `--correct / --no-correct` | on | LLM による ASR 補正 |
| `--summary-pipeline` | `multi` | `single` (単発) または `multi` (トピック → 議事録) |
| `--transcript-only` | off | 要約をスキップ |
| `--skip-preprocess` | off | 16kHz 変換をスキップ |

## 出力ファイル

`output/` (`--output-dir` で変更可) に以下を生成:

- `<timestamp>_<stem>_16k.wav`: 16kHz mono 正規化済み入力 (`--skip-preprocess` で抑止可)
- `<timestamp>_transcript.raw.txt`: ASR 生出力 (`--correct` 有効時のみ)
- `<timestamp>_transcript.txt`: 最終 transcript (補正後 / 補正無効時は ASR 出力)
- `<timestamp>_topics.md`: トピック抽出結果 (`--summary-pipeline multi` のみ)
- `<timestamp>_minutes.md`: 議事録 Markdown

## vocab ファイルの書式

```
# 固有名詞 (1 行 1 語、空行と # コメント可)
山田太郎
Anthropic
プロジェクトX
```

vocab は Whisper の `initial_prompt` に注入される (prompt の token 上限を超える分は警告付きで切り詰め)、
かつ LLM 補正のプロンプトに「正しい表記の固有名詞」として埋め込まれる。

## 参考論文

このプロジェクトの設計は以下の論文の知見に基づく:

- **WhisperX** (Bain+ 2023, [arXiv:2303.00747](https://arxiv.org/abs/2303.00747)) — forced alignment + diarization 統合
- **Powerset diarization** (Plaquet & Bredin 2023) — pyannote 3.1 の基礎
- **Streaming Sortformer** (NVIDIA 2025, [arXiv:2507.18446](https://arxiv.org/abs/2507.18446)) — pyannote 比で DER 約半減
- **Whisper hallucination on non-speech** (Barański+ 2025, [arXiv:2501.11378](https://arxiv.org/abs/2501.11378)) — 後処理での幻覚抑制
- **Multitask Whisper biasing** ([arXiv:2309.09552](https://arxiv.org/abs/2309.09552)) / **Zero-shot trie biasing** ([arXiv:2508.17796](https://arxiv.org/abs/2508.17796)) — contextual biasing
- **GenSEC** (Chen+ 2024, [arXiv:2409.09785](https://arxiv.org/abs/2409.09785)) — LLM による ASR 補正
- **Action-Item-Driven Summarization** ([arXiv:2312.17581](https://arxiv.org/abs/2312.17581)) — 多段要約

設計詳細は `openspec/changes/transcription-research-upgrade/` 参照。

## 開発

OpenSpec を仕様駆動開発に使っている:

```bash
npm i -g @fission-ai/openspec
openspec list
openspec show transcription-research-upgrade
openspec validate transcription-research-upgrade --strict
```
