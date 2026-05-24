## Context

`speaks` は単一の `main.py` (約 280 行) に前処理 / diarization / ASR / 要約をすべて詰め込んだ CLI。動作はするが、論文ベースの改善 (WhisperX 流 forced alignment、Sortformer、幻覚抑制、contextual biasing、LLM 補正、多段要約) を入れるにはモノリシックすぎる。

依存しているライブラリは Python 3.14 / faster-whisper 1.2 / pyannote-audio 4.0 / torch 2.11。Ollama はローカルで動いている前提。話者分離は HuggingFace token が必要。

参考論文 (proposal 参照):
- WhisperX (Bain+ 2023, arXiv:2303.00747)
- Powerset diarization (Plaquet & Bredin 2023)
- Streaming Sortformer (NVIDIA 2025, arXiv:2507.18446)
- Whisper hallucination investigation (Barański+ 2025, arXiv:2501.11378)
- Multitask Whisper biasing (2023, arXiv:2309.09552), Zero-shot trie biasing (2025, arXiv:2508.17796)
- GenSEC (Chen+ 2024, arXiv:2409.09785), Action-Item-Driven Summarization (2023, arXiv:2312.17581)

## Goals / Non-Goals

**Goals:**
- 論文ベースの 6 つの改善を 1 つの change で導入する
- 既定動作は現行と互換 (出力ファイル名・フォーマット維持)。新機能はオプトインまたは無害な範囲のみデフォルト有効
- `main.py` を機能単位でモジュール分割し、テスト可能にする
- 各機能を CLI フラグで個別オン/オフできるようにする

**Non-Goals:**
- Whisper モデル本体の fine-tune (推論時改善のみ)
- ストリーミング (リアルタイム) 推論 — オフラインのファイル入力のみ
- 評価ベンチマーク自動化 (本 change のスコープ外、別 change で扱う)
- Sortformer の本格的な品質チューニング — まず動くパイプラインを通すまで

## Decisions

### D1. モジュール分割

**決定:** `main.py` を以下に分割し、`speaks/` パッケージ化する。

```
speaks/
  __init__.py
  cli.py              # typer エントリポイント
  preprocess.py       # 16kHz / mono / 正規化
  diarize.py          # pyannote / sortformer の抽象化
  transcribe.py       # faster-whisper ラッパ
  align.py            # wav2vec2 forced alignment
  biasing.py          # vocab ファイル読み込み / prompt 構築 / (将来 trie)
  postprocess.py      # 幻覚抑制 (反復除去 / 低信頼除去 / 定型句除外)
  merge.py            # word ↔ speaker マージ (旧 _merge_words_by_speaker)
  correct.py          # LLM による transcript 補正
  summarize.py        # 単段 / 多段要約
  ollama.py           # Ollama HTTP クライアントを共通化
  prompts.py          # 補正 / トピック / 議事録の各プロンプトテンプレート
  io.py               # 入出力ファイルパス / フォーマット
```

`pyproject.toml` の `[project.scripts]` は `speaks = "speaks.cli:app"` に変更。`main.py` は薄いシムとして残し、後方互換のためだけに `from speaks.cli import app` する。

**代替案:** モジュール分割せず `main.py` に詰める。→ 却下。テスト可能性と将来の差分レビューが厳しい。

### D2. forced alignment の実装

**決定:** `whisperx` パッケージそのものは導入せず、`transformers` の wav2vec2 を直接使う薄い実装にする (`speaks/align.py`)。理由:
- `whisperx` は Python バージョンや torch との依存制約が厳しめで、3.14 / torch 2.11 で動作する保証が薄い
- 必要なのは「word 区間の Viterbi アライメント」だけで、wav2vec2 + CTC で 100 行程度
- アライナのモデルは `--align-model` で差し替え可能、既定は `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`

アラインは Whisper の segment 単位で実施し、segment ごとに失敗してもパイプライン全体は継続。失敗時はその segment の word timestamps を Whisper 内蔵のまま残す。

**代替案:** `whisperx` をそのまま使う。→ 依存解決リスクが高い。

### D3. diarization の抽象化

**決定:** `Diarizer` プロトコルを切り、`PyannoteDiarizer` / `SortformerDiarizer` の 2 実装を持つ。Sortformer は `nemo_toolkit` を **optional extra** (`speaks[sortformer]`) として扱い、未インストール時は import エラーを CLI 層で握って親切なメッセージを出す。

```python
class Diarizer(Protocol):
    def __call__(self, wav_path: str) -> list[tuple[float, float, str]]: ...
```

返り値の `(start, end, speaker_label)` 形式は現行と同じ。

**代替案:** Sortformer を必須依存にする。→ NeMo は重い。pyannote ユーザに不要な負荷を強いるので却下。

### D4. 幻覚後処理の戦略

**決定:** 後処理は次の 3 段。`--no-hallucination-filter` で全段オフ。

1. **反復縮約**: segment 内 word 列を見て、同一 token 列が 3 回以上連続したら 1 回に縮約 (`はい はい はい はい はい` → `はい`)
2. **低信頼除去**: `avg_logprob < -1.5` かつ `no_speech_prob > 0.5` の segment 全体を破棄
3. **定型フレーズ除去**: 同梱の `hallucination_phrases.txt` (例: `ご視聴ありがとうございました`、`チャンネル登録` 等の YouTube 由来定型句) に完全一致する segment を除外

定型フレーズ辞書は `speaks/data/hallucination_phrases.ja.txt` に同梱し、`--hallucination-phrases <path>` で差し替え可能。

**代替案:** `Listen Like a Teacher` 的に encoder 側で抑制。→ Whisper 本体の改造が必要。スコープ外。

### D5. contextual biasing は段階的に

**決定:**
- Phase 1 (この change): `--vocab-file` 読み込み + `initial_prompt` への注入のみ実装。`--biasing-mode prompt` を既定。`trie` / `both` は **未実装** として CLI には用意し、警告を出して prompt にフォールバック
- Phase 2 (別 change): trie 実装

これは「論文の知見を仕様に反映」と「現実的なスコープ」のバランス。trie biasing は faster-whisper の decoder にフックを刺す必要があり、本 change で全部やると肥大化する。

**代替案:** Phase 1 で trie まで全部やる。→ 規模が膨らみすぎ、リリースが遠のく。

### D6. LLM 補正プロンプトの安全性

**決定:** 補正プロンプトには次のガードを明示的に書く:
- 「行を増減させてはいけない」
- 「`[start - end] SPEAKER:` の構造は完全保持」
- 「内容を創作してはならない。表記揺れ / 助詞 / 同音異義の選択に限定」
- 「与えられた語彙リストの表記に揃えること」

補正後、行数を入力と比較し、増減があれば警告ログを出し、補正なしで進める (fail open)。raw も `<prefix>_transcript.raw.txt` に保存して後追い可能にする。

**代替案:** 行ごとに LLM を呼ぶ。→ レイテンシ × 行数で爆発。

### D7. 多段要約のステージ分割

**決定:** 既定 `multi` パイプラインは 2 段:

1. **トピック抽出** (`prompts.TOPIC_PROMPT`): transcript → 5 件以内のトピック箇条書き。`<prefix>_topics.md` に保存
2. **議事録合成** (`prompts.MINUTES_PROMPT`): transcript + 抽出済みトピックを渡して議事録 Markdown を生成

これは AutoMin / Action-Item-Driven Summarization の知見を踏まえつつ、ローカル LLM の現実的な能力に合わせた最小段数。論文では 4-5 段に分けるものもあるが、`llama3.1:8b` の品質ではむしろノイズが増えるので 2 段に留める。

**代替案:** 3 段以上 (トピック → アクションアイテム抽出 → 議事録合成)。→ 効果検証なしに段数を増やすと品質が逆に落ちる可能性があるため、まず 2 段で運用。

### D8. CLI 互換性

**決定:** 既存フラグ (`--whisper-model`, `--ollama-model`, `--transcript-only`, `--no-diarize`, `--hf-token`, `--output-dir`, `--initial-prompt`, `--skip-preprocess`) はすべて維持し、追加フラグだけを足す。

追加フラグ:
- `--align / --no-align` (既定: `--align`)
- `--align-model <hf-id>`
- `--diarizer pyannote|sortformer` (既定: `pyannote`)
- `--vocab-file <path>`
- `--biasing-mode prompt|trie|both` (既定: `prompt`、`trie`/`both` は警告)
- `--hallucination-filter / --no-hallucination-filter` (既定: 有効)
- `--hallucination-phrases <path>`
- `--correct / --no-correct` (既定: 有効)
- `--correction-model <ollama-model>` (既定: `--ollama-model` と同じ)
- `--summary-pipeline single|multi` (既定: `multi`)

## Risks / Trade-offs

- **[wav2vec2 アライナの言語固有性]** 既定モデルは日本語専用。多言語化する場合は別モデルが必要 → 言語ごとにマップを持たせる余地を残すが、本 change では日本語固定。
- **[Sortformer の品質ばらつき]** Streaming 版のオフライン利用は pyannote と同等以上という報告だが、日本語ベンチが少ない → Sortformer は opt-in。既定は pyannote 維持。
- **[多段要約のレイテンシ増]** ステージが 2 倍 → 大体 1.5〜2 倍の時間。`--summary-pipeline single` で旧挙動に戻せる。
- **[LLM 補正の改悪リスク]** 補正で意味が変わるケースがある → raw を併存保存 + 行数不一致時は補正破棄。
- **[依存追加]** `transformers` が増える (wav2vec2 用)。faster-whisper が既に近傍の重量級依存を引いているため、絶対量はそこまで悪化しない見込み。

## Migration Plan

1. ブランチ `claude/transcription-paper-research-mZcr7` 上で実装
2. `main.py` → `speaks/` への分割を最初のコミット (機能変更なし)
3. 機能を 1 つずつ実装するコミットを連ねる (preprocess → diarize → transcribe → align → biasing → postprocess → correct → summarize → cli)
4. 既存利用者向けに README に migration note と新フラグ一覧を追記
5. ロールバック: 各機能はフラグで無効化可能なので、問題時はフラグで回避できる。コード全体の revert は単一 PR 単位

## Open Questions

- (Q1) `transformers` 追加が `uv.lock` 解決に影響しないか — 実装時に確認
- (Q2) Sortformer のオフライン推論で何 spk 上限まで現実的か — 試した上でドキュメントに記載
- (Q3) 補正で行数が一致しないケースがどの程度発生するか — 実音声でログを取って判断、必要なら chunk 分割補正を後続 change で
