## 1. パッケージ化と既存挙動の温存

- [x] 1.1 `speaks/` パッケージを新設し、`main.py` のロジックを以下に移植 (関数シグネチャは一旦維持): `preprocess.py`, `diarize.py`, `transcribe.py`, `merge.py`, `summarize.py`, `ollama.py`, `prompts.py`, `io.py`, `cli.py`
- [x] 1.2 `pyproject.toml` の `[project.scripts]` を `speaks = "speaks.cli:app"` に変更
- [x] 1.3 `main.py` を `from speaks.cli import app; app()` の薄いシムにする
- [ ] 1.4 既存 CLI を 1 本実音声で smoke 実行し、出力ファイル名・内容が現行と同一であることを確認 (回帰なし) — リモート環境では実音声が無いため、`speaks --help` の出力が現行と同一であることのみ確認済み

## 2. forced alignment

- [ ] 2.1 `transformers` を `pyproject.toml` に追加し `uv sync`
- [ ] 2.2 `speaks/align.py` を実装: `Wav2Vec2ForCTC` + `Wav2Vec2Processor` で segment 単位の Viterbi アライメント、word 区間を `(word, start, end)` で返す
- [ ] 2.3 アライナのデフォルトモデルを `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` にする
- [ ] 2.4 `transcribe.py` で word_timestamps の置換ロジックを追加 (アライン失敗時は Whisper 内蔵にフォールバック)
- [ ] 2.5 `--align / --no-align`, `--align-model <hf-id>` を CLI に追加
- [ ] 2.6 既存テスト音声で word 境界が Whisper 内蔵より細かくなることを確認

## 3. 幻覚抑制

- [ ] 3.1 `speaks/postprocess.py` を新設し、`avg_logprob` / `no_speech_prob` を受け取って segment フィルタする関数を実装
- [ ] 3.2 同 module に「反復縮約」を実装 (segment 内 word 列で同一 token 列 3 回以上連続を 1 回に縮約)
- [ ] 3.3 `speaks/data/hallucination_phrases.ja.txt` を作成し、YouTube 由来の代表的な定型句 (`ご視聴ありがとうございました`、`チャンネル登録お願いします` 等) を 10 件以上入れる
- [ ] 3.4 同 module に「定型フレーズ完全一致 segment 除外」を実装
- [ ] 3.5 `transcribe.py` から postprocess を呼ぶ。`avg_logprob` / `no_speech_prob` を segment と一緒に持ち回るためのデータ構造を整える
- [ ] 3.6 `--hallucination-filter / --no-hallucination-filter`, `--hallucination-phrases <path>` を CLI に追加

## 4. Sortformer 対応

- [ ] 4.1 `speaks/diarize.py` で `Diarizer` Protocol を定義し、`PyannoteDiarizer` を既存ロジックから移植
- [ ] 4.2 `SortformerDiarizer` を実装 (NeMo の Sortformer 推論呼び出しを薄くラップ)
- [ ] 4.3 `pyproject.toml` に optional extra `[project.optional-dependencies] sortformer = ["nemo_toolkit[asr]>=2.0.0"]` を追加
- [ ] 4.4 `cli.py` で `--diarizer pyannote|sortformer` を追加。`sortformer` 選択時に NeMo 未インストールなら親切なエラー
- [ ] 4.5 既定で pyannote が選択されることを確認

## 5. vocab biasing (Phase 1)

- [ ] 5.1 `speaks/biasing.py` を新設し、`load_vocab(path) -> list[str]` (`#` コメント / 空行スキップ) を実装
- [ ] 5.2 `build_initial_prompt(user_prompt, vocab, max_chars=100) -> str` を実装し、上限超過時に警告
- [ ] 5.3 `cli.py` に `--vocab-file <path>` を追加し、`transcribe` / `correct` の両方に語彙を流す
- [ ] 5.4 `--biasing-mode prompt|trie|both` を CLI に追加。`trie` / `both` 指定時は warning を出して `prompt` で動作 (Phase 2 へ向けたインターフェース予約)

## 6. LLM 補正

- [ ] 6.1 `speaks/ollama.py` で Ollama HTTP 呼び出しを共通化 (`generate(model, prompt, timeout) -> str`)
- [ ] 6.2 `speaks/prompts.py` に `CORRECTION_PROMPT` を新設 (構造保持・行数保持・創作禁止・語彙厳守)
- [ ] 6.3 `speaks/correct.py` を新設し、transcript を Ollama に渡して補正後テキストを返す関数を実装
- [ ] 6.4 補正後の行数が入力と一致しない場合は warning を stderr に出して補正破棄 (fail open)
- [ ] 6.5 `cli.py` に `--correct / --no-correct`, `--correction-model <ollama-model>` を追加
- [ ] 6.6 `<prefix>_transcript.raw.txt` と `<prefix>_transcript.txt` (補正後) を `--correct` 有効時の既定で両方保存
- [ ] 6.7 `--vocab-file` が指定されていれば、補正プロンプトに語彙を埋め込む

## 7. 多段要約

- [ ] 7.1 `prompts.py` に `TOPIC_PROMPT` (5 件以内のトピック箇条書き) と新 `MINUTES_PROMPT` (トピック注入版議事録) を追加
- [ ] 7.2 `summarize.py` を `run_summary(transcript, mode, ...)` に再構成: `mode=="single"` で旧挙動、`mode=="multi"` でトピック → 議事録の 2 段
- [ ] 7.3 トピック中間結果を `<prefix>_topics.md` に保存
- [ ] 7.4 `cli.py` に `--summary-pipeline single|multi` を追加 (既定 `multi`)
- [ ] 7.5 旧挙動 (`single`) が文字通り同一プロンプトで動くことを diff で確認

## 8. ドキュメントと検証

- [ ] 8.1 `README.md` を更新: 新フラグ一覧、参考論文 (proposal 内のリスト)、`vocab-file` のフォーマット、Sortformer の optional install 手順
- [ ] 8.2 `openspec validate transcription-research-upgrade --strict` が pass することを確認
- [ ] 8.3 サンプル WAV (5 分程度) で以下のマトリクスを smoke run:
  - (a) 全フラグ既定 (新パイプライン)
  - (b) `--summary-pipeline single --no-correct --no-align --no-hallucination-filter` (旧挙動エミュレート)
  - (c) `--vocab-file vocab.txt` を指定
- [ ] 8.4 `archive` 実行は別ターン (ユーザ確認後): `openspec archive transcription-research-upgrade --yes`
