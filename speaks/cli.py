"""speaks CLI エントリポイント (typer)。"""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from speaks.biasing import (
    build_initial_prompt,
    load_vocab,
    warn_unimplemented_biasing,
)
from speaks.correct import correct_transcript
from speaks.diarize import make_diarizer, resolve_hf_token
from speaks.io import OLLAMA_MODEL, WHISPER_MODEL
from speaks.postprocess import default_phrases_path, load_hallucination_phrases
from speaks.preprocess import preprocess
from speaks.summarize import run_summary
from speaks.transcribe import transcribe

app = typer.Typer(help="音声ファイルを文字起こし + 議事録要約")


@app.command()
def run(
    audio: Annotated[Path, typer.Argument(help="入力WAVファイルのパス")],
    whisper_model: Annotated[str, typer.Option(help="Whisperモデル")] = WHISPER_MODEL,
    ollama_model: Annotated[str, typer.Option(help="Ollamaモデル (要約)")] = OLLAMA_MODEL,
    correction_model: Annotated[
        Optional[str],
        typer.Option(help="ASR 補正に使う Ollama モデル (未指定なら --ollama-model と同じ)"),
    ] = None,
    transcript_only: Annotated[bool, typer.Option("--transcript-only", help="文字起こしのみ")] = False,
    no_diarize: Annotated[bool, typer.Option("--no-diarize", help="話者分離を無効化")] = False,
    diarizer: Annotated[
        str,
        typer.Option(help="話者分離バックエンド (pyannote|sortformer)"),
    ] = "pyannote",
    hf_token: Annotated[Optional[str], typer.Option(help="HuggingFaceトークン")] = None,
    output_dir: Annotated[Path, typer.Option(help="出力ディレクトリ")] = Path("output"),
    initial_prompt: Annotated[
        Optional[str], typer.Option(help="Whisperへの文脈プロンプト (固有名詞補正)")
    ] = None,
    vocab_file: Annotated[
        Optional[Path],
        typer.Option(help="固有名詞リスト (1 行 1 語、# でコメント)"),
    ] = None,
    biasing_mode: Annotated[
        str,
        typer.Option(help="バイアシングモード (prompt|trie|both, Phase 1 は prompt のみ実装)"),
    ] = "prompt",
    skip_preprocess: Annotated[bool, typer.Option("--skip-preprocess", help="前処理 (16kHz化) をスキップ")] = False,
    align: Annotated[
        bool,
        typer.Option("--align/--no-align", help="wav2vec2 forced alignment を有効化"),
    ] = True,
    align_model: Annotated[
        Optional[str], typer.Option(help="forced alignment に使う HF モデル ID")
    ] = None,
    hallucination_filter: Annotated[
        bool,
        typer.Option(
            "--hallucination-filter/--no-hallucination-filter",
            help="幻覚抑制の後処理を有効化",
        ),
    ] = True,
    hallucination_phrases: Annotated[
        Optional[Path],
        typer.Option(help="定型幻覚句の辞書ファイル (未指定なら同梱の日本語辞書)"),
    ] = None,
    correct: Annotated[
        bool,
        typer.Option("--correct/--no-correct", help="LLM による ASR 補正を有効化"),
    ] = True,
    summary_pipeline: Annotated[
        str,
        typer.Option(help="要約パイプライン (single|multi)"),
    ] = "multi",
):
    """音声ファイルを文字起こしして議事録を生成する。"""
    if not audio.exists():
        typer.echo(f"エラー: ファイルが見つかりません: {audio}", err=True)
        raise typer.Exit(1)

    # vocab 読み込み
    vocab: list[str] = []
    if vocab_file is not None:
        try:
            vocab = load_vocab(vocab_file)
        except FileNotFoundError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(1) from None

    biasing_mode = warn_unimplemented_biasing(biasing_mode)
    merged_prompt = build_initial_prompt(initial_prompt, vocab)

    # 幻覚句辞書
    phrases: list[str] = []
    if hallucination_filter:
        phrases_path = hallucination_phrases or default_phrases_path()
        try:
            phrases = load_hallucination_phrases(phrases_path)
        except FileNotFoundError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(1) from None

    # HF トークン (pyannote 用)
    token = hf_token or resolve_hf_token()
    if not no_diarize and diarizer == "pyannote" and not token:
        typer.echo(
            "エラー: 話者分離 (pyannote) には HuggingFace トークンが必要です。\n"
            "  huggingface-cli login でログインするか、\n"
            "  --hf-token <token> または環境変数 HF_TOKEN を設定してください。\n"
            "  Sortformer を使うには --diarizer sortformer、\n"
            "  話者分離なしで実行するには --no-diarize を指定してください。",
            err=True,
        )
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    processed_audio = audio if skip_preprocess else preprocess(audio, output_dir)

    speaker_turns = None
    if not no_diarize:
        diar = make_diarizer(diarizer, token)
        try:
            speaker_turns = diar(str(processed_audio))
        except RuntimeError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(1) from None

    transcript = transcribe(
        str(processed_audio),
        whisper_model,
        speaker_turns,
        merged_prompt,
        align=align,
        align_model=align_model,
        hallucination_filter=hallucination_filter,
        hallucination_phrases=phrases,
    )

    raw_path = output_dir / f"{prefix}_transcript.raw.txt"
    transcript_path = output_dir / f"{prefix}_transcript.txt"

    # LLM 補正
    if correct:
        raw_path.write_text(transcript, encoding="utf-8")
        typer.echo(f"  → {raw_path}", err=True)
        transcript = correct_transcript(
            transcript,
            correction_model or ollama_model,
            vocab=vocab,
            step="[2/3]" if not transcript_only else "[2/2]",
        )

    transcript_path.write_text(transcript, encoding="utf-8")
    typer.echo(f"  → {transcript_path}", err=True)

    if transcript_only:
        return

    step = "[3/3]" if not no_diarize else "[2/2]"
    result = run_summary(transcript, ollama_model, summary_pipeline, step=step)
    if "topics" in result:
        topics_path = output_dir / f"{prefix}_topics.md"
        topics_path.write_text(result["topics"], encoding="utf-8")
        typer.echo(f"  → {topics_path}", err=True)
    minutes_path = output_dir / f"{prefix}_minutes.md"
    minutes_path.write_text(result["minutes"], encoding="utf-8")
    typer.echo(f"  → {minutes_path}", err=True)


if __name__ == "__main__":
    app()
