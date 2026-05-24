"""speaks CLI エントリポイント (typer)。"""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from speaks.diarize import diarize, resolve_hf_token
from speaks.io import OLLAMA_MODEL, WHISPER_MODEL
from speaks.preprocess import preprocess
from speaks.summarize import summarize
from speaks.transcribe import transcribe

app = typer.Typer(help="音声ファイルを文字起こし + 議事録要約")


@app.command()
def run(
    audio: Annotated[Path, typer.Argument(help="入力WAVファイルのパス")],
    whisper_model: Annotated[str, typer.Option(help="Whisperモデル")] = WHISPER_MODEL,
    ollama_model: Annotated[str, typer.Option(help="Ollamaモデル")] = OLLAMA_MODEL,
    transcript_only: Annotated[bool, typer.Option("--transcript-only", help="文字起こしのみ")] = False,
    no_diarize: Annotated[bool, typer.Option("--no-diarize", help="話者分離を無効化")] = False,
    hf_token: Annotated[Optional[str], typer.Option(help="HuggingFaceトークン")] = None,
    output_dir: Annotated[Path, typer.Option(help="出力ディレクトリ")] = Path("output"),
    initial_prompt: Annotated[Optional[str], typer.Option(help="Whisperへの文脈プロンプト (固有名詞補正)")] = None,
    skip_preprocess: Annotated[bool, typer.Option("--skip-preprocess", help="前処理 (16kHz化) をスキップ")] = False,
):
    """音声ファイルを文字起こしして議事録を生成する。"""
    if not audio.exists():
        typer.echo(f"エラー: ファイルが見つかりません: {audio}", err=True)
        raise typer.Exit(1)

    token = hf_token or resolve_hf_token()
    if not no_diarize and not token:
        typer.echo(
            "エラー: 話者分離には HuggingFace トークンが必要です。\n"
            "  huggingface-cli login でログインするか、\n"
            "  --hf-token <token> または環境変数 HF_TOKEN を設定してください。\n"
            "  話者分離なしで実行するには --no-diarize を指定してください。",
            err=True,
        )
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    processed_audio = audio if skip_preprocess else preprocess(audio, output_dir)

    speaker_turns = None
    if not no_diarize:
        speaker_turns = diarize(str(processed_audio), token)

    transcript = transcribe(
        str(processed_audio), whisper_model, speaker_turns, initial_prompt
    )

    transcript_path = output_dir / f"{prefix}_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    typer.echo(f"  → {transcript_path}", err=True)

    if not transcript_only:
        step = "[3/3]" if not no_diarize else "[2/2]"
        summary = summarize(transcript, ollama_model, step)
        minutes_path = output_dir / f"{prefix}_minutes.md"
        minutes_path.write_text(summary, encoding="utf-8")
        typer.echo(f"  → {minutes_path}", err=True)


if __name__ == "__main__":
    app()
