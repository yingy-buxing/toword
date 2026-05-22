from __future__ import annotations

import argparse
import json
import os
import site
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal


OutputFormat = Literal["txt", "srt", "vtt", "json"]
LogCallback = Callable[[str], None]


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio or video file with faster-whisper.",
    )
    parser.add_argument("input", type=Path, help="Audio/video file to transcribe.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file or output directory. Defaults to ./outputs/<input-stem>.<format>.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("txt", "srt", "vtt", "json"),
        default="txt",
        help="Output format.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="large-v3",
        help="Whisper model name or local CTranslate2 model path.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Inference device. Use cuda for NVIDIA GPU acceleration.",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type, e.g. float16, int8_float16, int8.",
    )
    parser.add_argument(
        "--cuda-dll-dir",
        type=Path,
        help="Windows-only directory that contains CUDA DLLs such as cublas64_12.dll.",
    )
    parser.add_argument(
        "--language",
        help="Language code such as zh, en, ja. Omit for auto-detection.",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Transcribe original language or translate speech to English.",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size.")
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Filter long non-speech sections with Silero VAD.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word timestamps in JSON output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Use BatchedInferencePipeline when > 0. Faster but uses more VRAM.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output: Path | None, output_format: OutputFormat) -> Path:
    if output is None:
        return Path("outputs") / f"{input_path.stem}.{output_format}"

    if output.suffix:
        return output

    return output / f"{input_path.stem}.{output_format}"


def format_timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def render_txt(segments: Iterable[TranscriptSegment]) -> str:
    return "\n".join(segment.text.strip() for segment in segments if segment.text.strip()) + "\n"


def render_srt(segments: Iterable[TranscriptSegment]) -> str:
    blocks: list[str] = []
    for segment in segments:
        blocks.append(
            "\n".join(
                (
                    str(segment.index),
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}",
                    segment.text.strip(),
                )
            )
        )
    return "\n\n".join(blocks) + "\n"


def render_vtt(segments: Iterable[TranscriptSegment]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        blocks.append(
            "\n".join(
                (
                    f"{format_timestamp(segment.start, '.')} --> {format_timestamp(segment.end, '.')}",
                    segment.text.strip(),
                )
            )
        )
        blocks.append("")
    return "\n".join(blocks)


def render_json(segments: list[TranscriptSegment], info: object) -> str:
    payload = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": [asdict(segment) for segment in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_transcript(segments: list[TranscriptSegment], info: object, output_format: OutputFormat) -> str:
    if output_format == "txt":
        return render_txt(segments)
    if output_format == "srt":
        return render_srt(segments)
    if output_format == "vtt":
        return render_vtt(segments)
    return render_json(segments, info)


def load_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Install it with `pip install -r requirements.txt`."
        ) from exc

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def cuda_dll_candidates(user_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if user_dir is not None:
        candidates.append(user_dir.expanduser())

    env_dir = os.environ.get("TOWORD_CUDA_DLL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    for root in {Path(sys.prefix), Path(sys.base_prefix)}:
        candidates.append(root / "Lib" / "site-packages" / "torch" / "lib")

    try:
        candidates.append(Path(site.getusersitepackages()) / "torch" / "lib")
    except Exception:
        pass

    return candidates


def prepare_windows_cuda_dlls(device: str, user_dir: Path | None) -> Path | None:
    if os.name != "nt" or device == "cpu":
        return None

    for candidate in cuda_dll_candidates(user_dir):
        if (candidate / "cublas64_12.dll").exists():
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(candidate))
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"
            return candidate

    return None


def transcribe(
    args: argparse.Namespace,
    log_callback: LogCallback | None = None,
) -> tuple[list[TranscriptSegment], object]:
    log = log_callback or print
    model = load_model(args.model, args.device, args.compute_type)

    if args.batch_size > 0:
        from faster_whisper import BatchedInferencePipeline

        runner = BatchedInferencePipeline(model=model)
        raw_segments, info = runner.transcribe(
            str(args.input),
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            language=args.language,
            task=args.task,
            vad_filter=args.vad_filter,
            word_timestamps=args.word_timestamps,
        )
    else:
        raw_segments, info = model.transcribe(
            str(args.input),
            beam_size=args.beam_size,
            language=args.language,
            task=args.task,
            vad_filter=args.vad_filter,
            word_timestamps=args.word_timestamps,
        )

    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments, start=1):
        log(
            f"[{format_timestamp(segment.start, '.')} -> {format_timestamp(segment.end, '.')}] "
            f"{segment.text.strip()}"
        )
        segments.append(
            TranscriptSegment(
                index=index,
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
        )

    return segments, info


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 2

    args.input = input_path
    output_path = resolve_output_path(input_path, args.output, args.format).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading model '{args.model}' on {args.device} with compute_type={args.compute_type}...",
        flush=True,
    )
    cuda_dll_dir = prepare_windows_cuda_dlls(args.device, args.cuda_dll_dir)
    if cuda_dll_dir is not None:
        print(f"Using CUDA DLL directory: {cuda_dll_dir}", flush=True)

    try:
        segments, info = transcribe(args)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        if args.device == "cuda":
            print(
                "CUDA mode needs a supported NVIDIA driver plus cuBLAS/cuDNN runtime libraries.",
                file=sys.stderr,
            )
        return 1

    output_path.write_text(render_transcript(segments, info, args.format), encoding="utf-8")
    print(f"Saved transcript: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
