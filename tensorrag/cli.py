from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .answering import answer_any
from .cache import build_cache_payload, default_cache_name, load_cache, save_cache
from .data import (
    AXIS_CHUNKING_VERSION,
    print_structure_snapshot,
)
from .utils import load_client, set_debug

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
CACHE_DIR = PROJECT_DIR / "cache"
CANONICAL_EXAMPLE_NAME = "iec81346_example.json"


def ensure_repo_layout() -> None:
    for directory in (INPUT_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_path(
    path: str | None,
    default_name: str,
    *,
    preferred_dir: Path,
    search_dirs: tuple[Path, ...],
) -> str:
    if path is None:
        return str(preferred_dir / default_name)

    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)

    for base_dir in search_dirs:
        resolved = base_dir / candidate
        if resolved.exists():
            return str(resolved)

    return str(preferred_dir / candidate)


def resolve_input_path(path: str | None) -> str:
    return resolve_path(
        path,
        CANONICAL_EXAMPLE_NAME,
        preferred_dir=INPUT_DIR,
        search_dirs=(PROJECT_DIR, INPUT_DIR),
    )


def resolve_cache_path(path: str | None, input_path: str) -> str:
    return resolve_path(
        path,
        default_cache_name(input_path),
        preferred_dir=CACHE_DIR,
        search_dirs=(PROJECT_DIR, CACHE_DIR),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TensorRAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build cache from input data")
    build_parser.add_argument(
        "--input",
        dest="input_path",
        default=CANONICAL_EXAMPLE_NAME,
        help="JSON file name or path; ./input is searched automatically",
    )
    build_parser.add_argument("--cache", help="Cache JSON path")
    build_parser.add_argument("--debug", action="store_true", help="Enable debug logs")

    ask_parser = subparsers.add_parser("ask", help="Answer a query from cache")
    ask_parser.add_argument(
        "--input",
        dest="input_path",
        default=CANONICAL_EXAMPLE_NAME,
        help="JSON file name or path; ./input is searched automatically",
    )
    ask_parser.add_argument("--cache", help="Cache JSON path")
    ask_parser.add_argument("--query", required=True, help="User query")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ask_parser.add_argument("--debug", action="store_true", help="Enable debug logs")

    return parser.parse_args()


def build_command(args: argparse.Namespace) -> None:
    ensure_repo_layout()
    set_debug(args.debug)
    input_path = resolve_input_path(args.input_path)
    cache_path = resolve_cache_path(args.cache, input_path)

    client = load_client()
    payload = build_cache_payload(input_path=input_path, client=client)
    save_cache(cache_path, payload)
    print(f"Cache saved to {cache_path}")


def load_runtime_data(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    ensure_repo_layout()
    input_path = resolve_input_path(args.input_path)
    cache_path = resolve_cache_path(args.cache, input_path)
    if not Path(cache_path).exists():
        raise FileNotFoundError(
            "Cache file not found: "
            f"{cache_path}. Run `tensorrag build --input \"{input_path}\"` first."
        )
    payload = load_cache(cache_path)
    return payload, input_path, cache_path


def validate_cache_payload(payload: dict[str, Any], cache_path: str, input_path: str) -> None:
    missing_keys = [
        key
        for key in ("tensor", "axes", "chunks", "axis_chunks", "cell_chunks", "chunk_embeddings")
        if key not in payload
    ]
    if missing_keys:
        raise ValueError(
            "Cache file is incomplete: "
            f"{cache_path}. Missing keys: {', '.join(missing_keys)}. "
            f"Run `tensorrag build --input \"{input_path}\"` again."
        )

    if payload.get("axis_chunking_version") != AXIS_CHUNKING_VERSION:
        raise ValueError(
            "Cache file is outdated: "
            f"{cache_path}. Run `tensorrag build --input \"{input_path}\"` again."
        )


def ask_command(args: argparse.Namespace) -> None:
    set_debug(args.debug)
    payload, input_path, cache_path = load_runtime_data(args)
    validate_cache_payload(payload, cache_path, input_path)
    client = load_client()

    if args.debug:
        print_structure_snapshot(payload["axes"], payload["tensor"], input_path=input_path)

    answer = answer_any(
        client=client,
        chunks=payload["chunks"],
        axes=payload["axes"],
        axis_metadata=payload.get("axis_metadata"),
        tensor=payload["tensor"],
        axis_chunks=payload["axis_chunks"],
        cell_chunks=payload.get("cell_chunks"),
        chunk_embeddings=payload.get("chunk_embeddings"),
        query=args.query,
        top_k=args.top_k,
    )
    print(answer)


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_command(args)
        return
    ask_command(args)


if __name__ == "__main__":
    main()
