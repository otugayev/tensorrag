from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .data import (
    AXIS_CHUNKING_VERSION,
    build_axis_chunks,
    build_cell_chunks,
    build_chunk_embeddings,
    load_axis_metadata,
    load_chunks,
    load_tensor_axes,
)


def default_cache_name(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return f"{base}.tensor_cache.json"


def default_cache_path(input_path: str, directory: str | None = None) -> str:
    directory = directory or os.path.dirname(input_path) or "."
    return os.path.join(directory, default_cache_name(input_path))


def build_cache_payload(input_path: str, client: OpenAI) -> dict[str, Any]:
    tensor, axes = load_tensor_axes(input_path)
    axis_metadata = load_axis_metadata(input_path, axes)
    chunks = load_chunks(input_path)
    axis_chunks = build_axis_chunks(chunks, axes)
    cell_chunks = build_cell_chunks(tensor, axis_chunks)
    chunk_embeddings = build_chunk_embeddings(chunks, client)
    return {
        "input_path": os.path.abspath(input_path),
        "embedding_model": "text-embedding-3-small",
        "axis_chunking_version": AXIS_CHUNKING_VERSION,
        "tensor": tensor,
        "axes": axes,
        "axis_metadata": axis_metadata,
        "chunks": chunks,
        "axis_chunks": axis_chunks,
        "cell_chunks": cell_chunks,
        "chunk_embeddings": chunk_embeddings,
    }


def save_cache(cache_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_cache(cache_path: str) -> dict[str, Any]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
