from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .utils import (
    canonical_cell_id,
    clean,
    display_with_underscores,
    normalize_text,
    underscore_display_text,
    unique,
)

AXIS_CHUNKING_VERSION = 2


def _load_json_input(input_path: str) -> dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object.")
    return payload


def load_tensor_axes(input_path: str) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, str | None]]]]:
    payload = _load_json_input(input_path)
    raw_axes = payload.get("axes")
    raw_tensor = payload.get("tensor")
    if not isinstance(raw_axes, dict) or not raw_axes:
        raise ValueError("JSON input must contain a non-empty 'axes' object.")
    if not isinstance(raw_tensor, list) or not raw_tensor:
        raise ValueError("JSON input must contain a non-empty 'tensor' list.")

    axes: dict[str, dict[str, dict[str, str | None]]] = {}
    for axis_name, node_map in raw_axes.items():
        if not isinstance(node_map, dict):
            raise ValueError(f"Axis '{axis_name}' must map node ids to parent ids.")
        axes[axis_name] = {}
        for node_id, parent_id in node_map.items():
            axes[axis_name][str(node_id)] = {
                "parent": clean(parent_id),
            }

    tensor: list[dict[str, str]] = []
    for raw_cell in raw_tensor:
        if not isinstance(raw_cell, dict):
            raise ValueError("Every tensor cell in JSON must be an object.")
        cell: dict[str, str] = {}
        for axis_name in axes:
            node_id = clean(raw_cell.get(axis_name))
            if not node_id:
                raise ValueError(f"Tensor cell is missing axis '{axis_name}'.")
            if node_id not in axes[axis_name]:
                raise ValueError(f"Tensor cell references unknown node '{node_id}' on axis '{axis_name}'.")
            cell[axis_name] = node_id
        tensor.append(cell)

    return tensor, axes


def load_chunks(input_path: str) -> list[dict[str, str]]:
    payload = _load_json_input(input_path)
    raw_chunks = payload.get("text")
    if not isinstance(raw_chunks, list):
        raise ValueError("JSON input must contain a 'text' list.")
    chunks: list[dict[str, str]] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise ValueError("Every text chunk in JSON must be an object.")
        chunk_id = clean(raw_chunk.get("id"))
        text = clean(raw_chunk.get("text"))
        if not chunk_id or not text:
            continue
        chunks.append({"id": chunk_id, "text": text})
    return chunks


def load_axis_metadata(
    input_path: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> dict[str, dict[str, Any]]:
    payload = _load_json_input(input_path)
    raw_metadata = payload.get("axis_meta", {})
    metadata: dict[str, dict[str, Any]] = {}

    if isinstance(raw_metadata, dict):
        items = raw_metadata.items()
    elif isinstance(raw_metadata, list):
        items = [
            (clean(item.get("axis_name")), item)
            for item in raw_metadata
            if isinstance(item, dict)
        ]
    else:
        raise ValueError("'axis_meta' in JSON must be an object or a list.")

    for axis_name, raw_meta in items:
        if not axis_name or axis_name not in axes or not isinstance(raw_meta, dict):
            continue
        aliases_raw = raw_meta.get("aliases", [])
        if isinstance(aliases_raw, str):
            aliases = unique([alias.strip() for alias in aliases_raw.split(";") if alias.strip()])
        elif isinstance(aliases_raw, list):
            aliases = unique([str(alias).strip() for alias in aliases_raw if str(alias).strip()])
        else:
            aliases = []
        metadata[axis_name] = {
            "role": clean(raw_meta.get("role")),
            "aliases": aliases,
        }

    return metadata


def _node_candidate_phrases(
    node_id: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> list[str]:
    def strip_leading_code(segment: str) -> str:
        words = segment.strip().split()
        if len(words) >= 2 and re.fullmatch(r"[=+\-]?[a-zA-Z]+\d+[a-zA-Z0-9]*", words[0]):
            return " ".join(words[1:])
        return segment.strip()

    phrases: list[str] = []
    display_node_id = ".".join(
        stripped
        for stripped in (strip_leading_code(segment) for segment in node_id.split("."))
        if stripped
    )
    full_variants = (
        node_id.replace(".", " ").replace("->", " "),
        node_id.replace(".", " ").replace("->", " to "),
        display_node_id.replace(".", " ").replace("->", " "),
        display_node_id.replace(".", " ").replace("->", " to "),
    )
    leaf = node_id.split(".")[-1]
    display_leaf = strip_leading_code(leaf)
    leaf_variants = (
        leaf.replace("->", " "),
        leaf.replace("->", " to "),
        display_leaf.replace("->", " "),
        display_leaf.replace("->", " to "),
    )
    for phrase in [
        *(normalize_text(variant, axes) for variant in full_variants),
        *(normalize_text(variant, axes) for variant in leaf_variants),
    ]:
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def extract_explicit_axis_mentions(
    text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> dict[str, list[str]]:
    normalized = normalize_text(text, axes)
    mentions: dict[str, list[str]] = {axis_name: [] for axis_name in axes}
    if not normalized:
        return mentions

    matches: list[dict[str, Any]] = []
    for axis_name, axis_nodes in axes.items():
        for node_id in axis_nodes:
            for phrase in _node_candidate_phrases(node_id, axes):
                pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])")
                for match in pattern.finditer(normalized):
                    matches.append(
                        {
                            "axis": axis_name,
                            "node_id": node_id,
                            "start": match.start(),
                            "end": match.end(),
                            "token_len": len(phrase.split()),
                            "char_len": len(phrase),
                            "depth": len(node_id.split(".")),
                        }
                    )

    matches.sort(
        key=lambda item: (item["token_len"], item["char_len"], item["depth"]),
        reverse=True,
    )

    occupied: list[tuple[int, int]] = []
    for match in matches:
        span = (match["start"], match["end"])
        if any(not (span[1] <= taken[0] or span[0] >= taken[1]) for taken in occupied):
            continue
        occupied.append(span)
        axis_name = match["axis"]
        node_id = match["node_id"]
        if node_id not in mentions[axis_name]:
            mentions[axis_name].append(node_id)

    for axis_name, node_ids in mentions.items():
        mentions[axis_name] = sorted(
            node_ids,
            key=lambda node_id: min(
                (
                    match["start"]
                    for match in matches
                    if match["axis"] == axis_name and match["node_id"] == node_id
                ),
                default=0,
            ),
        )

    return mentions


def build_axis_chunks(
    chunks: list[dict[str, str]],
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> dict[str, dict[str, list[str]]]:
    axis_chunks = {
        axis_name: {node_id: [] for node_id in axis_nodes}
        for axis_name, axis_nodes in axes.items()
    }

    for chunk in chunks:
        chunk_mentions = extract_explicit_axis_mentions(chunk["text"], axes)
        for axis_name, node_ids in chunk_mentions.items():
            for node_id in node_ids:
                axis_chunks[axis_name][node_id].append(chunk["id"])

    for axis_name, by_node in axis_chunks.items():
        for node_id, ids in by_node.items():
            axis_chunks[axis_name][node_id] = unique(ids)

    return axis_chunks


def build_cell_chunks(
    tensor: list[dict[str, str]],
    axis_chunks: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    chunk_hits_by_axis: dict[str, dict[str, set[str]]] = {}
    for axis_name, node_to_chunk_ids in axis_chunks.items():
        for node_id, chunk_ids in node_to_chunk_ids.items():
            for chunk_id in chunk_ids:
                chunk_hits_by_axis.setdefault(chunk_id, {}).setdefault(axis_name, set()).add(node_id)

    cell_chunks: dict[str, list[str]] = {canonical_cell_id(cell): [] for cell in tensor}
    for chunk_id, axis_hits in chunk_hits_by_axis.items():
        matched_axes = [axis_name for axis_name, node_ids in axis_hits.items() if node_ids]
        # Treat cell evidence conservatively: a chunk must explicitly ground at least
        # two axes before we use it as direct support for a tensor cell.
        if len(matched_axes) < 2:
            continue

        for cell in tensor:
            supported_axes = 0
            conflict = False
            for axis_name, node_ids in axis_hits.items():
                cell_node = cell.get(axis_name)
                if cell_node in node_ids:
                    supported_axes += 1
                    continue
                conflict = True
                break
            if conflict or supported_axes < 2:
                continue
            cell_chunks[canonical_cell_id(cell)].append(chunk_id)

    for cell_id, chunk_ids in cell_chunks.items():
        cell_chunks[cell_id] = unique(chunk_ids)

    return cell_chunks


def build_chunk_embeddings(
    chunks: list[dict[str, str]],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> dict[str, list[float]]:
    embeddings_by_chunk_id: dict[str, list[float]] = {}
    if not chunks:
        return embeddings_by_chunk_id

    for start in range(0, len(chunks), max(1, int(batch_size))):
        batch = chunks[start : start + max(1, int(batch_size))]
        inputs = [chunk["text"] for chunk in batch]
        response = client.embeddings.create(model=model, input=inputs)
        for chunk, item in zip(batch, response.data):
            embeddings_by_chunk_id[chunk["id"]] = item.embedding

    return embeddings_by_chunk_id


def format_axis_tree(axis_nodes: dict[str, dict[str, Any]]) -> str:
    if not axis_nodes:
        return "(empty)"
    children: dict[str, list[str]] = {node_id: [] for node_id in axis_nodes}
    roots: list[str] = []
    for node_id, meta in axis_nodes.items():
        parent = meta.get("parent")
        if parent and parent in children:
            children[parent].append(node_id)
        else:
            roots.append(node_id)
    for node_id in children:
        children[node_id].sort()

    lines: list[str] = []

    def walk(node_id: str, depth: int) -> None:
        display_leaf = underscore_display_text(node_id.split(".")[-1])
        display_node = underscore_display_text(node_id)
        lines.append(f'{"  " * depth}- {display_leaf} ({display_node})')
        for child in children[node_id]:
            walk(child, depth + 1)

    for root in sorted(set(roots)):
        walk(root, 0)
    return "\n".join(lines)


def print_structure_snapshot(
    axes: dict[str, dict[str, dict[str, str | None]]],
    tensor: list[dict[str, str]],
    input_path: str | None = None,
) -> None:
    print("\n==== STRUCTURE SNAPSHOT ====")
    if input_path:
        print(f"DATA FILE: {input_path}")
    for axis_name in sorted(axes):
        print(f"\n==== {axis_name.upper()} TREE ====")
        print(format_axis_tree(axes[axis_name]))
    print("\n==== TENSOR_CELLS ====")
    print(json.dumps(display_with_underscores(tensor), ensure_ascii=False, indent=2))
