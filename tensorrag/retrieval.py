from __future__ import annotations

import math

from .data import extract_explicit_axis_mentions
from .utils import canonical_cell_id, normalized_terms


def expand_node(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    node_id: str,
) -> set[str]:
    if axis_name not in axes or node_id not in axes[axis_name]:
        return set()

    axis_nodes = axes[axis_name]
    parent_of: dict[str, str | None] = {node: meta.get("parent") for node, meta in axis_nodes.items()}
    children: dict[str, list[str]] = {}
    for node, parent in parent_of.items():
        if parent and parent in axis_nodes:
            children.setdefault(parent, []).append(node)

    out: set[str] = {node_id}

    stack = [node_id]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child in out:
                continue
            out.add(child)
            stack.append(child)

    current = node_id
    while True:
        parent = parent_of.get(current)
        if not parent or parent in out:
            break
        out.add(parent)
        current = parent

    return out


def _ancestor_nodes(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    node_id: str,
) -> list[str]:
    out: list[str] = []
    current = axes.get(axis_name, {}).get(node_id, {}).get("parent")
    while current:
        out.append(current)
        current = axes.get(axis_name, {}).get(current, {}).get("parent")
    return out


def _same_axis_branch_nodes(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    node_id: str,
) -> set[str]:
    return expand_node(axes, axis_name, node_id)


def _direct_node_chunk_ids(
    axis_chunks: dict[str, dict[str, list[str]]],
    axis_name: str,
    node_id: str,
) -> set[str]:
    return set(axis_chunks.get(axis_name, {}).get(node_id, []))


def infer_axis_nodes_from_query(
    query: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> dict[str, list[str]]:
    return extract_explicit_axis_mentions(query, axes)


def filter_tensor_cells(
    tensor: list[dict[str, str]],
    axes: dict[str, dict[str, dict[str, str | None]]],
    matched_axis_nodes: dict[str, list[str]],
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    constrained_axes = [axis for axis, nodes in matched_axis_nodes.items() if nodes]

    if not constrained_axes:
        admissible_nodes: dict[str, set[str]] = {axis_name: set() for axis_name in axes}
        for cell in tensor:
            for axis_name, node_id in cell.items():
                admissible_nodes[axis_name].update(expand_node(axes, axis_name, node_id))
        return list(tensor), admissible_nodes

    admissible_cells: list[dict[str, str]] = []
    for cell in tensor:
        keep = True
        for axis_name in constrained_axes:
            cell_node = cell.get(axis_name)
            if not cell_node:
                keep = False
                break
            expanded = expand_node(axes, axis_name, cell_node)
            if not any(query_node in expanded for query_node in matched_axis_nodes[axis_name]):
                keep = False
                break
        if keep:
            admissible_cells.append(cell)

    admissible_nodes: dict[str, set[str]] = {axis_name: set() for axis_name in axes}
    for cell in admissible_cells:
        for axis_name, node_id in cell.items():
            admissible_nodes[axis_name].update(expand_node(axes, axis_name, node_id))

    return admissible_cells, admissible_nodes


def _lexical_score(
    query: str,
    text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> int:
    return len(normalized_terms(query, axes) & normalized_terms(text, axes))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_for_query(
    chunks: list[dict[str, str]],
    query: str,
    top_k: int,
    tensor: list[dict[str, str]],
    axis_chunks: dict[str, dict[str, list[str]]],
    cell_chunks: dict[str, list[str]] | None,
    axes: dict[str, dict[str, dict[str, str | None]]],
    query_embedding: list[float] | None = None,
    chunk_embeddings: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    matched_axis_nodes = infer_axis_nodes_from_query(query, axes)
    admissible_cells, admissible_nodes_by_axis = filter_tensor_cells(tensor, axes, matched_axis_nodes)

    primary_ids: set[str] = set()
    if cell_chunks is not None:
        for cell in admissible_cells:
            primary_ids.update(cell_chunks.get(canonical_cell_id(cell), []))

    candidate_ids: set[str] = set()
    for axis_name, node_ids in admissible_nodes_by_axis.items():
        for node_id in node_ids:
            candidate_ids.update(axis_chunks.get(axis_name, {}).get(node_id, []))
    candidate_ids.update(primary_ids)

    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    candidates = [chunks_by_id[cid] for cid in sorted(candidate_ids) if cid in chunks_by_id]
    if not candidates:
        candidates = list(chunks)

    def rank_key(chunk: dict[str, str]) -> tuple[int, float, int, str]:
        lexical = _lexical_score(query, chunk["text"], axes)
        semantic = 0.0
        if query_embedding is not None and chunk_embeddings is not None:
            chunk_vec = chunk_embeddings.get(chunk["id"])
            if chunk_vec is not None:
                semantic = _cosine_similarity(query_embedding, chunk_vec)
        return (1 if chunk["id"] in primary_ids else 0, semantic, lexical, chunk["id"])

    ranked = sorted(candidates, key=rank_key, reverse=True)
    cap = max(1, int(top_k))

    # Coverage pass: include at least one high-ranked chunk per mapped node branch.
    selected_ids: set[str] = set()
    selected_chunks: list[dict[str, str]] = []

    def add_first_ranked_match(chunk_ids: set[str]) -> None:
        if len(selected_chunks) >= cap:
            return
        for chunk in ranked:
            cid = chunk["id"]
            if cid in chunk_ids and cid not in selected_ids:
                selected_ids.add(cid)
                selected_chunks.append(chunk)
                return

    for axis_name, query_nodes in matched_axis_nodes.items():
        if len(selected_chunks) >= cap:
            break
        if not query_nodes:
            continue
        deferred_cross_axis_ids: list[set[str]] = []
        deferred_context_ids: list[set[str]] = []
        for query_node in query_nodes:
            if len(selected_chunks) >= cap:
                break
            same_axis_branch_chunk_ids: set[str] = set()
            direct_node_chunk_ids = _direct_node_chunk_ids(axis_chunks, axis_name, query_node)
            same_axis_context_chunk_ids: set[str] = set()
            cross_axis_chunk_ids: set[str] = set()
            supporting_cells: list[dict[str, str]] = []
            for branch_node in _same_axis_branch_nodes(axes, axis_name, query_node):
                same_axis_branch_chunk_ids.update(axis_chunks.get(axis_name, {}).get(branch_node, []))
            same_axis_context_chunk_ids = same_axis_branch_chunk_ids - direct_node_chunk_ids
            for cell in admissible_cells:
                cell_axis_node = cell.get(axis_name)
                if not cell_axis_node:
                    continue
                if query_node not in expand_node(axes, axis_name, cell_axis_node):
                    continue
                supporting_cells.append(cell)
                if cell_chunks is not None:
                    cross_axis_chunk_ids.update(cell_chunks.get(canonical_cell_id(cell), []))
                for cell_axis, cell_node in cell.items():
                    if cell_axis == axis_name:
                        continue
                    for related_node in expand_node(axes, cell_axis, cell_node):
                        cross_axis_chunk_ids.update(axis_chunks.get(cell_axis, {}).get(related_node, []))
            add_first_ranked_match(same_axis_branch_chunk_ids)
            add_first_ranked_match(same_axis_context_chunk_ids)

            deferred_cross_axis_ids.append(cross_axis_chunk_ids)

            # Add one supporting context chunk from ancestor nodes on other axes.
            context_chunk_ids: set[str] = set()
            for cell in supporting_cells:
                for cell_axis, cell_node in cell.items():
                    if cell_axis == axis_name:
                        continue
                    for ancestor in _ancestor_nodes(axes, cell_axis, cell_node):
                        context_chunk_ids.update(axis_chunks.get(cell_axis, {}).get(ancestor, []))
            deferred_context_ids.append(context_chunk_ids)

        for cross_axis_chunk_ids in deferred_cross_axis_ids:
            if len(selected_chunks) >= cap:
                break
            add_first_ranked_match(cross_axis_chunk_ids)

        for context_chunk_ids in deferred_context_ids:
            if len(selected_chunks) >= cap:
                break
            add_first_ranked_match(context_chunk_ids)

    top_chunks = list(selected_chunks)
    for chunk in ranked:
        if len(top_chunks) >= cap:
            break
        if chunk["id"] in selected_ids:
            continue
        selected_ids.add(chunk["id"])
        top_chunks.append(chunk)

    return {
        "top_chunks": top_chunks,
        "axis_nodes": matched_axis_nodes,
        "admissible_cells": admissible_cells,
        "admissible_nodes_by_axis": {k: sorted(v) for k, v in admissible_nodes_by_axis.items()},
        "primary_chunk_ids": sorted(primary_ids),
    }
