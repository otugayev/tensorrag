from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .data import extract_explicit_axis_mentions
from .retrieval import expand_node, retrieve_for_query
from .utils import dbg, normalize_text, normalized_terms, unique

_DEFAULT_AXIS_ALIASES = {
    "product": {
        "product",
        "component",
        "part",
        "item",
        "equipment",
        "asset",
        "assembly",
        "subcomponent",
    },
    "location": {
        "location",
        "place",
        "area",
        "site",
        "position",
        "elevation",
        "floor",
        "space",
        "room",
        "zone",
        "pit",
        "where",
    },
    "function": {
        "function",
        "capability",
        "purpose",
        "operation",
        "process",
        "task",
    },
}

_GENERIC_PART_TERMS = {
    "part",
    "component",
    "subcomponent",
}

_ROLE_LABELS = {
    "what_it_is": "product/component hierarchy",
    "where_it_is": "location/place hierarchy",
    "what_it_does": "function/capability hierarchy",
}


def axis_aliases(
    axis_name: str,
    axis_metadata: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    axis_key = axis_name.strip().lower()
    aliases = {axis_key}
    if axis_metadata and axis_name in axis_metadata:
        aliases.update(alias.lower() for alias in axis_metadata[axis_name].get("aliases", []))
    aliases.update(_DEFAULT_AXIS_ALIASES.get(axis_key, set()))
    return aliases


def build_axis_semantics(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    semantics: dict[str, str] = {}
    for axis_name in axes:
        metadata = (axis_metadata or {}).get(axis_name, {})
        if metadata.get("role"):
            semantics[axis_name] = _ROLE_LABELS.get(str(metadata["role"]), str(metadata["role"]))
            continue
        axis_key = axis_name.strip().lower()
        if axis_key == "product":
            semantics[axis_name] = "product/component hierarchy"
        elif axis_key == "location":
            semantics[axis_name] = "location/place hierarchy"
        elif axis_key == "function":
            semantics[axis_name] = "function/capability hierarchy"
        else:
            semantics[axis_name] = "distinct ontology axis; do not equate it with other axes"
    return semantics


def build_axis_policy_spec(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    policy: dict[str, dict[str, Any]] = {}
    for axis_name in axes:
        metadata = (axis_metadata or {}).get(axis_name, {})
        role = str(metadata.get("role") or "").strip().lower()
        axis_key = axis_name.strip().lower()
        if role == "what_it_is" or (not role and axis_key == "product"):
            policy[axis_name] = {
                "structure": "tree",
                "relation": "has-part / part-of",
                "role": "what_it_is",
                "transitive_relation": True,
                "within_axis_inheritance": {
                    "intrinsic": "none",
                    "context": "none",
                    "default_rule": "downward_only",
                    "aggregate": "not_inherited",
                },
                "cross_axis_rule": "structural_join_only",
                "notes": [
                    "A parent has descendants as parts structurally, but does not inherit their intrinsic properties.",
                    "Specific child facts override broader defaults from ancestors.",
                ],
            }
            continue
        if role == "where_it_is" or (not role and axis_key == "location"):
            policy[axis_name] = {
                "structure": "tree",
                "relation": "contains / located-in",
                "role": "where_it_is",
                "transitive_relation": True,
                "within_axis_inheritance": {
                    "intrinsic": "none",
                    "context": "same_branch_context_only",
                    "default_rule": "downward_only",
                    "aggregate": "not_inherited",
                },
                "cross_axis_rule": "structural_join_only",
                "notes": [
                    "Context properties such as floor elevation may support descendants on the same branch if no more specific conflicting fact exists.",
                    "Local properties such as color apply only to the explicitly named location node.",
                ],
            }
            continue
        if role == "what_it_does" or (not role and axis_key == "function"):
            policy[axis_name] = {
                "structure": "tree",
                "relation": "decomposes-into / subfunction-of",
                "role": "what_it_does",
                "transitive_relation": True,
                "within_axis_inheritance": {
                    "intrinsic": "none",
                    "context": "limited_same_branch_only",
                    "default_rule": "downward_only",
                    "aggregate": "not_inherited",
                },
                "cross_axis_rule": "structural_join_only",
                "notes": [
                    "A subfunction is not automatically identical to the whole parent function.",
                    "Leaf-level functional facts do not automatically become properties of the whole branch.",
                ],
            }
            continue
        policy[axis_name] = {
            "structure": "tree",
            "relation": "axis-specific hierarchy",
            "role": role or "axis-specific role",
            "transitive_relation": True,
            "within_axis_inheritance": {
                "intrinsic": "none",
                "context": "same_branch_only_if_explicitly_supported",
                "default_rule": "downward_only",
                "aggregate": "not_inherited",
            },
            "cross_axis_rule": "structural_join_only",
            "notes": [
                "Unknown axis: be conservative and avoid automatic inheritance except clearly scoped defaults.",
            ],
        }
    return policy


def build_property_inheritance_spec() -> dict[str, dict[str, Any]]:
    return {
        "intrinsic": {
            "definition": "Own property of a node such as material, color, mass, size, geometry, local state.",
            "inheritance": "none",
            "examples": ["made of copper", "made of steel", "painted blue", "painted red"],
        },
        "context": {
            "definition": "Branch context property such as floor elevation, belonging to a zone, room, or area.",
            "inheritance": "same_axis_only; typically downward along the same branch when no more specific conflicting fact exists",
            "examples": ["floor is located at elevation 725", "located in zone A"],
        },
        "default_rule": {
            "definition": "A scoped general rule over a subtree.",
            "inheritance": "downward_only_with_exceptions; specific facts override general rules",
            "examples": ["all unit components are made from stainless steel, except the generator"],
        },
        "aggregate": {
            "definition": "Summary or roll-up property of a group, branch, or whole system.",
            "inheritance": "not inherited; compute or state separately",
            "examples": ["total power", "total mass", "overall count"],
        },
        "cross_axis_rule": {
            "definition": "Cross-axis relations are structural joins only.",
            "inheritance": "never transfer predicates across different axes",
            "examples": [
                "location color does not become product color",
                "product material does not become location material",
            ],
        },
    }


def infer_target_answer_axis(
    query: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    normalized_query = normalize_text(query, axes)
    padded_query = f" {normalized_query} "
    query_terms = [term for term in normalized_query.split() if term]
    explicit_query_nodes = extract_explicit_axis_mentions(query, axes)
    explicit_axes = [axis_name for axis_name, node_ids in explicit_query_nodes.items() if node_ids]

    if "location" in axes and (
        normalized_query.startswith("where ")
        or " located " in padded_query
        or " location " in padded_query
    ):
        return "location"

    if "function" in axes and (
        normalized_query.startswith("what does ")
        or normalized_query.startswith("what do ")
        or normalized_query.startswith("what function ")
        or normalized_query.startswith("what functions ")
        or normalized_query.startswith("which function ")
        or normalized_query.startswith("which functions ")
    ):
        return "function"

    if "product" in axes and (
        normalized_query.startswith("what performs ")
        or normalized_query.startswith("which product ")
        or normalized_query.startswith("what product ")
        or normalized_query.startswith("which component ")
        or normalized_query.startswith("what component ")
        or normalized_query.startswith("which part ")
        or normalized_query.startswith("what part ")
        or normalized_query.startswith("which equipment ")
        or normalized_query.startswith("what equipment ")
    ):
        return "product"

    best_axis: str | None = None
    best_score = 0
    tie = False

    for axis_name in axes:
        score = 0
        for alias in axis_aliases(axis_name, axis_metadata):
            alias_norm = normalize_text(alias, axes)
            if not alias_norm:
                continue
            alias_terms = alias_norm.split()
            if len(alias_terms) > 1:
                if f" {alias_norm} " in f" {normalized_query} ":
                    score = max(score, 4)
                continue
            alias_term = alias_terms[0]
            for query_term in query_terms:
                if query_term == alias_term:
                    score = max(score, 4)
                    continue
                if _is_single_edit_or_transposition(query_term, alias_term):
                    score = max(score, 3)
                    continue
                if len(alias_term) >= 5 and (alias_term in query_term or query_term in alias_term):
                    score = max(score, 2)

        if score > best_score:
            best_axis = axis_name
            best_score = score
            tie = False
        elif score and score == best_score:
            tie = True

    if best_score == 0 or tie:
        if len(explicit_axes) == 1:
            return explicit_axes[0]
        return None

    if best_axis == "product" and any(term in _GENERIC_PART_TERMS for term in query_terms):
        if len(explicit_axes) == 1 and explicit_axes[0] != "product":
            return explicit_axes[0]
    return best_axis


def _is_single_edit_or_transposition(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False

    if len(left) == len(right):
        mismatches = [idx for idx, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(mismatches) == 1:
            return True
        if len(mismatches) == 2:
            i, j = mismatches
            return j == i + 1 and left[i] == right[j] and left[j] == right[i]
        return False

    if len(left) > len(right):
        left, right = right, left

    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def _build_text_context(chunks: list[dict[str, str]]) -> str:
    if not chunks:
        return "(no chunks selected)"
    return "\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in chunks)


def _build_prompt(
    query: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
    tensor: list[dict[str, str]],
    retrieval_result: dict[str, Any],
) -> str:
    requested_answer_axis = infer_target_answer_axis(query, axes, axis_metadata)
    text_context = _build_text_context(retrieval_result["top_chunks"])
    struct_context = json.dumps({"axes": axes, "tensor": tensor}, ensure_ascii=False, indent=2)
    admissible_cells = json.dumps(retrieval_result.get("admissible_cells", []), ensure_ascii=False, indent=2)
    admissible_nodes = json.dumps(
        retrieval_result.get("admissible_nodes_by_axis", {}),
        ensure_ascii=False,
        indent=2,
    )
    matched_nodes = json.dumps(
        retrieval_result.get("axis_nodes", {}),
        ensure_ascii=False,
        indent=2,
    )
    entity_hints = json.dumps(
        _build_entity_context_hints(axes, retrieval_result),
        ensure_ascii=False,
        indent=2,
    )
    cross_axis_paths = json.dumps(
        _build_cross_axis_path_hints(axes, retrieval_result),
        ensure_ascii=False,
        indent=2,
    )
    axis_semantics = json.dumps(build_axis_semantics(axes, axis_metadata), ensure_ascii=False, indent=2)
    axis_policy = json.dumps(build_axis_policy_spec(axes, axis_metadata), ensure_ascii=False, indent=2)
    property_policy = json.dumps(build_property_inheritance_spec(), ensure_ascii=False, indent=2)
    explicit_mentions = json.dumps(
        _build_explicit_chunk_mentions(axes, retrieval_result["top_chunks"]),
        ensure_ascii=False,
        indent=2,
    )
    structural_candidates = json.dumps(
        _build_structural_target_candidates(
            axes=axes,
            target_axis=requested_answer_axis,
            retrieval_result=retrieval_result,
        ),
        ensure_ascii=False,
        indent=2,
    )
    branch_support = json.dumps(
        _build_same_axis_support_hints(
            axes=axes,
            retrieval_result=retrieval_result,
        ),
        ensure_ascii=False,
        indent=2,
    )

    return f'''You are a constrained question-answering assistant.

Use only the provided evidence.
Do not use outside knowledge.

STRUCTURED_CONTEXT_JSON:
{struct_context}

FILTERED_RELATIONS_JSON:
{admissible_cells}

ADMISSIBLE_NODES_BY_AXIS_JSON:
{admissible_nodes}

QUERY_MAPPED_NODES_JSON:
{matched_nodes}

ENTITY_CONTEXT_HINTS_JSON:
{entity_hints}

CROSS_AXIS_PATH_HINTS_JSON:
{cross_axis_paths}

AXIS_SEMANTICS_JSON:
{axis_semantics}

AXIS_POLICY_SPEC_JSON:
{axis_policy}

PROPERTY_INHERITANCE_SPEC_JSON:
{property_policy}

EXPLICIT_CHUNK_MENTIONS_JSON:
{explicit_mentions}

STRUCTURAL_TARGET_CANDIDATES_JSON:
{structural_candidates}

SAME_AXIS_SUPPORT_HINTS_JSON:
{branch_support}

REQUESTED_ANSWER_AXIS:
{json.dumps(requested_answer_axis, ensure_ascii=False)}

TEXT_CONTEXT:
{text_context}

QUESTION:
"""{query}"""

Guidelines:
1. Use only facts from the provided contexts.
2. Answer only the asked aspect and ignore unrelated details.
3. If needed, combine multiple provided facts into one supported conclusion.
4. Inheritance within the same axis branch is allowed in both directions only when there is no conflicting more-specific fact.
5. For comparisons, provide a direct comparison when evidence is sufficient.
6. If an entity is present in ADMISSIBLE_NODES_BY_AXIS_JSON, treat it as structurally covered even if the exact leaf token is absent in text.
7. You may use facts from the nearest admissible ancestor or descendant relation on the same axis branch when no conflicting more-specific fact exists.
8. Use ENTITY_CONTEXT_HINTS_JSON to map queried entities to admissible related context when direct mentions are sparse.
9. Use CROSS_AXIS_PATH_HINTS_JSON for general traversal: start from the queried node on its axis, move to the nearest admissible node or nodes on another axis through the tensor, then continue on that new axis to find explicit support.
10. Respect axis semantics strictly. Nodes from different axes are different kinds of entities.
11. A tensor cell means co-membership in one structured relation, not synonymy, not identity, and not automatically a component-of relation across axes.
12. Follow AXIS_POLICY_SPEC_JSON exactly for what may propagate within an axis.
13. Follow PROPERTY_INHERITANCE_SPEC_JSON exactly: intrinsic properties do not propagate; context properties may propagate only within the same location branch; default rules propagate downward only within the same axis branch; aggregate properties are not inherited and must be computed separately.
14. Facts in TEXT_CONTEXT attach only to the entities explicitly named in the same chunk unless an allowed same-axis inheritance rule from AXIS_POLICY_SPEC_JSON and PROPERTY_INHERITANCE_SPEC_JSON applies.
15. FILTERED_RELATIONS_JSON gives admissible correspondence across axes, but it does not transfer properties or predicates from one axis to another.
16. When the question asks for a property or value of a queried node and the property is not explicitly stated on that node's own axis, use CROSS_AXIS_PATH_HINTS_JSON and same-axis traversal to find the nearest admissible support on another axis, then answer from that support.
17. Do not classify a location as a component/product unless TEXT_CONTEXT explicitly says so.
18. For component/part/subcomponent questions, answer from the product axis hierarchy or explicit text only. Do not return location or function nodes as components.
19. For location questions, answer from the location axis hierarchy or explicit text only.
20. If the question asks whether a location is a component of a product, the default answer is no unless explicit text evidence states that component relation.
21. If the question starts from one axis and asks for another axis, first traverse within the source axis to admissible tensor cells, then move across the tensor to the nearest admissible candidate or candidates on the requested axis, then continue within the requested axis branch to gather support.
22. If REQUESTED_ANSWER_AXIS is not null, answer on that axis only when the relevant fact is explicitly supported for that axis, supported by an allowed same-axis inheritance rule, or supported by the nearest admissible cross-axis candidate set in STRUCTURAL_TARGET_CANDIDATES_JSON.
23. When multiple distinct nodes remain admissible on the requested axis, explicitly state that the query has multiple supported answers and list all supported nodes instead of choosing one arbitrarily.
24. If TEXT_CONTEXT explicitly supports only one of the structurally admissible target nodes, answer with that single supported node.
25. Do not use assumptions or probabilistic language; state only evidence-supported conclusions.
26. If evidence is insufficient after rules 1-25, say so briefly.
27. Keep the answer concise.
28. Keep numeric comparisons arithmetically consistent (if A > B, do not describe A as lower than B).
'''
def _build_entity_context_hints(
    axes: dict[str, dict[str, dict[str, str | None]]],
    retrieval_result: dict[str, Any],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    hints: dict[str, dict[str, dict[str, list[str]]]] = {}
    matched_nodes: dict[str, list[str]] = retrieval_result.get("axis_nodes", {})
    admissible_cells: list[dict[str, str]] = retrieval_result.get("admissible_cells", [])

    for axis_name, query_nodes in matched_nodes.items():
        if not query_nodes:
            continue
        axis_hints: dict[str, dict[str, set[str]]] = {}
        for query_node in query_nodes:
            related_by_axis: dict[str, set[str]] = {}
            for cell in admissible_cells:
                cell_axis_node = cell.get(axis_name)
                if not cell_axis_node:
                    continue
                if query_node not in expand_node(axes, axis_name, cell_axis_node):
                    continue
                for other_axis, other_node in cell.items():
                    related_by_axis.setdefault(other_axis, set()).add(other_node)
            axis_hints[query_node] = related_by_axis

        hints[axis_name] = {
            query_node: {
                other_axis: sorted(values)
                for other_axis, values in related.items()
            }
            for query_node, related in axis_hints.items()
        }

    return hints


def _build_cross_axis_path_hints(
    axes: dict[str, dict[str, dict[str, str | None]]],
    retrieval_result: dict[str, Any],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    admissible_cells: list[dict[str, str]] = retrieval_result.get("admissible_cells", [])
    matched_axis_nodes: dict[str, list[str]] = retrieval_result.get("axis_nodes", {})
    hints: dict[str, dict[str, dict[str, list[str]]]] = {}

    for source_axis, source_nodes in matched_axis_nodes.items():
        if not source_nodes:
            continue
        axis_hints: dict[str, dict[str, list[str]]] = {}
        for source_node in source_nodes:
            target_map: dict[str, list[str]] = {}
            for target_axis in axes:
                if target_axis == source_axis:
                    continue
                target_nodes = _best_related_axis_nodes(
                    axes=axes,
                    admissible_cells=admissible_cells,
                    source_axis=source_axis,
                    source_node=source_node,
                    target_axis=target_axis,
                )
                if target_nodes:
                    target_map[target_axis] = target_nodes
            if target_map:
                axis_hints[source_node] = target_map
        if axis_hints:
            hints[source_axis] = axis_hints

    return hints


def _node_display_name(node_id: str) -> str:
    if "." in node_id:
        return node_id.rsplit(".", 1)[-1]
    return node_id


def _phrase_matches(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _extract_answer_candidates(
    answer_text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> list[dict[str, Any]]:
    normalized_answer = normalize_text(answer_text, axes)
    candidates: list[dict[str, Any]] = []

    for axis_name, axis_nodes in axes.items():
        for node_id in axis_nodes:
            full_phrase = normalize_text(node_id.replace(".", " ").replace("->", " "), axes)
            display_phrase = normalize_text(_node_display_name(node_id), axes)

            score = None
            matched_phrase = None
            if full_phrase and _phrase_matches(normalized_answer, full_phrase):
                score = (len(full_phrase.split()), 2, len(full_phrase))
                matched_phrase = full_phrase
            elif display_phrase and _phrase_matches(normalized_answer, display_phrase):
                score = (len(display_phrase.split()), 1, len(display_phrase))
                matched_phrase = display_phrase

            if score is None:
                continue

            candidates.append(
                {
                    "axis": axis_name,
                    "node_id": node_id,
                    "matched_phrase": matched_phrase,
                    "score": score,
                }
            )

    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _normalize_answer_node_spelling(
    answer_text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
) -> str:
    normalized = answer_text
    all_node_ids: list[str] = []
    for axis_nodes in axes.values():
        all_node_ids.extend(axis_nodes.keys())

    for node_id in sorted(set(all_node_ids), key=len, reverse=True):
        if node_id in normalized:
            normalized = normalized.replace(node_id, _node_display_name(node_id))

    normalized = re.sub(r'"([^"\n]+)"', r"\1", normalized)
    return normalized


def _replace_node_spelling(answer_text: str, old_node: str, new_node: str) -> str:
    rewritten = answer_text
    old_full = old_node
    new_full = new_node
    old_display = _node_display_name(old_node)
    new_display = _node_display_name(new_node)

    if old_full and old_full in rewritten:
        rewritten = rewritten.replace(old_full, new_full)
    if old_display and old_display in rewritten:
        rewritten = rewritten.replace(old_display, new_display)
    return rewritten


def _collapse_projected_self_reference(answer_text: str, node_id: str) -> str:
    rewritten = answer_text
    node_variants = [node_id, _node_display_name(node_id)]
    for node_text in sorted({variant for variant in node_variants if variant}, key=len, reverse=True):
        escaped = re.escape(node_text)
        rewritten = re.sub(
            rf"\b{escaped} is [^,.]{{0,120}}(?:the )?{escaped}, which\b",
            f"{node_text}, which",
            rewritten,
        )
        rewritten = re.sub(
            rf"\b{escaped}, which is\b",
            f"{node_text} is",
            rewritten,
        )
        rewritten = re.sub(
            rf"\b{escaped}, which has\b",
            f"{node_text} has",
            rewritten,
        )
    return rewritten


def _project_answer_to_query_branch(
    answer_text: str,
    target_axis: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    query_target_nodes: list[str],
) -> str:
    if not query_target_nodes:
        return answer_text

    candidates = _extract_answer_candidates(answer_text, axes)
    if not candidates:
        return answer_text

    answer_axis = candidates[0]["axis"]
    answer_node = candidates[0]["node_id"]
    if answer_axis != target_axis or answer_node in query_target_nodes:
        return answer_text

    same_branch_query_nodes = [
        query_node
        for query_node in query_target_nodes
        if _is_same_axis_branch(axes, target_axis, answer_node, query_node)
    ]
    if len(same_branch_query_nodes) != 1:
        return answer_text

    rewritten = _replace_node_spelling(answer_text, answer_node, same_branch_query_nodes[0])
    return _collapse_projected_self_reference(rewritten, same_branch_query_nodes[0])


def _build_explicit_chunk_mentions(
    axes: dict[str, dict[str, dict[str, str | None]]],
    chunks: list[dict[str, str]],
) -> dict[str, dict[str, list[str]]]:
    return {
        chunk["id"]: extract_explicit_axis_mentions(chunk["text"], axes)
        for chunk in chunks
    }


def _axis_and_node_terms(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
    node_ids: list[str],
) -> set[str]:
    terms: set[str] = set()
    for axis_name in axes:
        for alias in axis_aliases(axis_name, axis_metadata):
            terms.update(normalized_terms(alias, axes))
    for node_id in node_ids:
        terms.update(normalized_terms(node_id, axes))
        terms.update(normalized_terms(_node_display_name(node_id), axes))
    return terms


def _query_predicate_terms(
    query: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
) -> set[str]:
    terms = set(normalized_terms(query, axes))
    explicit_query_nodes = extract_explicit_axis_mentions(query, axes)
    query_node_ids = [
        node_id
        for node_ids in explicit_query_nodes.values()
        for node_id in node_ids
    ]
    predicate_terms = terms - _axis_and_node_terms(axes, axis_metadata, query_node_ids)
    return predicate_terms or terms


def _answer_predicate_terms(
    answer_text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
) -> set[str]:
    terms = set(normalized_terms(answer_text, axes))
    answer_node_ids = [
        candidate["node_id"]
        for candidate in _extract_answer_candidates(answer_text, axes)
    ]
    predicate_terms = terms - _axis_and_node_terms(axes, axis_metadata, answer_node_ids)
    return predicate_terms


def _select_fact_support_chunks_from_terms(
    predicate_terms: set[str],
    axes: dict[str, dict[str, dict[str, str | None]]],
    chunks: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not chunks:
        return []

    if not predicate_terms:
        return []

    best_score = 0
    scored: list[tuple[int, dict[str, str]]] = []
    for chunk in chunks:
        score = len(predicate_terms & normalized_terms(chunk["text"], axes))
        scored.append((score, chunk))
        best_score = max(best_score, score)

    if best_score > 0:
        return [chunk for score, chunk in scored if score == best_score]
    return []


def _select_fact_support_chunks(
    query: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
    chunks: list[dict[str, str]],
) -> list[dict[str, str]]:
    predicate_terms = _query_predicate_terms(query, axes, axis_metadata)
    support_chunks = _select_fact_support_chunks_from_terms(predicate_terms, axes, chunks)
    if support_chunks:
        return support_chunks
    return [chunks[0]] if chunks else []


def _select_answer_support_chunks(
    answer_text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
    chunks: list[dict[str, str]],
) -> list[dict[str, str]]:
    predicate_terms = _answer_predicate_terms(answer_text, axes, axis_metadata)
    return _select_fact_support_chunks_from_terms(predicate_terms, axes, chunks)


def _format_insufficient_axis_answer(target_axis: str) -> str:
    return f"Insufficient evidence to identify a {target_axis.lower()} for this query."


def _is_same_or_ancestor(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    candidate_ancestor: str,
    candidate_descendant: str,
) -> bool:
    current = candidate_descendant
    while current:
        if current == candidate_ancestor:
            return True
        current = axes.get(axis_name, {}).get(current, {}).get("parent")
    return False


def _is_same_axis_branch(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    left_node: str,
    right_node: str,
) -> bool:
    return _is_same_or_ancestor(axes, axis_name, left_node, right_node) or _is_same_or_ancestor(
        axes, axis_name, right_node, left_node
    )


def _node_depth(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    node_id: str,
) -> int:
    depth = 0
    current = node_id
    while current:
        current = axes.get(axis_name, {}).get(current, {}).get("parent")
        if current:
            depth += 1
    return depth


def _branch_distance(
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_name: str,
    left_node: str,
    right_node: str,
) -> int | None:
    if left_node not in axes.get(axis_name, {}) or right_node not in axes.get(axis_name, {}):
        return None

    if left_node == right_node:
        return 0

    left_distances = {left_node: 0}
    current = left_node
    distance = 0
    while True:
        parent = axes.get(axis_name, {}).get(current, {}).get("parent")
        if not parent:
            break
        distance += 1
        left_distances[parent] = distance
        current = parent

    current = right_node
    distance = 0
    while True:
        if current in left_distances:
            return distance + left_distances[current]
        parent = axes.get(axis_name, {}).get(current, {}).get("parent")
        if not parent:
            break
        distance += 1
        current = parent

    return None


def _best_related_axis_nodes(
    axes: dict[str, dict[str, dict[str, str | None]]],
    admissible_cells: list[dict[str, str]],
    source_axis: str,
    source_node: str,
    target_axis: str,
) -> list[str]:
    candidates: list[tuple[int, int, int, str]] = []
    for cell in admissible_cells:
        source_cell_node = cell.get(source_axis)
        target_cell_node = cell.get(target_axis)
        if not source_cell_node or not target_cell_node:
            continue
        distance = _branch_distance(axes, source_axis, source_node, source_cell_node)
        if distance is None:
            continue
        candidates.append(
            (
                distance,
                -_node_depth(axes, source_axis, source_cell_node),
                _node_depth(axes, target_axis, target_cell_node),
                target_cell_node,
            )
        )

    if not candidates:
        return []

    candidates.sort()
    best_rank = candidates[0][:2]
    best_targets = {
        (target_depth, target_node)
        for distance, depth_rank, target_depth, target_node in candidates
        if (distance, depth_rank) == best_rank
    }
    return [
        target_node
        for _, target_node in sorted(best_targets, key=lambda item: (item[0], _node_display_name(item[1]), item[1]))
    ]


def _build_structural_target_candidates(
    axes: dict[str, dict[str, dict[str, str | None]]],
    target_axis: str | None,
    retrieval_result: dict[str, Any],
) -> dict[str, list[str]]:
    if target_axis is None:
        return {}

    admissible_cells = retrieval_result.get("admissible_cells", [])
    matched_axis_nodes = retrieval_result.get("axis_nodes", {})
    candidates: dict[str, list[str]] = {}

    for source_axis, source_nodes in matched_axis_nodes.items():
        if source_axis == target_axis or not source_nodes:
            continue
        for source_node in source_nodes:
            target_nodes = _best_related_axis_nodes(
                axes=axes,
                admissible_cells=admissible_cells,
                source_axis=source_axis,
                source_node=source_node,
                target_axis=target_axis,
            )
            if not target_nodes:
                continue
            candidates.setdefault(source_node, [])
            for target_node in target_nodes:
                if target_node not in candidates[source_node]:
                    candidates[source_node].append(target_node)

    return candidates


def _build_same_axis_support_hints(
    axes: dict[str, dict[str, dict[str, str | None]]],
    retrieval_result: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    hints: dict[str, dict[str, list[dict[str, str]]]] = {}
    matched_axis_nodes = retrieval_result.get("axis_nodes", {})
    top_chunks = retrieval_result.get("top_chunks", [])

    for axis_name, query_nodes in matched_axis_nodes.items():
        if not query_nodes:
            continue
        axis_hints: dict[str, list[dict[str, str]]] = {}
        for query_node in query_nodes:
            candidates: list[tuple[int, int, str, str]] = []
            for chunk in top_chunks:
                mentions = extract_explicit_axis_mentions(chunk["text"], axes).get(axis_name, [])
                for mentioned_node in mentions:
                    distance = _branch_distance(axes, axis_name, query_node, mentioned_node)
                    if distance is None:
                        continue
                    candidates.append(
                        (
                            distance,
                            -_node_depth(axes, axis_name, mentioned_node),
                            chunk["id"],
                            mentioned_node,
                        )
                    )
            candidates.sort()
            axis_hints[query_node] = [
                {"chunk_id": chunk_id, "node_id": node_id}
                for _, _, chunk_id, node_id in candidates[:3]
            ]
        hints[axis_name] = axis_hints

    return hints


def _select_query_relevant_support_chunks(
    target_axis: str,
    query_target_nodes: list[str],
    axes: dict[str, dict[str, dict[str, str | None]]],
    chunks: list[dict[str, str]],
) -> list[dict[str, str]]:
    relevant_chunks: list[dict[str, str]] = []
    for chunk in chunks:
        chunk_mentions = extract_explicit_axis_mentions(chunk["text"], axes)
        target_mentions = chunk_mentions.get(target_axis, [])
        if not target_mentions:
            continue
        keep = False
        for mentioned_node in target_mentions:
            if any(
                _is_same_axis_branch(axes, target_axis, mentioned_node, query_node)
                for query_node in query_target_nodes
            ):
                keep = True
                break
        if keep:
            relevant_chunks.append(chunk)
    return relevant_chunks


def _extract_axis_answer_nodes(
    answer_text: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    target_axis: str,
) -> list[str]:
    node_ids: list[str] = []
    for candidate in _extract_answer_candidates(answer_text, axes):
        if candidate["axis"] != target_axis:
            continue
        node_id = candidate["node_id"]
        if node_id not in node_ids:
            node_ids.append(node_id)
    return node_ids


def _flatten_structural_target_nodes(structural_candidates: dict[str, list[str]]) -> list[str]:
    node_ids: list[str] = []
    for target_nodes in structural_candidates.values():
        for node_id in target_nodes:
            if node_id not in node_ids:
                node_ids.append(node_id)
    return node_ids


def _best_aligned_structural_target_nodes(
    target_axis: str,
    matched_axis_nodes: dict[str, list[str]],
    admissible_cells: list[dict[str, str]],
    axes: dict[str, dict[str, dict[str, str | None]]],
    top_chunks: list[dict[str, str]] | None = None,
) -> list[str]:
    constrained_source_axes = [
        axis_name
        for axis_name, node_ids in matched_axis_nodes.items()
        if axis_name != target_axis and node_ids
    ]
    if not constrained_source_axes:
        return []

    scored_cells: list[tuple[tuple[int, int], str]] = []
    for cell in admissible_cells:
        target_node = cell.get(target_axis)
        if not target_node:
            continue

        axis_distances: list[int] = []
        for axis_name in constrained_source_axes:
            cell_node = cell.get(axis_name)
            if not cell_node:
                axis_distances = []
                break

            distance_options = [
                distance
                for query_node in matched_axis_nodes[axis_name]
                if (distance := _branch_distance(axes, axis_name, query_node, cell_node)) is not None
            ]
            if not distance_options:
                axis_distances = []
                break
            axis_distances.append(min(distance_options))

        if not axis_distances:
            continue

        scored_cells.append(((sum(axis_distances), max(axis_distances)), target_node))

    if not scored_cells:
        return []

    scored_cells.sort(key=lambda item: item[0])
    best_rank = scored_cells[0][0]
    target_nodes = unique(
        target_node
        for rank, target_node in scored_cells
        if rank == best_rank
    )

    if top_chunks:
        explicitly_supported = _explicitly_supported_structural_target_nodes(
            target_axis=target_axis,
            target_nodes=target_nodes,
            axes=axes,
            chunks=top_chunks,
        )
        if explicitly_supported:
            return explicitly_supported

    return target_nodes


def _explicitly_supported_structural_target_nodes(
    target_axis: str,
    target_nodes: list[str],
    axes: dict[str, dict[str, dict[str, str | None]]],
    chunks: list[dict[str, str]],
) -> list[str]:
    target_node_set = set(target_nodes)
    supported_node_set: set[str] = set()

    for chunk in chunks:
        mentions = extract_explicit_axis_mentions(chunk["text"], axes).get(target_axis, [])
        for node_id in mentions:
            if node_id in target_node_set:
                supported_node_set.add(node_id)

    return [node_id for node_id in target_nodes if node_id in supported_node_set]


def _axis_label(axis_name: str) -> str:
    axis_key = axis_name.strip().lower()
    if axis_key in {"product", "location", "function"}:
        return axis_key
    return axis_key or "node"


def _pluralize_axis_label(axis_name: str) -> str:
    label = _axis_label(axis_name)
    if label.endswith("y"):
        return f"{label[:-1]}ies"
    if label.endswith("s"):
        return label
    return f"{label}s"


def _format_node_list(node_ids: list[str]) -> str:
    labels = [_node_display_name(node_id) for node_id in node_ids]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _format_structural_multi_answer(
    target_axis: str,
    source_node: str | None,
    target_nodes: list[str],
) -> str:
    plural_label = _pluralize_axis_label(target_axis)
    if source_node:
        return (
            f"The {plural_label} associated with {_node_display_name(source_node)} are "
            f"{_format_node_list(target_nodes)}."
        )
    return f"The admissible {plural_label} are {_format_node_list(target_nodes)}."


def _structural_multi_answer(
    target_axis: str,
    matched_axis_nodes: dict[str, list[str]],
    structural_candidates: dict[str, list[str]],
    admissible_cells: list[dict[str, str]],
    axes: dict[str, dict[str, dict[str, str | None]]],
    top_chunks: list[dict[str, str]],
) -> tuple[str | None, list[str]] | None:
    if matched_axis_nodes.get(target_axis):
        return None
    flattened_target_nodes = _best_aligned_structural_target_nodes(
        target_axis=target_axis,
        matched_axis_nodes=matched_axis_nodes,
        admissible_cells=admissible_cells,
        axes=axes,
        top_chunks=top_chunks,
    ) or _flatten_structural_target_nodes(structural_candidates)
    if len(flattened_target_nodes) <= 1:
        return None

    explicitly_supported = _explicitly_supported_structural_target_nodes(
        target_axis=target_axis,
        target_nodes=flattened_target_nodes,
        axes=axes,
        chunks=top_chunks,
    )
    if len(explicitly_supported) == 1:
        return None

    source_node: str | None = None
    if len(structural_candidates) == 1:
        source_node = next(iter(structural_candidates))

    target_nodes = explicitly_supported if len(explicitly_supported) > 1 else flattened_target_nodes
    return source_node, target_nodes


def _validate_answer_against_requested_axis(
    answer_text: str,
    query: str,
    target_axis: str,
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
    matched_axis_nodes: dict[str, list[str]],
    admissible_cells: list[dict[str, str]],
    top_chunks: list[dict[str, str]],
) -> str | None:
    candidates = _extract_answer_candidates(answer_text, axes)
    query_target_nodes = matched_axis_nodes.get(target_axis, [])
    structural_candidates = _build_structural_target_candidates(
        axes=axes,
        target_axis=target_axis,
        retrieval_result={
            "axis_nodes": matched_axis_nodes,
            "admissible_cells": admissible_cells,
        },
    )
    structural_target_nodes = _best_aligned_structural_target_nodes(
        target_axis=target_axis,
        matched_axis_nodes=matched_axis_nodes,
        admissible_cells=admissible_cells,
        axes=axes,
        top_chunks=top_chunks,
    ) or _flatten_structural_target_nodes(structural_candidates)
    answer_support_chunks = _select_answer_support_chunks(answer_text, axes, axis_metadata, top_chunks)
    if answer_support_chunks:
        support_chunks = answer_support_chunks
    elif query_target_nodes:
        support_chunks = _select_query_relevant_support_chunks(
            target_axis=target_axis,
            query_target_nodes=query_target_nodes,
            axes=axes,
            chunks=top_chunks,
        )
    else:
        support_chunks = _select_fact_support_chunks(query, axes, axis_metadata, top_chunks)

    support_mentions: dict[str, list[str]] = {axis_name: [] for axis_name in axes}
    for chunk in support_chunks:
        chunk_mentions = extract_explicit_axis_mentions(chunk["text"], axes)
        for axis_name, node_ids in chunk_mentions.items():
            for node_id in node_ids:
                if node_id not in support_mentions[axis_name]:
                    support_mentions[axis_name].append(node_id)

    if not candidates:
        return None

    answer_target_nodes = _extract_axis_answer_nodes(answer_text, axes, target_axis)
    if not answer_target_nodes:
        return _format_insufficient_axis_answer(target_axis)
    if query_target_nodes and not all(node_id in query_target_nodes for node_id in answer_target_nodes):
        return _format_insufficient_axis_answer(target_axis)

    if not support_mentions.get(target_axis):
        if not structural_target_nodes:
            return _format_insufficient_axis_answer(target_axis)
        if len(structural_target_nodes) > 1 and not query_target_nodes:
            if set(answer_target_nodes) == set(structural_target_nodes):
                return None
            return _format_insufficient_axis_answer(target_axis)
        if all(node_id in structural_target_nodes for node_id in answer_target_nodes):
            return None
        return _format_insufficient_axis_answer(target_axis)

    if not all(
        any(_is_same_axis_branch(axes, target_axis, support_node, answer_node) for support_node in support_mentions[target_axis])
        for answer_node in answer_target_nodes
    ):
        return _format_insufficient_axis_answer(target_axis)
    return None


def answer_any(
    client: OpenAI,
    chunks: list[dict[str, str]],
    axes: dict[str, dict[str, dict[str, str | None]]],
    axis_metadata: dict[str, dict[str, Any]] | None,
    tensor: list[dict[str, str]],
    axis_chunks: dict[str, dict[str, list[str]]],
    cell_chunks: dict[str, list[str]] | None,
    chunk_embeddings: dict[str, list[float]] | None,
    query: str,
    top_k: int,
) -> str:
    query_embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    if not query_embedding_response.data:
        raise RuntimeError("Failed to build query embedding.")
    query_embedding = query_embedding_response.data[0].embedding

    retrieval_result = retrieve_for_query(
        chunks=chunks,
        query=query,
        top_k=top_k,
        tensor=tensor,
        axis_chunks=axis_chunks,
        cell_chunks=cell_chunks,
        axes=axes,
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
    )
    dbg("PRIMARY_CHUNK_IDS", retrieval_result.get("primary_chunk_ids", []))
    dbg("SELECTED_CHUNK_IDS", [chunk["id"] for chunk in retrieval_result["top_chunks"]])

    prompt = _build_prompt(query, axes, axis_metadata, tensor, retrieval_result)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer_text = _normalize_answer_node_spelling(
        response.choices[0].message.content.strip(),
        axes,
    )
    target_axis = infer_target_answer_axis(query, axes, axis_metadata)
    if target_axis:
        structural_candidates = _build_structural_target_candidates(
            axes=axes,
            target_axis=target_axis,
            retrieval_result=retrieval_result,
        )
        answer_text = _project_answer_to_query_branch(
            answer_text=answer_text,
            target_axis=target_axis,
            axes=axes,
            query_target_nodes=retrieval_result.get("axis_nodes", {}).get(target_axis, []),
        )
        structural_multi = _structural_multi_answer(
            target_axis=target_axis,
            matched_axis_nodes=retrieval_result.get("axis_nodes", {}),
            structural_candidates=structural_candidates,
            admissible_cells=retrieval_result.get("admissible_cells", []),
            axes=axes,
            top_chunks=retrieval_result["top_chunks"],
        )
        if structural_multi:
            structural_multi_source_node, structural_multi_target_nodes = structural_multi
            if set(_extract_axis_answer_nodes(answer_text, axes, target_axis)) != set(structural_multi_target_nodes):
                return _format_structural_multi_answer(
                    target_axis=target_axis,
                    source_node=structural_multi_source_node,
                    target_nodes=structural_multi_target_nodes,
                )
        validated = _validate_answer_against_requested_axis(
            answer_text=answer_text,
            query=query,
            target_axis=target_axis,
            axes=axes,
            axis_metadata=axis_metadata,
            matched_axis_nodes=retrieval_result.get("axis_nodes", {}),
            admissible_cells=retrieval_result.get("admissible_cells", []),
            top_chunks=retrieval_result["top_chunks"],
        )
        if validated:
            if structural_multi:
                structural_multi_source_node, structural_multi_target_nodes = structural_multi
                return _format_structural_multi_answer(
                    target_axis=target_axis,
                    source_node=structural_multi_source_node,
                    target_nodes=structural_multi_target_nodes,
                )
            return validated
    return answer_text
