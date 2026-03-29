from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

DEBUG = False
PROJECT_DIR = Path(__file__).resolve().parent.parent

STOPWORDS = {
    "a",
    "an",
    "are",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "where",
    "which",
}


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = enabled


def dbg(title: str, obj: Any | None = None) -> None:
    if not DEBUG:
        return
    print(f"\n==== {title} ====")
    if obj is None:
        return
    try:
        print(json.dumps(display_with_underscores(obj), ensure_ascii=False, indent=2))
    except Exception:
        print(display_with_underscores(obj))


def clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def canonical_cell_id(cell: dict[str, str]) -> str:
    return json.dumps(cell, ensure_ascii=False, sort_keys=True)


def _basic_normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _axis_segment_phrases(
    axes: dict[str, dict[str, dict[str, str | None]]] | None,
) -> list[str]:
    if not axes:
        return []

    phrases: set[str] = set()
    for axis_nodes in axes.values():
        for node_id in axis_nodes:
            for raw_segment in str(node_id).split("."):
                for normalized_segment in (
                    _basic_normalize_text(raw_segment.replace("->", " ")),
                    _basic_normalize_text(raw_segment.replace("->", " to ")),
                ):
                    if len(normalized_segment.split()) >= 2:
                        phrases.add(normalized_segment)

    return sorted(phrases, key=lambda phrase: (len(phrase.split()), len(phrase)), reverse=True)


def normalize_text(
    text: str,
    axes: dict[str, dict[str, dict[str, str | None]]] | None = None,
) -> str:
    normalized = _basic_normalize_text(text)
    if not axes:
        return normalized

    for phrase in _axis_segment_phrases(axes):
        token = phrase.replace(" ", "_")
        pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])")
        normalized = pattern.sub(token, normalized)

    return normalized


def compact_text(
    text: str,
    axes: dict[str, dict[str, dict[str, str | None]]] | None = None,
) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text, axes))


def normalized_terms(
    text: str,
    axes: dict[str, dict[str, dict[str, str | None]]] | None = None,
) -> set[str]:
    terms: set[str] = set()
    for token in normalize_text(text, axes).split():
        if token in STOPWORDS:
            continue
        terms.add(token)
        if len(token) > 3 and token.endswith("s") and "_" not in token:
            terms.add(token[:-1])
    return terms


def underscore_display_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", str(text).strip())
    return collapsed.replace(" ", "_")


def display_with_underscores(value: Any) -> Any:
    if isinstance(value, str):
        return underscore_display_text(value)
    if isinstance(value, list):
        return [display_with_underscores(item) for item in value]
    if isinstance(value, dict):
        return {
            display_with_underscores(key): display_with_underscores(item)
            for key, item in value.items()
        }
    return value

def load_client() -> OpenAI:
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)
