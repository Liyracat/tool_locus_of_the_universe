from __future__ import annotations

import re
from typing import Optional
from fastapi import HTTPException

_IMPORT_DELIM_PATTERN = re.compile(r"(。|！　|？　|♪　|[a-z]\. |\)\. |\.\" |! |\? )")


def _normalize_import_text(raw_text: str) -> str:
    return raw_text.replace("\r\n", "\n").replace("\r", "\n")


def _find_first_delim_end(text: str, start: int) -> Optional[int]:
    match = _IMPORT_DELIM_PATTERN.search(text, pos=start)
    return match.end() if match else None


def _find_first_double_newline(text: str, start: int) -> Optional[int]:
    idx = text.find("\n\n", start)
    return idx if idx != -1 else None


def _split_speaker_text(text: str, *, allow_overlap: bool) -> list[str]:
    parts: list[str] = []
    text_len = len(text)
    start = 0
    prev_end = 0
    while start < text_len:
        while start < text_len and text[start].isspace():
            start += 1
        if start >= text_len:
            break
        search_from = max(start, prev_end) if allow_overlap else start
        min_double_start = start + 300
        double_search_from = max(search_from, min_double_start)
        next_double = _find_first_double_newline(text, double_search_from)

        next_delim = None
        min_delim_start = start + 450
        if text_len > min_delim_start:
            delim_search_from = max(search_from, min_delim_start)
            next_delim = _find_first_delim_end(text, delim_search_from)

        end: Optional[int] = None
        if next_double is not None:
            end = next_double
        elif next_delim is not None:
            end = next_delim

        if end is None:
            end = start + 600 if (start + 600) < text_len else text_len

        if end <= start:
            end = min(start + 1, text_len)

        segment = text[start:end].strip()
        if segment:
            parts.append(segment)

        prev_end = end
        if prev_end >= text_len:
            break

        if allow_overlap:
            overlap_start = max(0, prev_end - 80)
            overlap_delim = _find_first_delim_end(text, overlap_start)
            if overlap_delim is not None and overlap_delim < prev_end:
                start = overlap_delim
                continue
        start = prev_end

    return parts


def _merge_short_segments(segments: list[str], *, min_len: int) -> list[str]:
    if not segments:
        return []
    merged: list[str] = []
    buffer = ""
    for segment in segments:
        if buffer:
            buffer = f"{buffer}\n{segment}"
        else:
            buffer = segment
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def _split_import_blocks(raw_text: str, speaker_map: dict[str, dict]) -> list[tuple[dict, str]]:
    normalized = _normalize_import_text(raw_text)
    lines = normalized.split("\n")
    blocks: list[tuple[dict, str]] = []
    current_speaker: Optional[dict] = None
    current_lines: list[str] = []

    def flush_block() -> None:
        nonlocal current_lines
        if current_speaker is None:
            return
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append((current_speaker, text))
        current_lines = []

    for line in lines:
        normalized_line = line.strip().replace("：", ":")
        if normalized_line:
            speaker_label, remainder = normalized_line.split(":", 1) if ":" in normalized_line else (None, None)
            candidate_label = speaker_label or normalized_line
            if candidate_label and candidate_label in speaker_map:
                flush_block()
                current_speaker = speaker_map[candidate_label]
                remainder = remainder.strip() if remainder else ""
                if remainder:
                    current_lines.append(remainder)
                continue
        if current_speaker is None:
            if not normalized_line:
                continue
            raise HTTPException(status_code=400, detail="Speaker definition line is required before content.")
        current_lines.append(line.rstrip())

    flush_block()
    return blocks


def split_import_text(raw_text: str, speaker_map: dict[str, dict]) -> list[dict]:
    parts: list[dict] = []
    message_id = 0
    text_id = 0
    for speaker, text in _split_import_blocks(raw_text, speaker_map):
        message_id += 1
        text_id = 0
        segments = _split_speaker_text(text, allow_overlap=True)
        segments = _merge_short_segments(segments, min_len=300)
        for segment in segments:
            text_id += 1
            parts.append(
                {
                    "message_id": message_id,
                    "text_id": text_id,
                    "speaker_id": speaker["speaker_id"],
                    "speaker_name": speaker["speaker_name"],
                    "contents": segment,
                    "conversation_at": None,
                }
            )
    return parts
