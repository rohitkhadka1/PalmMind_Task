from __future__ import annotations

from typing import Iterable, List


def split_recursive(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def _split(block: str, level: int = 0) -> Iterable[str]:
        if len(block) <= max_tokens or level >= len(separators):
            yield block
            return
        sep = separators[level]
        parts = block.split(sep)
        if len(parts) == 1:
            yield from _split(block, level + 1)
            return
        acc: list[str] = []
        for p in parts:
            candidate = (sep.join(acc + [p]) + sep) if sep.strip() else ("".join(acc + [p]))
            if len(candidate) > max_tokens and acc:
                chunk = sep.join(acc)
                yield from _window(chunk, max_tokens, overlap)
                acc = [p]
            else:
                acc.append(p)
        if acc:
            chunk = sep.join(acc)
            yield from _window(chunk, max_tokens, overlap)

    def _window(block: str, size: int, ov: int) -> Iterable[str]:
        if len(block) <= size:
            yield block
            return
        start = 0
        while start < len(block):
            end = min(start + size, len(block))
            yield block[start:end]
            if end == len(block):
                break
            start = end - ov

    return [c.strip() for c in _split(text, 0) if c.strip()]


def split_fixed(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    if size <= 0:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = end - overlap
    return [c.strip() for c in chunks if c.strip()]

