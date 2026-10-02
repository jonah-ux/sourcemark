"""Find a quote inside a document again, even after the document changed.

The strategy follows the fuzzy-anchoring approach used by web annotation
tools: try the recorded position, then every exact occurrence of the quote
(disambiguated by surrounding context and distance from the old position),
then a fuzzy search. The fuzzy search is a k-gram offset vote: every shared
k-gram between quote and document votes for the document offset where the
quote would start; the winning offset is then scored with a sequence matcher.
That keeps it linear-ish on large files while tolerating edits inside the quote.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .textnorm import normalize_newlines, squash

DEFAULT_CONTEXT = 32
DEFAULT_K = 8
DEFAULT_MIN_SIMILARITY = 0.72
# A fuzzy hit is only trusted when it is very similar, or moderately similar AND its
# surroundings agree. Code is full of near-identical lines; similarity alone over-matches.
STRONG_SIMILARITY = 0.90
MIN_CONTEXT_FOR_FUZZY = 0.55


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    similarity: float  # 1.0 == exact quote text
    method: str  # "position" | "exact" | "loose" | "fuzzy"
    context_score: float  # 0..1, how well prefix/suffix agree


def _context_score(doc: str, start: int, end: int, prefix: str, suffix: str, *, worst: bool = False) -> float:
    """Agreement of the text around [start, end) with the recorded prefix/suffix.

    ``worst=True`` returns the weaker side: a deleted span leaves its prefix and
    suffix adjacent, so one side can agree with a *neighbouring* line while the
    other does not. Trust requires both sides.
    """
    if not prefix and not suffix:
        return 1.0
    scores = []
    if prefix:
        got = doc[max(0, start - len(prefix)) : start]
        scores.append(SequenceMatcher(None, got, prefix, autojunk=False).ratio())
    if suffix:
        got = doc[end : end + len(suffix)]
        scores.append(SequenceMatcher(None, got, suffix, autojunk=False).ratio())
    return min(scores) if worst else sum(scores) / len(scores)


def _all_occurrences(doc: str, needle: str) -> list[int]:
    out, i = [], doc.find(needle)
    while i != -1:
        out.append(i)
        i = doc.find(needle, i + 1)
    return out


def locate(
    doc: str,
    exact: str,
    prefix: str = "",
    suffix: str = "",
    hint_start: int | None = None,
    *,
    k: int = DEFAULT_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    strong_similarity: float = STRONG_SIMILARITY,
    min_context: float = MIN_CONTEXT_FOR_FUZZY,
    max_fuzzy_len: int = 20000,
    fuzzy: bool = True,
) -> Match | None:
    """Locate ``exact`` in ``doc``. Returns the best match or None."""
    doc = normalize_newlines(doc)
    exact = normalize_newlines(exact)
    if not exact:
        return None

    # 1. Recorded position still holds the same text.
    if hint_start is not None and doc[hint_start : hint_start + len(exact)] == exact:
        ctx = _context_score(doc, hint_start, hint_start + len(exact), prefix, suffix)
        # A decoy copy can land on the old offset; only trust it if the context agrees
        # or if it is the only copy.
        if ctx >= 0.5 or doc.count(exact) == 1:
            return Match(hint_start, hint_start + len(exact), 1.0, "position", ctx)

    # 2+. Gather every candidate and score them on ONE scale. A far-away copy whose
    # surroundings disagree must not beat an edited line sitting in its original context.
    cands: list[Match] = []
    for pos in _all_occurrences(doc, exact):
        cands.append(Match(pos, pos + len(exact), 1.0, "exact", _context_score(doc, pos, pos + len(exact), prefix, suffix)))
    if not cands:
        loose = _loose_find(doc, exact, hint_start)
        if loose is not None:
            s_, e_ = loose
            cands.append(Match(s_, e_, 1.0, "loose", _context_score(doc, s_, e_, prefix, suffix)))

    def worst(m: Match) -> float:
        return _context_score(doc, m.start, m.end, prefix, suffix, worst=True)

    best = max(cands, key=lambda m: _score(m, worst(m), hint_start, len(doc))) if cands else None
    # An exact/loose hit with poor surroundings may be a coincidental copy; let a fuzzy
    # candidate near the old position compete.
    if fuzzy and len(exact) <= max_fuzzy_len and (best is None or ((prefix or suffix) and worst(best) < 0.5)):
        fz = _fuzzy(doc, exact, prefix, suffix, hint_start, k, min_similarity)
        if fz is not None and _trusted_fuzzy(doc, fz, prefix, suffix, hint_start, strong_similarity, min_context):
            if best is None or _score(fz, worst(fz), hint_start, len(doc)) > _score(best, worst(best), hint_start, len(doc)):
                best = fz
    return best


def _score(m: Match, worst_ctx: float, hint_start: int | None, doc_len: int) -> float:
    """Similarity dominates; two-sided context and proximity to the old offset break ties."""
    prox = 0.0
    if hint_start is not None and doc_len:
        prox = 1.0 - min(1.0, abs(m.start - hint_start) / max(1, doc_len))
    return 0.55 * m.similarity + 0.35 * worst_ctx + 0.10 * prox


def _trusted_fuzzy(doc, m, prefix, suffix, hint_start, strong, min_ctx) -> bool:
    if m.similarity >= strong:
        return True
    if (prefix or suffix) and _context_score(doc, m.start, m.end, prefix, suffix, worst=True) >= min_ctx:
        return True
    # Edited in place: a moderately similar match within a few lines of where it was.
    if hint_start is not None and m.similarity >= 0.75:
        lo, hi = sorted((hint_start, m.start))
        if doc.count("\n", lo, hi) <= 3:
            return True
    return False


def _loose_find(doc: str, exact: str, hint_start: int | None) -> tuple[int, int] | None:
    """Find ``exact`` ignoring whitespace differences; map back to raw offsets."""
    target = squash(exact)
    if not target:
        return None
    # Build squashed doc with a map from squashed index -> raw index.
    raw_idx: list[int] = []
    chars: list[str] = []
    prev_ws = True
    for i, ch in enumerate(doc):
        if ch.isspace():
            if not prev_ws:
                chars.append(" ")
                raw_idx.append(i)
            prev_ws = True
        else:
            chars.append(ch)
            raw_idx.append(i)
            prev_ws = False
    sq = "".join(chars)
    hits = _all_occurrences(sq, target)
    if not hits:
        return None
    if hint_start is not None:
        hits.sort(key=lambda h: abs(raw_idx[h] - hint_start))
    h = hits[0]
    start = raw_idx[h]
    end = raw_idx[h + len(target) - 1] + 1
    return start, end


def _fuzzy(
    doc: str,
    exact: str,
    prefix: str,
    suffix: str,
    hint_start: int | None,
    k: int,
    min_similarity: float,
) -> Match | None:
    k = max(3, min(k, len(exact)))
    grams: dict[str, list[int]] = {}
    for i in range(0, len(exact) - k + 1):
        grams.setdefault(exact[i : i + k], []).append(i)
    if not grams:
        return None

    votes: Counter[int] = Counter()
    # Coarse bucket so small insertions/deletions inside the quote still agree.
    bucket = max(8, len(exact) // 8)
    for j in range(0, len(doc) - k + 1):
        positions = grams.get(doc[j : j + k])
        if positions:
            for p in positions[:4]:
                votes[(j - p) // bucket] += 1
    if not votes:
        return None

    best: Match | None = None
    slack = max(16, len(exact) // 4)
    for b, _count in votes.most_common(5):
        approx = max(0, b * bucket)
        lo = max(0, approx - slack)
        hi = min(len(doc), approx + len(exact) + 2 * slack + bucket)
        window = doc[lo:hi]
        sm = SequenceMatcher(None, window, exact, autojunk=False)
        blocks = [blk for blk in sm.get_matching_blocks() if blk.size]
        if not blocks:
            continue
        s = lo + blocks[0].a
        e = lo + blocks[-1].a + blocks[-1].size
        sim = SequenceMatcher(None, doc[s:e], exact, autojunk=False).ratio()
        if sim < min_similarity:
            continue
        ctx = _context_score(doc, s, e, prefix, suffix)
        cand = Match(s, e, sim, "fuzzy", ctx)
        if best is None or _better(cand, best, hint_start):
            best = cand
    return best


def _better(a: Match, b: Match, hint_start: int | None) -> bool:
    sa = a.similarity * 0.8 + a.context_score * 0.2
    sb = b.similarity * 0.8 + b.context_score * 0.2
    if abs(sa - sb) > 1e-9:
        return sa > sb
    if hint_start is None:
        return False
    return abs(a.start - hint_start) < abs(b.start - hint_start)
