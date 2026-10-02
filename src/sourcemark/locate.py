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
import unicodedata

from .textnorm import normalize_newlines, squash

DEFAULT_CONTEXT = 32
DEFAULT_K = 8
DEFAULT_MIN_SIMILARITY = 0.72
# A fuzzy hit is only trusted when it is very similar, or moderately similar AND its
# surroundings agree. Code is full of near-identical lines; similarity alone over-matches.
STRONG_SIMILARITY = 0.90
DISTINCTIVE_CHARS = 24  # an exact, unique quote at least this long stands on its own
# Look-alike code shares structure: neighbouring `if y<0:` vs `if x is None:` still scores ~0.56.
# A genuinely shifted line keeps its real neighbours (~1.0), so a short copy must clear 0.8.
EXACT_CONTEXT_MIN = 0.8
MIN_CONTEXT_FOR_FUZZY = 0.55


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


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
    unique_when_marked: bool = True,
) -> Match | None:
    """Locate ``exact`` in ``doc``. Returns the best match or None.

    ``unique_when_marked=False`` says the quote already occurred more than once when it was
    cited: a single copy left now is not evidence that it is the cited one.
    """
    doc = _nfc(normalize_newlines(doc))
    exact = _nfc(normalize_newlines(exact))
    prefix, suffix = _nfc(prefix), _nfc(suffix)
    if not exact.strip():
        return None

    # 1. Recorded position still holds the same text.
    if hint_start is not None and doc[hint_start : hint_start + len(exact)] == exact:
        ctx = _context_score(doc, hint_start, hint_start + len(exact), prefix, suffix)
        # A decoy copy can land on the old offset; only trust it if the context agrees
        # or if it is the only copy.
        if ctx >= 0.5 or doc.count(exact) == 1:
            return Match(hint_start, hint_start + len(exact), 1.0, "position", ctx)

    # 2. Exact text elsewhere; choose by context, then by distance to the hint.
    whole = _whole_lines(prefix, suffix)
    occ = [o for o in _all_occurrences(doc, exact) if not whole or _on_line_bounds(doc, o, o + len(exact))]
    if occ:
        def rank(pos: int) -> tuple[float, float]:
            ctx = _context_score(doc, pos, pos + len(exact), prefix, suffix)
            dist = abs(pos - hint_start) if hint_start is not None else 0
            return (ctx, -dist)

        best = max(occ, key=rank)
        ctx = _context_score(doc, best, best + len(exact), prefix, suffix)
        worst = _context_score(doc, best, best + len(exact), prefix, suffix, worst=True)
        distinctive = len(occ) == 1 and len(exact.strip()) >= DISTINCTIVE_CHARS and unique_when_marked
        # A short or repeated line found away from its context is usually a different
        # occurrence (e.g. `return None` in another function), not the cited one.
        if not (prefix or suffix) or worst >= EXACT_CONTEXT_MIN or distinctive:
            return Match(best, best + len(exact), 1.0, "exact", ctx)
        return None

    # 2b. Same words, only layout changed: the citation still says the same thing. The same
    # gates apply: whole lines stay whole lines, and a non-distinctive hit needs its context.
    loose = _loose_find(doc, exact, hint_start)
    if loose is not None:
        s_, e_, n_hits = loose
        if not whole or _on_line_bounds(doc, s_, e_):
            worst = _context_score(doc, s_, e_, prefix, suffix, worst=True)
            distinctive = n_hits == 1 and len(exact.strip()) >= DISTINCTIVE_CHARS and unique_when_marked
            if not (prefix or suffix) or worst >= EXACT_CONTEXT_MIN or distinctive:
                return Match(s_, e_, 1.0, "loose", _context_score(doc, s_, e_, prefix, suffix))

    # 3. Fuzzy: k-gram offset voting, then score the winning windows. A fuzzy hit is
    # trusted only when very similar, or when BOTH sides of its context agree.
    if not fuzzy or len(exact) > max_fuzzy_len:
        return None
    m = _fuzzy(doc, exact, prefix, suffix, hint_start, k, min_similarity)
    if m is None:
        return None
    if _is_a_neighbour(doc, m, exact, prefix, suffix):
        return None
    if m.similarity >= strong_similarity:
        return m
    if (prefix or suffix) and _context_score(doc, m.start, m.end, prefix, suffix, worst=True) >= min_context:
        return m
    return None


def _is_a_neighbour(doc: str, m: Match, exact: str, prefix: str, suffix: str) -> bool:
    """A deleted line's neighbour slides into its place and often looks alike
    (``json.loads(blob)`` / ``json.loads(data)``). The match IS that neighbour when it
    resembles the recorded neighbour at least as much as the quote, and the line actually
    next to it is no longer the recorded neighbour (an in-place edit keeps its neighbours)."""
    if "\n" in exact.strip():
        return False
    g = doc[m.start : m.end].strip()
    line_s = doc.rfind("\n", 0, m.start) + 1
    line_e = doc.find("\n", m.end)
    line_e = len(doc) if line_e < 0 else line_e
    after = doc[line_e + 1 :].split("\n", 1)[0].strip() if line_e < len(doc) else ""
    before = doc[: max(0, line_s - 1)].rsplit("\n", 1)[-1].strip() if line_s > 0 else ""
    sides = []
    if suffix.startswith("\n"):
        rest = suffix[1:]
        sides.append((rest.split("\n")[0].strip(), "\n" in rest, "next", after))
    if prefix.endswith("\n"):
        rest = prefix[:-1]
        sides.append((rest.split("\n")[-1].strip(), "\n" in rest, "prev", before))
    for n, complete, side, actual in sides:
        if len(n) < 8:
            continue
        # A neighbour cut at the context edge is the START of the next line / END of the previous one.
        def clip(t: str) -> str:
            return t if complete else (t[: len(n)] if side == "next" else t[-len(n) :])

        looks_like_neighbour = SequenceMatcher(None, clip(g), n, autojunk=False).ratio() >= m.similarity
        neighbour_still_there = SequenceMatcher(None, clip(actual), n, autojunk=False).ratio() >= 0.9
        if looks_like_neighbour and not neighbour_still_there:
            return True
    return False


def _whole_lines(prefix: str, suffix: str) -> bool:
    """The cited text was whole lines (it started and ended at line boundaries)."""
    if not prefix and not suffix:
        return False  # no context recorded: nothing says where the quote began or ended
    return (not prefix or prefix.endswith("\n")) and (not suffix or suffix.startswith("\n"))


def _on_line_bounds(doc: str, start: int, end: int) -> bool:
    """[start, end) begins and ends a line, ignoring indentation and trailing spaces."""
    left = doc[doc.rfind("\n", 0, start) + 1 : start]
    nl = doc.find("\n", end)
    right = doc[end : len(doc) if nl < 0 else nl]
    return not left.strip() and not right.strip()


def _loose_find(doc: str, exact: str, hint_start: int | None) -> tuple[int, int, int] | None:
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
    return start, end, len(hits)


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
    line_mode = len(exact) > LINE_MODE_CHARS
    for b, _count in votes.most_common(3 if line_mode else 5):
        approx = max(0, b * bucket)
        lo = max(0, approx - slack)
        hi = min(len(doc), approx + len(exact) + 2 * slack + bucket)
        if line_mode:
            span = _line_span(doc, lo, hi, exact)
            if span is None:
                continue
            s, e, sim = span
        else:
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


# Above this length, character-level SequenceMatcher is quadratic in practice (a 12k-char
# block took 21 s); match whole lines instead, which is what a long citation is made of.
LINE_MODE_CHARS = 1500


def _line_span(doc: str, lo: int, hi: int, exact: str) -> tuple[int, int, float] | None:
    # Widen to whole lines so line items compare cleanly.
    lo = doc.rfind("\n", 0, lo) + 1
    nl = doc.find("\n", hi)
    hi = len(doc) if nl < 0 else nl
    wl = doc[lo:hi].split("\n")
    el = exact.split("\n")
    offs, pos = [], lo
    for ln in wl:
        offs.append(pos)
        pos += len(ln) + 1
    blocks = [blk for blk in SequenceMatcher(None, wl, el, autojunk=False).get_matching_blocks() if blk.size]
    if not blocks:
        return None
    first, last = blocks[0], blocks[-1]
    # Align to the quote's own first and last line: an edited first line has no matching
    # block, but it is still part of the cited span.
    a0 = max(0, first.a - first.b)
    a1 = min(len(wl) - 1, last.a + last.size - 1 + (len(el) - (last.b + last.size)))
    s = offs[a0]
    e = offs[a1] + len(wl[a1])
    sim = SequenceMatcher(None, doc[s:e].split("\n"), el, autojunk=False).ratio()
    return s, e, sim


def _better(a: Match, b: Match, hint_start: int | None) -> bool:
    sa = a.similarity * 0.8 + a.context_score * 0.2
    sb = b.similarity * 0.8 + b.context_score * 0.2
    if abs(sa - sb) > 1e-9:
        return sa > sb
    if hint_start is None:
        return False
    return abs(a.start - hint_start) < abs(b.start - hint_start)
