# How it works

## A mark

| Part | Contents | Why |
|---|---|---|
| Quote selector | exact text + 32 characters before and after | finds the text again; context disambiguates duplicates |
| Position | character offsets, 1-based start and end lines | the cheapest check; also ranks candidates |
| Fingerprints | sha256 of the normalized quote, a whitespace-blind variant, and the whole document | verifies without storing text (secret-bearing quotes keep only these) |
| Source | path; git repository, commit, path in repo, blob | follows renames; pins the revision that was cited |
| Id | content-addressed: same quote, source and revision → same id | stable, deduplicated |

Normalization (for fingerprints): Unicode NFC, LF newlines, trailing spaces removed,
runs of spaces collapsed. The raw text is what gets stored.

## Re-finding text in a document

1. **Recorded position** still holds the exact text, and its context agrees (or it is the only copy).
2. **Exact text elsewhere**, ranked by surrounding context, then distance from the old position.
3. **Same words, different layout** (reindented, reflowed): whitespace-insensitive search.
4. **Fuzzy**: every shared 8-character gram votes for where the quote would start; the top windows
   are scored with a sequence matcher. A fuzzy hit is trusted only at ≥ 0.90 similarity, or at
   ≥ 0.72 when **both** the text before and the text after agree (≥ 0.55). One-sided agreement is
   exactly what a deleted line looks like: its neighbours close up around a similar line.

## Resolving a mark

The original path first (an intact citation never pays for git), then paths git reports as renames
or copies since the cited commit, then files under the search roots that contain a distinctive line
of the quote (via ripgrep when present). When the original only yields an edited match, an exact copy
elsewhere wins: cut-and-paste is a move, not an edit. Status is line-based.

## Checking an agent's citations

Evidence comes from the session's own tool calls. Line-level verdicts require line-level evidence;
files merely named in a command, or printed together with other output, count only at file level.
Each inline-code quote belongs to the nearest citation in its sentence, and only code *expressions*
are checked as quotes (identifiers are references, not quotations). Elided quotes (`foo(...)`) are
checked fragment by fragment.

## The ledger

SQLite in WAL mode. Each event stores `sha256(prev_hash | kind | session | payload)`; events are
appended inside `BEGIN IMMEDIATE` transactions so concurrent hook processes keep one chain.
`verify-ledger` recomputes it.
