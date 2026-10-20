"""
Repetition profile: three numbers summarising perseveration in a transcript.

Repetition — saying the same word, phrase or sentence over and over — is one
of the clearest surface signs of disordered language, both in aphasia and in
degenerate language-model output ("I can you I I I I I ..."). This module
finds repeated units in a text and reports:

    rep_shortest_len    length (in words) of the shortest repeated unit
    rep_shortest_count  how many times that shortest unit occurs
    rep_max_count       the largest occurrence count of ANY repeated unit

Definition of "repeated" (decided after inspecting many real cases):

  * A word or SHORT phrase counts only when the repeats are CONTINUOUS —
    immediately adjacent, like "the the the" or "I can you I can you".
    (Re-using a short phrase somewhere else in a story is normal language.)
  * A SENTENCE or LONG phrase (LONG_MIN words or more) counts whether the
    repeats are adjacent or scattered anywhere in the text, because saying a
    whole sentence twice is unusual wherever it happens.

Lengths are counted in words, ignoring punctuation.
"""

import re
from collections import defaultdict

# A token is a word (letters/digits, allowing internal hyphens/apostrophes)
# or a single punctuation mark.
TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)

# Phrases at least this many words long also count when their repeats are
# scattered (not adjacent). Below this, only adjacent repeats count.
# NOTE: 6 was chosen empirically: at 5, ordinary fluent
# stories fire on legitimately re-used frames like "get ready for the ball";
# at 6 every hit we inspected was a genuine repetition.
LONG_MIN = 6

# The longest unit we search for. Must exceed the longest block a text might
# repeat, or a long multi-sentence loop gets clipped and mis-detected.
MAX_UNIT = 100

# For turning a token list back into readable text.
NO_SPACE_BEFORE = set(".,!?;:)]}%")
NO_SPACE_AFTER = set("([{$")


def smallest_period(s):
    """Smallest p such that s is built from repetitions of s[:p].

    NOTE: this is the classic KMP "failure function". For
    "PAPAPA" it returns 2 (the unit "PA"); for "hello" it returns 5 (no
    repetition, the whole string is its own unit).
    """
    n = len(s)
    fail = [0] * n
    k = 0
    for i in range(1, n):
        while k and s[i] != s[k]:
            k = fail[k - 1]
        if s[i] == s[k]:
            k += 1
        fail[i] = k
    return n - fail[-1] if n else 0


def split_glued_repeats(token):
    """Split a token like "PAPAPAPA" or "TheTheThe" into its repeated units.

    Language models sometimes glue repeats together with no spaces, so a
    plain tokenizer sees one long word and misses the repetition entirely.

    NOTE: real English contains innocent doublings —
    "papa", "mama", "byebye", "2020". We only split when the unit repeats
    three or more times, or twice with a unit of at least four characters.
    """
    n = len(token)
    if n < 3:
        return [token]
    period = smallest_period(token)
    if period == 0 or period >= n:
        return [token]
    repeats = n // period
    if not (repeats >= 3 or (repeats == 2 and period >= 4 and n % period == 0)):
        return [token]
    units = [token[i * period:(i + 1) * period] for i in range(repeats)]
    remainder = token[repeats * period:]
    if remainder:
        units.append(remainder)
    return units


def tokenize(text):
    """Split text into lowercase word / punctuation tokens."""
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    tokens = []
    for token in TOKEN_RE.findall(text):
        if WORD_RE.match(token) and len(token) >= 4:
            tokens.extend(split_glued_repeats(token))
        else:
            tokens.append(token)
    return [t.lower() for t in tokens]


def untokenize(tokens):
    """Join tokens back into readable text (used for the example string)."""
    out = []
    previous = ""
    for token in tokens:
        if not out or token in NO_SPACE_BEFORE or previous in NO_SPACE_AFTER:
            out.append(token)
        else:
            out.append(" " + token)
        previous = token
    return "".join(out).strip()


def unit_length_in_words(unit):
    """Length of a unit counting words only. A unit made purely of punctuation
    ("[ [ [") falls back to its token count so it is never reported as 0."""
    n_words = sum(1 for token in unit if WORD_RE.match(token))
    return n_words if n_words else len(unit)


def repetition_profile(text):
    """Find every repeated unit in ``text`` and summarise it.

    Returns a dictionary with rep_shortest_len, rep_shortest_count,
    rep_max_count, plus rep_shortest_str (the shortest repeated unit as
    text, handy for checking what was detected) and the full list of units.
    All counts are 0 when nothing repeats.
    """
    tokens = tokenize(text)
    n = len(tokens)
    claimed = [False] * n      # positions already explained by a repetition
    units = []                 # (length_in_words, count, unit_tokens)

    # --- Pass 1: continuous runs of any length, smallest unit first ---------
    # NOTE: we look for the SMALLEST unit that repeats at
    # each position. Searching longest-first would read "PA PA PA PA" as
    # "PAPA PAPA" (one repeat instead of three).
    i = 0
    while i < n:
        unit_len = 0
        for length in range(1, min(MAX_UNIT, (n - i) // 2) + 1):
            if tokens[i:i + length] == tokens[i + length:i + 2 * length]:
                unit_len = length
                break
        if unit_len == 0:
            i += 1
            continue
        count = 1
        while tokens[i + count * unit_len:i + (count + 1) * unit_len] == tokens[i:i + unit_len]:
            count += 1
        unit = tuple(tokens[i:i + unit_len])
        units.append((unit_length_in_words(unit), count, unit))
        for j in range(i, i + count * unit_len):
            claimed[j] = True
        i += count * unit_len

    # --- Pass 2: long phrases repeated anywhere, longest first --------------
    # NOTE: longest-first with position claiming reports only
    # MAXIMAL repeated phrases. Otherwise every sub-phrase of a repeated
    # sentence would fire too, and rep_shortest_len would always collapse to
    # LONG_MIN.
    for length in range(min(MAX_UNIT, n // 2), LONG_MIN - 1, -1):
        positions_of = defaultdict(list)
        for start in range(0, n - length + 1):
            positions_of[tuple(tokens[start:start + length])].append(start)
        for unit, starts in positions_of.items():
            if len(starts) < 2 or not any(WORD_RE.match(t) for t in unit):
                continue
            picked = []
            last_end = -1
            for start in starts:   # non-overlapping and not already claimed
                if start >= last_end and not all(claimed[start:start + length]):
                    picked.append(start)
                    last_end = start + length
            if len(picked) >= 2:
                units.append((unit_length_in_words(unit), len(picked), unit))
                for start in picked:
                    for j in range(start, start + length):
                        claimed[j] = True

    if not units:
        return {"rep_shortest_len": 0, "rep_shortest_count": 0,
                "rep_max_count": 0, "rep_shortest_str": "", "units": []}

    shortest_len = min(u[0] for u in units)
    # Among units tied for shortest, report the one that repeats most.
    shortest = max((u for u in units if u[0] == shortest_len), key=lambda u: u[1])
    return {
        "rep_shortest_len": shortest_len,
        "rep_shortest_count": shortest[1],
        "rep_max_count": max(u[1] for u in units),
        "rep_shortest_str": untokenize(shortest[2]),
        "units": sorted(units, key=lambda u: (u[0], -u[1])),
    }


def compute(text):
    """The three feature columns for one transcript."""
    profile = repetition_profile(text)
    return {column: profile[column] for column in COLUMNS}


COLUMNS = ["rep_shortest_len", "rep_shortest_count", "rep_max_count"]
