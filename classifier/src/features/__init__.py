"""
The four feature groups. Each module exposes ``COLUMNS`` (the names of the
columns it produces) and a way to compute them from transcript text:

    narrative.compute(text)      -> dict      (no models needed)
    repetition.compute(text)     -> dict      (no models needed)
    proxies.compute(text)        -> dict      (no models needed)
    semsimilar: closeness to each group's transcripts (small embedding model)
"""

from . import narrative, repetition, proxies  # noqa: F401
