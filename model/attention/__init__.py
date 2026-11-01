"""The two fixed attention backends (policy: ``docs/decisions.md`` D4).

``model/layers.py`` imports the constants directly and dispatches on ``T``:
T > 1 -> ``PREFILL_BACKEND`` (sdpa), T == 1 -> ``DECODE_BACKEND``
(hand_baddbmm). There is no name, registry or config knob to pick another
pair, so a silent swap is impossible by construction.
"""

from model.attention.base import AttentionBackend
from model.attention.hand_baddbmm import BACKEND as DECODE_BACKEND
from model.attention.sdpa import BACKEND as PREFILL_BACKEND

__all__ = ["AttentionBackend", "PREFILL_BACKEND", "DECODE_BACKEND", "provenance"]


def provenance() -> dict[str, str]:
    """Constant manifest/summary payload; recorded so a future swap is visible."""
    return {"prefill": PREFILL_BACKEND.name, "decode": DECODE_BACKEND.name}
