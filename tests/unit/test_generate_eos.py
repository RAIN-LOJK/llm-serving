"""EOS divergence ledger, tested without training a model.

The fake model emits a deterministic token per row and per step. It removes a row from
its own active list exactly when it emits EOS, because ``generate`` culls the same row
from buffer/attention_mask/KVcache before the next forward.
"""

from __future__ import annotations

import torch

from llm_utils.inference import generate

EOS = 100


class _FakeConfig:
    n_layer = 1
    max_position_embeddings = 64
    vocab_size = 128


class _FakeModel:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.active: list[dict[str, int]] = []
        self.step = -1
        self.started = False

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, past_keys_values=None):
        batch = input_ids.shape[0]
        if not self.started:
            self.started = True
            starts = [int(x) for x in input_ids[:, -1].tolist()]
            self.active = [{"start": s, "eos_at": 3 + (s % 3)} for s in starts]
            self.step = 0
        else:
            self.step += 1

        next_tokens = [self._token(row, self.step) for row in self.active]
        # generate() will cull rows whose sampled token is EOS; mirror that here.
        self.active = [row for row, tok in zip(self.active, next_tokens) if tok != EOS]

        logits = torch.full((batch, input_ids.shape[1], self.config.vocab_size), -10.0)
        for i, tok in enumerate(next_tokens):
            logits[i, -1, tok] = 10.0
        return logits, None

    @staticmethod
    def _token(row: dict[str, int], step: int) -> int:
        if step >= row["eos_at"]:
            return EOS
        return row["start"] + step + 1


def _batch():
    seqs = [torch.tensor([1, 2, 3, 10]), torch.tensor([1, 2, 20]), torch.tensor([1, 30])]
    width = max(len(s) for s in seqs)
    tokens = torch.zeros((len(seqs), width), dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for i, seq in enumerate(seqs):
        tokens[i, width - len(seq) :] = seq
        mask[i, width - len(seq) :] = 1
    return seqs, tokens, mask


def test_eos_divergence_matches_single():
    seqs, tokens, mask = _batch()
    got = generate(_FakeModel(), tokens, 20, temperature=0.0, eos_token=EOS, attention_mask=mask)
    ref = [
        generate(_FakeModel(), seq.unsqueeze(0), 20, temperature=0.0, eos_token=EOS)[0]
        for seq in seqs
    ]

    assert [len(x) for x in got] == [len(x) for x in ref] == [5, 6, 4]
    for i, (a, b) in enumerate(zip(got, ref)):
        assert torch.equal(a, b), f"row {i}: batch={a.tolist()} single={b.tolist()}"
        assert int(a[-1]) == EOS


def test_all_rows_truncated_to_sampled_length():
    """The preallocated buffer must not leak unsampled zeros into outputs."""
    seqs, tokens, mask = _batch()
    got = generate(_FakeModel(), tokens, 20, temperature=0.0, eos_token=EOS, attention_mask=mask)
    assert [len(x) for x in got] == [5, 6, 4]
    for out in got:
        assert int(out[-1]) == EOS
