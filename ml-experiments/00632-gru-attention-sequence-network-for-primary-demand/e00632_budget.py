# ruff: noqa: E501, D102, D103
"""Experiment 632: token counts and memory estimates of the GRU plus self-attention network.

Prints, for each token option, how many tokens the joint self-attention sees, the size of one
attention score matrix, and a rough peak-memory estimate of one training batch. The estimate is
deliberately pessimistic: it assumes the attention scores are materialised in float32 (the plain
`softmax(QK^T)` form). PyTorch's fused `scaled_dot_product_attention` avoids that, so the real peak
is lower, and the first timing run measures it.

Usage: python e00632_budget.py
"""

from dataclasses import dataclass
from typing import Final

HALF_HOURS_PER_DAY: Final[int] = 48
BYTES_F32: Final[int] = 4


@dataclass(frozen=True)
class TokenOption:
    """One way of cutting the history and the forecast horizon into tokens."""

    name: str
    history_days: int
    forecast_days: int
    half_hours_per_token: int  # 1 = half-hourly tokens, 2 = hourly, 4 = patches of 4 half-hours

    @property
    def history_tokens(self) -> int:
        return self.history_days * HALF_HOURS_PER_DAY // self.half_hours_per_token

    @property
    def future_tokens(self) -> int:
        return self.forecast_days * HALF_HOURS_PER_DAY // self.half_hours_per_token

    @property
    def tokens(self) -> int:
        return self.history_tokens + self.future_tokens


OPTIONS: Final[tuple[TokenOption, ...]] = (
    TokenOption("A: 7 d in, 7 d out, half-hourly", 7, 7, 1),
    TokenOption("B: 7 d in, 2 d out, half-hourly", 7, 2, 1),
    TokenOption("C: 7 d in, 7 d out, hourly", 7, 7, 2),
    TokenOption("D: 7 d in, 7 d out, patches of 4", 7, 7, 4),
)


def attention_bytes(*, option: TokenOption, batch: int, heads: int, layers: int) -> int:
    """Return the bytes of all attention score matrices of one batch, kept for the backward pass."""
    return batch * heads * option.tokens**2 * BYTES_F32 * layers


def activation_bytes(*, option: TokenOption, batch: int, width: int, layers: int) -> int:
    """Return a rough byte count of the other activations kept for the backward pass.

    Per token and layer: query, key, value, attention output, two layer norms, and a feed-forward
    block four times the model width. The two GRUs and the three encoders add about the same again.
    """
    per_token_per_layer = (4 + 2 + 4 + 4) * width
    gru_and_encoders = 2 * option.tokens * 8 * width
    return (
        batch * option.tokens * per_token_per_layer * layers * BYTES_F32
        + batch * gru_and_encoders * BYTES_F32
    )


def main() -> None:
    width, heads, layers, batch = 64, 4, 3, 32
    print(f"width {width}, {heads} heads, {layers} attention layers, batch {batch}, float32")
    print(
        f"{'option':38s} {'in':>5s} {'out':>5s} {'scores/layer MB':>16s} {'peak GB (plain)':>16s}"
    )
    for o in OPTIONS:
        per_layer = batch * heads * o.tokens**2 * BYTES_F32 / 1e6
        peak = (
            attention_bytes(option=o, batch=batch, heads=heads, layers=layers)
            + activation_bytes(option=o, batch=batch, width=width, layers=layers)
        ) / 1e9
        print(
            f"{o.name:38s} {o.history_tokens:5d} {o.future_tokens:5d} {per_layer:16.1f} {peak:16.2f}"
        )


if __name__ == "__main__":
    main()
