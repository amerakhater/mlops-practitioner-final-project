"""Compute the memory a transformer needs at FP16, INT8 and 4-bit.

    uv run python scripts/vram_table.py
    uv run python scripts/vram_table.py --config artifacts/qwen2.5-1.5b/config.json
    uv run python scripts/vram_table.py --model llama-3.1-70b --vram-budget 80

Derives the parameter count from the architecture and then the memory from the
parameter count, rather than quoting a number from a model card. Two reasons
that matters here:

* A model card's "70B" is rounded, and the question "does a 70B model fit on an
  80 GB GPU" is decided in the last few gigabytes. Rounding the parameter count
  to one significant figure throws away exactly the precision the question
  turns on.
* **Weights are not the whole story and the KV cache is where the 4-bit saving
  actually goes.** Quantizing weights to 4 bits frees tens of gigabytes, and on
  a single-GPU deployment essentially all of it is re-spent on context length,
  because the KV cache is usually left at FP16 and grows linearly with tokens.
  A table of weight sizes alone makes 4-bit look like pure headroom; it is a
  reallocation.

Every number this prints is reproducible from the architecture constants below,
so a reader can check the arithmetic instead of trusting it.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Bytes per stored parameter.
#
# The 4-bit row is 4.25 rather than 4.0 and the extra quarter-bit is not a
# rounding allowance: group-wise schemes store a scale (and, when asymmetric, a
# zero point) for every group of weights, and those live at higher precision.
# At the standard group_size of 128 that overhead is 16 bits of scale plus 16
# bits of zero point per 128 weights = 0.25 bits per weight. Ignoring it
# understates a 70B checkpoint by about two gigabytes.
BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": (4.0 + 32.0 / 128.0) / 8.0,
}

# Runtime memory that is not weights and not KV cache: the CUDA context, the
# allocator's fragmentation, cuBLAS/cuDNN workspaces and the activations of the
# layer currently executing. Measured in the 1-2 GiB range for a single-stream
# server on one device; 1.5 is a working figure, stated so it can be argued
# with rather than buried.
RUNTIME_OVERHEAD_GIB = 1.5


@dataclass(frozen=True)
class Architecture:
    """The dimensions a decoder-only transformer's parameter count follows from."""

    name: str
    layers: int
    hidden: int
    intermediate: int
    attention_heads: int
    kv_heads: int
    vocab: int
    head_dim: int = 0
    tie_embeddings: bool = False

    @property
    def effective_head_dim(self) -> int:
        return self.head_dim or self.hidden // self.attention_heads

    @property
    def parameters(self) -> int:
        """Total parameters, summed from the actual projection shapes.

        Grouped-query attention is why ``k_proj`` and ``v_proj`` are not the
        same size as ``q_proj``: with 64 query heads and 8 key/value heads the
        K and V projections are an eighth of the width, which is also what
        shrinks the KV cache by 8x and is the single reason a 70B model has any
        hope of a long context on one card.
        """
        head = self.effective_head_dim
        q_out = self.attention_heads * head
        kv_out = self.kv_heads * head

        attention = (
            self.hidden * q_out  # q_proj
            + self.hidden * kv_out  # k_proj
            + self.hidden * kv_out  # v_proj
            + q_out * self.hidden  # o_proj
        )
        # SwiGLU: gate and up are both hidden -> intermediate, down comes back.
        mlp = 3 * self.hidden * self.intermediate
        # Two RMSNorms per layer, one weight vector each.
        norms = 2 * self.hidden

        per_layer = attention + mlp + norms
        embeddings = self.vocab * self.hidden
        head_weights = 0 if self.tie_embeddings else self.vocab * self.hidden
        return self.layers * per_layer + embeddings + head_weights + self.hidden

    def weight_gib(self, precision: str) -> float:
        return self.parameters * BYTES[precision] / 1024**3

    def kv_cache_gib(self, tokens: int, precision: str = "fp16") -> float:
        """KV cache for ``tokens`` tokens, summed over layers.

        Two tensors (K and V) per layer, each ``kv_heads x head_dim`` wide per
        token. Independent of the weight precision: quantizing weights to 4
        bits does nothing to the cache unless the cache is quantized too.
        """
        per_token_bytes = (
            2 * self.layers * self.kv_heads * self.effective_head_dim * BYTES[precision]
        )
        return per_token_bytes * tokens / 1024**3

    def kv_bytes_per_token(self, precision: str = "fp16") -> float:
        return (
            2 * self.layers * self.kv_heads * self.effective_head_dim * BYTES[precision]
        )


# Published architectures, so the 70B question can be answered without a
# 141 GB download. Every field is from the model's own config.json.
KNOWN = {
    "qwen2.5-1.5b": Architecture(
        name="Qwen2.5-1.5B-Instruct",
        layers=28,
        hidden=1536,
        intermediate=8960,
        attention_heads=12,
        kv_heads=2,
        vocab=151936,
        tie_embeddings=True,
    ),
    "qwen2.5-0.5b": Architecture(
        name="Qwen2.5-0.5B-Instruct",
        layers=24,
        hidden=896,
        intermediate=4864,
        attention_heads=14,
        kv_heads=2,
        vocab=151936,
        tie_embeddings=True,
    ),
    "llama-3.1-70b": Architecture(
        name="Llama-3.1-70B",
        layers=80,
        hidden=8192,
        intermediate=28672,
        attention_heads=64,
        kv_heads=8,
        vocab=128256,
        tie_embeddings=False,
    ),
}


def from_config(path: Path) -> Architecture:
    """Read an architecture out of a Hugging Face config.json."""
    config = json.loads(path.read_text(encoding="utf-8"))
    return Architecture(
        name=config.get("_name_or_path") or path.parent.name,
        layers=int(config["num_hidden_layers"]),
        hidden=int(config["hidden_size"]),
        intermediate=int(config["intermediate_size"]),
        attention_heads=int(config["num_attention_heads"]),
        kv_heads=int(config.get("num_key_value_heads", config["num_attention_heads"])),
        vocab=int(config["vocab_size"]),
        head_dim=int(config.get("head_dim", 0)),
        tie_embeddings=bool(config.get("tie_word_embeddings", False)),
    )


def print_table(arch: Architecture, context: int, budget_gib: float | None) -> None:
    print(f"\n{arch.name}")
    print(
        f"  layers {arch.layers}, hidden {arch.hidden}, "
        f"intermediate {arch.intermediate}"
    )
    print(
        f"  attention heads {arch.attention_heads}, kv heads {arch.kv_heads}, "
        f"head_dim {arch.effective_head_dim}"
    )
    print(f"  vocab {arch.vocab:,}, tied embeddings {arch.tie_embeddings}")
    print(f"  parameters: {arch.parameters:,} ({arch.parameters / 1e9:.2f} B)")

    kv = arch.kv_cache_gib(context)
    print(
        f"\n  KV cache at {context:,} tokens, FP16: {kv:.2f} GiB "
        f"({arch.kv_bytes_per_token() / 1024:.1f} KiB/token)"
    )

    header = (
        f"\n  {'precision':<10} {'bits/w':>7} {'weights':>10} {'+KV':>9} "
        f"{'+runtime':>10} {'total':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 3))
    for precision in ("fp32", "fp16", "int8", "int4"):
        weights = arch.weight_gib(precision)
        total = weights + kv + RUNTIME_OVERHEAD_GIB
        fits = ""
        if budget_gib is not None:
            fits = "  FITS" if total <= budget_gib else "  does not fit"
        print(
            f"  {precision:<10} {BYTES[precision] * 8:>7.2f} "
            f"{weights:>9.2f}G {kv:>8.2f}G {RUNTIME_OVERHEAD_GIB:>9.2f}G "
            f"{total:>8.2f}G{fits}"
        )

    if budget_gib is None:
        return

    print(f"\n  against a {budget_gib:.0f} GiB budget:")
    for precision in ("fp16", "int8", "int4"):
        weights = arch.weight_gib(precision)
        spare = budget_gib - weights - RUNTIME_OVERHEAD_GIB
        if spare <= 0:
            print(
                f"    {precision:<5}: weights alone leave "
                f"{spare:.1f} GiB -- will not load"
            )
            continue
        tokens = int(spare * 1024**3 / arch.kv_bytes_per_token())
        print(
            f"    {precision:<5}: {weights:.1f} GiB weights, {spare:.1f} GiB spare "
            f"-> {tokens:,} tokens of FP16 KV cache"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        choices=sorted(KNOWN),
        help="A known architecture. Repeatable. Defaults to all of them.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="A local config.json to read instead."
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help="Context length the KV cache is sized for.",
    )
    parser.add_argument(
        "--vram-budget",
        type=float,
        default=None,
        help="GiB of VRAM to test against, e.g. 80 for an H100/A100-80GB.",
    )
    args = parser.parse_args()

    if args.config:
        if not args.config.is_file():
            print(f"ERROR config not found: {args.config}", file=sys.stderr)
            return 1
        print_table(from_config(args.config), args.context, args.vram_budget)
        return 0

    for key in args.model or sorted(KNOWN):
        print_table(KNOWN[key], args.context, args.vram_budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
