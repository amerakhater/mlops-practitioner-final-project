"""Measure *why* naive symmetric INT8 hurts an LLM, on the real model.

    uv run python scripts/quantization_analysis.py --limit-layers 4
    uv run python scripts/quantization_analysis.py --out reports/eval/qa.json

Module 4 asks for an explanation of activation outliers, attention sensitivity
and weight distributions. This script measures them on Qwen2.5-1.5B-Instruct
with this project's own corpus as the calibration set, so the report argues
from numbers taken here rather than from a paper's numbers taken elsewhere on
a different model and a different corpus.

Four things get measured, each answering one part of the question:

1. **Activation outliers.** Per-input-channel maximum absolute activation at
   every linear layer. The failure mode of per-tensor symmetric INT8 is that
   one channel's magnitude sets the scale for all of them, so the ratio
   ``max_channel / median_channel`` is the factor by which the typical channel
   is under-resolved. A handful of channels being 20-100x the median is the
   whole phenomenon.

2. **Weight distributions.** Per-output-channel weight ranges. Weights are
   far better behaved than activations -- near-Gaussian and similar across
   channels -- which is the asymmetry that makes *weight-only* quantization
   work at 4 bits while *activation* quantization needs 8 bits and per-token
   scales. It is also why this project quantizes weights only.

3. **Where the error actually lands.** Relative quantization error per module
   type. Attention projections and MLP projections do not degrade equally, and
   the q/k/v projections feed a softmax, which is why an error there moves the
   attention distribution rather than just perturbing a sum.

4. **Scheme comparison.** Measured reconstruction error for per-tensor
   symmetric INT8, per-channel symmetric INT8 and group-wise asymmetric INT4.
   The interesting result is the ordering: if group-wise INT4 beats per-tensor
   INT8 despite storing half as many bits, then "more bits is better" is the
   wrong model of quantization and *where the scales live* matters more than
   the bit width.

Runs entirely on CPU in float32. No GPU and no quantization library: the point
is the arithmetic, and doing it by hand is what makes it checkable.
"""

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from prodllm.bench.hardware import describe_hardware
from prodllm.quant import (
    quantize_dequantize,
    quantize_per_channel_symmetric,
    quantize_per_tensor_symmetric,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT = Path("reports/eval/quant_analysis.json")

# Module-name fragments grouped by role, so the error can be attributed to
# attention versus the MLP rather than to 196 individual layers.
ROLES = {
    "attn.q_proj": "attention_q",
    "attn.k_proj": "attention_k",
    "attn.v_proj": "attention_v",
    "attn.o_proj": "attention_out",
    "mlp.gate_proj": "mlp_gate",
    "mlp.up_proj": "mlp_up",
    "mlp.down_proj": "mlp_down",
}


def role_of(name: str) -> str | None:
    for fragment, role in ROLES.items():
        if name.endswith(fragment):
            return role
    return None


def quantize_group_padded(
    weight: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    """Group-wise round trip, padding an in_features that does not divide.

    ``prodllm.quant.group_quantize`` refuses a width that is not a multiple of
    the group size, because in the checkpoint path a silent pad would change
    the stored layout. Here the only use is measuring error, so padding is
    the right behaviour rather than a failure -- Qwen's MLP width of 8960 is
    divisible by 128, but this keeps the script usable on models where it is
    not.
    """
    in_features = weight.shape[1]
    remainder = in_features % group_size
    if not remainder:
        return quantize_dequantize(weight, bits, group_size)
    padded = torch.nn.functional.pad(weight, (0, group_size - remainder))
    return quantize_dequantize(padded, bits, group_size)[:, :in_features]


def relative_error(original: torch.Tensor, restored: torch.Tensor) -> float:
    """Frobenius-norm relative error, the standard reconstruction measure."""
    denominator = torch.linalg.norm(original)
    if denominator == 0:
        return 0.0
    return float(torch.linalg.norm(original - restored) / denominator)


def calibration_texts(chunks_path: Path, count: int, max_chars: int) -> list[str]:
    """Calibration text drawn from this project's own corpus.

    The outlier statistics below are a property of the *input distribution*,
    not of the weights, so they have to be measured on the text this service
    actually sees. Collecting them on web text and then serving Egyptian legal
    Arabic would describe some other deployment. The same argument applies to
    GPTQ's Hessian, which is why scripts/quantize_gptq.py calibrates from this
    same corpus.
    """
    if not chunks_path.is_file():
        raise FileNotFoundError(f"corpus chunks not found: {chunks_path}")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for chunk in chunks:
        parts = [chunk.get("text_ar", ""), chunk.get("text_en", "")]
        text = "\n".join(part for part in parts if part).strip()
        if len(text) > 200:
            texts.append(text[:max_chars])
        if len(texts) >= count:
            break
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.json")
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--limit-layers",
        type=int,
        default=0,
        help="Analyse only the first N decoder layers (0 = all 28).",
    )
    parser.add_argument(
        "--outlier-factor",
        type=float,
        default=10.0,
        help="A channel this many times the median absmax counts as an outlier.",
    )
    args = parser.parse_args()

    hardware = describe_hardware()
    print(f"hardware : {hardware.summary()}")
    print(f"model    : {args.model}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # torch_dtype, not dtype: transformers 4.x spells it the former way.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    print(f"params   : {sum(p.numel() for p in model.parameters()):,}")

    targets: dict[str, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or role_of(name) is None:
            continue
        if args.limit_layers:
            # names look like model.layers.<n>.self_attn.q_proj
            pieces = name.split(".")
            try:
                layer_index = int(pieces[pieces.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            if layer_index >= args.limit_layers:
                continue
        targets[name] = module
    print(f"linears  : {len(targets)} analysed")

    # Per-input-channel running absmax. Hooks rather than a rewritten forward:
    # this has to observe the real activations the real model produces.
    absmax: dict[str, torch.Tensor] = {}

    def make_hook(name: str) -> Callable[..., None]:
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            activation = inputs[0].detach()
            flat = activation.reshape(-1, activation.shape[-1]).abs().amax(dim=0)
            previous = absmax.get(name)
            absmax[name] = flat if previous is None else torch.maximum(previous, flat)

        return hook

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in targets.items()
    ]

    texts = calibration_texts(args.chunks, args.samples, args.max_chars)
    print(f"calibration: {len(texts)} passages from {args.chunks}")
    with torch.inference_mode():
        for index, text in enumerate(texts, start=1):
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
            )
            model(**batch)
            print(f"  calibrated {index}/{len(texts)}", flush=True)
    for handle in handles:
        handle.remove()

    print("\nanalysing...")
    rows: list[dict] = []
    for name, module in targets.items():
        channels = absmax.get(name)
        if channels is None:
            continue
        weight = module.weight.detach().float()

        median = float(channels.median())
        maximum = float(channels.max())
        ratio = maximum / median if median else 0.0
        outliers = int((channels > args.outlier_factor * median).sum())

        weight_per_channel_range = (
            weight.abs().amax(dim=1) / weight.abs().amax(dim=1).median()
        )

        rows.append(
            {
                "name": name,
                "role": role_of(name),
                "shape": list(weight.shape),
                "act_median_absmax": median,
                "act_max_absmax": maximum,
                "act_outlier_ratio": ratio,
                "act_outlier_channels": outliers,
                "act_channels": int(channels.numel()),
                "weight_channel_ratio": float(weight_per_channel_range.max()),
                "err_int8_per_tensor": relative_error(
                    weight, quantize_per_tensor_symmetric(weight, 8)
                ),
                "err_int8_per_channel": relative_error(
                    weight, quantize_per_channel_symmetric(weight, 8)
                ),
                "err_int4_group": relative_error(
                    weight, quantize_group_padded(weight, 4, args.group_size)
                ),
            }
        )
        print(
            f"  {name:<44s} outlier x{ratio:7.1f} "
            f"({outliers:3d} ch)  int8/tensor {rows[-1]['err_int8_per_tensor']:.4f} "
            f"int8/chan {rows[-1]['err_int8_per_channel']:.4f} "
            f"int4/grp {rows[-1]['err_int4_group']:.4f}",
            flush=True,
        )

    def mean_of(key: str, role: str | None = None) -> float:
        values = [r[key] for r in rows if role is None or r["role"] == role]
        return statistics.fmean(values) if values else 0.0

    by_role = {
        role: {
            "modules": sum(1 for r in rows if r["role"] == role),
            "act_outlier_ratio": mean_of("act_outlier_ratio", role),
            "err_int8_per_tensor": mean_of("err_int8_per_tensor", role),
            "err_int8_per_channel": mean_of("err_int8_per_channel", role),
            "err_int4_group": mean_of("err_int4_group", role),
        }
        for role in sorted({r["role"] for r in rows})
    }

    summary = {
        "schema_version": 1,
        "model": args.model,
        "group_size": args.group_size,
        "outlier_factor": args.outlier_factor,
        "calibration_passages": len(texts),
        "calibration_source": str(args.chunks),
        "max_seq_length": args.max_seq_length,
        "modules_analysed": len(rows),
        "overall": {
            "act_outlier_ratio": mean_of("act_outlier_ratio"),
            "weight_channel_ratio": mean_of("weight_channel_ratio"),
            "err_int8_per_tensor": mean_of("err_int8_per_tensor"),
            "err_int8_per_channel": mean_of("err_int8_per_channel"),
            "err_int4_group": mean_of("err_int4_group"),
        },
        "by_role": by_role,
        "modules": rows,
        "hardware": hardware.as_dict(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print("\n--- activation outliers (the reason per-tensor INT8 fails) ---")
    print(f"  mean max/median channel absmax : {overall['act_outlier_ratio']:.1f}x")
    print(f"  mean max/median weight channel : {overall['weight_channel_ratio']:.1f}x")
    print("\n--- mean weight reconstruction error by scheme ---")
    for key, label in (
        ("err_int8_per_tensor", "INT8 per-tensor symmetric"),
        ("err_int8_per_channel", "INT8 per-channel symmetric"),
        ("err_int4_group", f"INT4 group-wise g={args.group_size}"),
    ):
        print(f"  {label:<42s} {summary['overall'][key]:.5f}")
    print("\n--- by role ---")
    print(
        f"  {'role':<16s} {'n':>4s} {'outlier':>9s} {'int8/ten':>9s} "
        f"{'int8/chan':>10s} {'int4/grp':>9s}"
    )
    for role, stats in by_role.items():
        print(
            f"  {role:<16s} {stats['modules']:>4d} "
            f"{stats['act_outlier_ratio']:>8.1f}x {stats['err_int8_per_tensor']:>9.5f} "
            f"{stats['err_int8_per_channel']:>10.5f} {stats['err_int4_group']:>9.5f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
