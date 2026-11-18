"""GPTQ 4-bit quantization of the generator, measured before and after.

Run it with the isolated quantization interpreter, not with `uv run`:

    .caches/venv-quant/Scripts/python.exe scripts/quantize_gptq.py --limit 8
    .caches/venv-quant/Scripts/python.exe scripts/quantize_gptq.py

`requirements-quant.txt` explains how that interpreter is built and why it has
to exist: GPTQModel builds from source on Windows and its `setup.py` reads the
*already installed* torch version, so it cannot be resolved in the same pass as
the project's own dependencies. Keeping it in a separate interpreter is what
stops an offline optimization job from being able to perturb the locked
environment that the service and CI are tested against. Nothing under
`src/prodllm` imports anything from here; the outputs are a checkpoint and a
JSON record.

**Why GPTQ rather than AWQ.** The module permits either. AWQ was tried first
and abandoned on tooling grounds, which are recorded in reports/module-4.md:
AutoAWQ was archived in May 2025, its successor path runs through vLLM (Linux
and CUDA only), and `llm-compressor` requires `transformers>=5` against this
project's `<5` pin. GPTQModel runs on CPU on Windows with a pure-torch backend,
so GPTQ is the 4-bit method that can be executed and checked here rather than
described.

**What GPTQ does that round-to-nearest does not.** Naive quantization rounds
each weight independently and accepts whatever output error follows. GPTQ
quantizes a layer's weights column by column and, after fixing each column,
**updates the columns it has not reached yet to compensate for the error it
just introduced**, using the inverse Hessian of the layer's inputs -- estimated
from calibration data -- to decide how. The effect is that the layer's *output*
on calibration-like inputs is preserved far better than its weights are. Nobody
consumes the weights.

**Calibration and evaluation text are disjoint and seeded.** Calibrating and
then measuring perplexity on the same passages would report how well the
quantizer memorised its own calibration set. The split is recorded in the
output JSON so it can be reproduced.

**Calibration comes from this project's corpus.** The Hessian is an estimate of
the layer's input covariance, so it is a property of the input distribution.
Calibrating on web text and then serving Egyptian legal Arabic would estimate
it from the wrong distribution.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
import torch._dynamo

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT = Path("artifacts/qwen2.5-1.5b-gptq-4bit")
DEFAULT_RESULTS = Path("reports/eval/gptq_quantization.json")
DEFAULT_CHUNKS = Path("data/processed/chunks.json")

# Loaded by file path rather than imported: this interpreter has no prodllm
# installed, and the hardware recorder was written to depend on nothing but the
# standard library, psutil and nvidia-smi precisely so it could be reused from
# here. One hardware block format across every Module 4 measurement is the
# point of having it at all.
_HARDWARE_PATH = Path(__file__).resolve().parents[1] / "src/prodllm/bench/hardware.py"


def load_hardware_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prodllm_hardware", _HARDWARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_local_snapshot(model_id: str) -> str:
    """Return a local snapshot directory for ``model_id`` if one is cached.

    Windows-specific necessity rather than an optimization. The Hugging Face
    cache materialises a snapshot as symlinks into a blob store, and creating a
    symlink on Windows needs either Developer Mode or an elevated process --
    neither of which holds here. Downloads survive it by silently falling back
    to copying, but a *later* fetch of a file the first download skipped (the
    model's LICENSE, in this case) raises ``OSError: [WinError 1314] A required
    privilege is not held by the client`` and takes the whole job down.

    Passing the snapshot directory as a plain path skips hub resolution
    altogether, so nothing is fetched and nothing is symlinked. The repo id is
    still what gets recorded in the results, because that is the reproducible
    identifier; this only changes where the weights are read from.
    """
    if Path(model_id).is_dir():
        return model_id
    cache = os.environ.get("HF_HOME")
    if not cache:
        return model_id
    flattened = "models--" + model_id.replace("/", "--")
    snapshots = Path(cache) / "hub" / flattened / "snapshots"
    if not snapshots.is_dir():
        return model_id
    for candidate in sorted(snapshots.iterdir()):
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return str(candidate)
    return model_id


def corpus_passages(path: Path, min_chars: int) -> list[str]:
    """Bilingual article text, one string per article chunk."""
    chunks = json.loads(path.read_text(encoding="utf-8"))
    passages = []
    for chunk in chunks:
        parts = [chunk.get("text_ar", ""), chunk.get("text_en", "")]
        text = "\n".join(part for part in parts if part).strip()
        if len(text) >= min_chars:
            passages.append(text)
    return passages


def split_passages(
    passages: list[str], calibration: int, evaluation: int, seed: int
) -> tuple[list[str], list[str]]:
    """A seeded, disjoint calibration/evaluation split."""
    shuffled = list(passages)
    random.Random(seed).shuffle(shuffled)
    return (
        shuffled[:calibration],
        shuffled[calibration : calibration + evaluation],
    )


def perplexity(
    model: torch.nn.Module,
    tokenizer: object,
    passages: list[str],
    max_seq_length: int,
) -> tuple[float, float, int]:
    """Token-level perplexity over ``passages``, plus tokens/second.

    Each passage is scored on its own rather than concatenated into one long
    stream. Concatenation is the standard WikiText recipe, but this corpus is
    1149 unrelated legal articles: joining them would make the model predict
    the opening of article 912 from the end of article 377, and the result
    would be dominated by boundaries no real request contains.

    Returns ``(perplexity, tokens_per_second, tokens_scored)``.
    """
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()

    for text in passages:
        batch = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_length
        )
        input_ids = batch["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        with torch.inference_mode():
            output = model(input_ids=input_ids, labels=input_ids)
        # HF shifts labels internally and returns a mean over the (n-1)
        # predicted positions, so weight by that count to get a corpus-level
        # mean rather than a mean of per-passage means.
        predicted = input_ids.shape[1] - 1
        total_nll += float(output.loss) * predicted
        total_tokens += predicted

    elapsed = time.perf_counter() - started
    if not total_tokens:
        return float("nan"), 0.0, 0
    return (
        math.exp(total_nll / total_tokens),
        total_tokens / elapsed if elapsed else 0.0,
        total_tokens,
    )


def weight_file_mib(path: Path) -> float:
    """Just the weight files, so the 17 MB tokenizer does not flatter the ratio."""
    if not path.is_dir():
        return 0.0
    total = sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file() and f.suffix in {".safetensors", ".bin"}
    )
    return total / 1024**2


def directory_mib(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=0.01,
        help="Hessian damping. Too low and the Cholesky fails on a singular H.",
    )
    parser.add_argument(
        "--desc-act",
        action="store_true",
        help=(
            "Quantize columns in decreasing order of Hessian diagonal. Better "
            "quality, slower kernels -- off by default."
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=96,
        help="Passages used to estimate the Hessian.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=48,
        help="Held-out passages for perplexity. Disjoint from calibration.",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap calibration AND eval passages (smoke test).",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads or (os.cpu_count() or 1))
    torch.manual_seed(args.seed)

    # GPTQModel's TORCH backend wraps its dequantize-and-matmul in
    # torch.compile, and Inductor needs an MSVC C++ compiler -- `cl.exe` --
    # which is not installed on this machine. Without this the reload dies with
    # `InductorError: InvalidCxxCompiler: Compiler: cl is not found` *after*
    # the hour of quantization has already been spent. Suppressing dynamo
    # errors falls back to eager execution, which computes exactly the same
    # arithmetic; the only thing lost is a compilation speedup this machine
    # could not have used anyway, and the speed figures from this path are
    # already reported as non-representative.
    torch._dynamo.config.suppress_errors = True

    hardware = load_hardware_module().describe_hardware()
    print(f"hardware   : {hardware.summary()}")
    print(f"threads    : {torch.get_num_threads()}")

    if not args.chunks.is_file():
        print(f"ERROR corpus chunks not found: {args.chunks}", file=sys.stderr)
        return 1

    passages = corpus_passages(args.chunks, args.min_chars)
    calibration_n = args.limit or args.calibration_samples
    eval_n = args.limit or args.eval_samples
    calib_texts, eval_texts = split_passages(passages, calibration_n, eval_n, args.seed)
    if not calib_texts or not eval_texts:
        print(
            f"ERROR not enough passages: {len(passages)} available, "
            f"need {calibration_n + eval_n}",
            file=sys.stderr,
        )
        return 1
    print(f"corpus     : {len(passages)} passages >= {args.min_chars} chars")
    print(f"calibration: {len(calib_texts)}  held-out: {len(eval_texts)} (disjoint)")

    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolve_local_snapshot(args.model)
    if source != args.model:
        print(f"loading from : {source}")
    tokenizer = AutoTokenizer.from_pretrained(source)

    results: dict = {
        "schema_version": 1,
        "method": "GPTQ",
        "library": "gptqmodel==4.2.5",
        "model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "damp_percent": args.damp_percent,
        "desc_act": args.desc_act,
        "calibration_samples": len(calib_texts),
        "eval_samples": len(eval_texts),
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "calibration_source": str(args.chunks),
        "hardware": hardware.as_dict(),
        "notes": [],
    }

    # --- baseline -----------------------------------------------------------
    if not args.skip_baseline:
        print("\nloading the unquantized model (float32, CPU)...")
        # torch_dtype, not dtype: transformers 4.x spells it the former way and
        # renamed it only in 5.x. Both interpreters in this project are on 4.x.
        baseline = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=torch.float32
        )
        baseline.eval()
        params = sum(p.numel() for p in baseline.parameters())
        print(f"parameters : {params:,}")
        print("measuring baseline perplexity on the held-out split...")
        ppl, tps, tokens = perplexity(
            baseline, tokenizer, eval_texts, args.max_seq_length
        )
        print(f"  baseline perplexity {ppl:.4f}  ({tps:.1f} tok/s, {tokens} tokens)")
        results["baseline"] = {
            "dtype": "float32",
            "parameters": params,
            "perplexity": ppl,
            "tokens_per_second": tps,
            "tokens_scored": tokens,
        }
        del baseline

    # --- quantize -----------------------------------------------------------
    print(
        f"\nquantizing: {args.bits}-bit, group_size={args.group_size}, "
        f"damp={args.damp_percent}, desc_act={args.desc_act}"
    )
    quantize_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        desc_act=args.desc_act,
    )
    model = GPTQModel.load(source, quantize_config, device="cpu")

    started = time.perf_counter()
    model.quantize(calib_texts, batch_size=1, tokenizer=tokenizer)
    quantize_seconds = time.perf_counter() - started
    print(f"  quantized in {quantize_seconds / 60:.1f} min")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    del model

    weights_mib = weight_file_mib(args.out)
    print(f"  checkpoint {args.out}")
    print(
        f"  weights    {weights_mib:.1f} MiB "
        f"(directory {directory_mib(args.out):.1f} MiB)"
    )
    results["quantized"] = {
        "checkpoint": str(args.out),
        "weights_mib": weights_mib,
        "directory_mib": directory_mib(args.out),
        "quantize_minutes": quantize_seconds / 60,
    }

    # --- reload and measure -------------------------------------------------
    print("\nreloading the 4-bit checkpoint (TORCH backend) and measuring...")
    try:
        quantized = GPTQModel.load(str(args.out), device="cpu", backend=BACKEND.TORCH)
        ppl, tps, tokens = perplexity(
            quantized, tokenizer, eval_texts, args.max_seq_length
        )
        print(f"  4-bit perplexity {ppl:.4f}  ({tps:.1f} tok/s, {tokens} tokens)")
        results["quantized"].update(
            {
                "perplexity": ppl,
                "tokens_per_second": tps,
                "tokens_scored": tokens,
                "measured_via": "gptqmodel TORCH backend on CPU",
            }
        )
        results["notes"].append(
            "tokens_per_second is a CPU dequantize-and-matmul figure from the "
            "pure-torch backend. It is NOT a 4-bit kernel speedup: the CUDA "
            "kernels that make 4-bit fast cannot run on this machine, so the "
            "4-bit row is expected to be SLOWER here than float32, not faster."
        )
    except Exception as error:
        print(f"  TORCH backend failed ({type(error).__name__}: {error})")
        results["notes"].append(
            f"Could not run the quantized checkpoint: {type(error).__name__}: "
            f"{error}. Perplexity after quantization is NOT measured."
        )

    base = results.get("baseline", {})
    quant = results.get("quantized", {})
    if base.get("perplexity") and quant.get("perplexity"):
        delta = quant["perplexity"] - base["perplexity"]
        original_mib = base["parameters"] * 2 / 1024**2  # BF16 checkpoint
        results["comparison"] = {
            "perplexity_before": base["perplexity"],
            "perplexity_after": quant["perplexity"],
            "perplexity_delta": delta,
            "perplexity_delta_percent": 100 * delta / base["perplexity"],
            "bf16_weights_mib": original_mib,
            "weight_compression_ratio": (
                original_mib / quant["weights_mib"]
                if quant.get("weights_mib")
                else None
            ),
        }
        print(
            f"\nperplexity {base['perplexity']:.4f} -> {quant['perplexity']:.4f} "
            f"({results['comparison']['perplexity_delta_percent']:+.2f}%)"
        )
        print(
            f"weights    {original_mib:.1f} MiB (BF16) -> {quant['weights_mib']:.1f} "
            f"MiB (4-bit) = {results['comparison']['weight_compression_ratio']:.2f}x"
        )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
