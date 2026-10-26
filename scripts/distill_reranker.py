"""Stage 2 of re-ranker distillation: train the student on the teacher's scores.

    uv run python scripts/rerank_teacher_scores.py --max-queries 400   # stage 1
    uv run python scripts/distill_reranker.py --epochs 2               # stage 2

Reads the file stage 1 wrote and trains a small cross-encoder to reproduce the
teacher's *ordering* of each candidate list. The teacher is never loaded here.

**The student.** ``intfloat/multilingual-e5-small`` with a single-logit
classification head, truncated to the first four of its twelve layers. Three
reasons for that base rather than a purpose-built small model:

* It uses the **same tokenizer and the same 250 002-token sentencepiece
  vocabulary as the teacher** -- both are ``XLMRobertaTokenizerFast`` -- so the
  two models see identical token ids for the same text and the student is
  fitting the teacher's scores rather than the teacher's scores plus a
  retokenization. (The *encoder* differs: e5-small is a BERT stack and the
  teacher is XLM-RoBERTa. That does not matter here, because what is being
  transferred is a scoring function, not weights.)
* It is already **in this project's artifact cache**, because it is the
  retrieval encoder. The distilled re-ranker therefore adds no new download to
  a deployment.
* Its 384-wide, 12-layer encoder is small enough that **truncating to four
  layers leaves a model that runs on CPU inside a request**, which is the
  binding constraint -- this box has no usable GPU (see reports/module-4.md).

**What distillation actually buys here is compute, not parameter count**, and
the two come apart badly on a multilingual model. Dropping 8 of 12 layers
removes only ~12% of the parameters, because a 250 002 x 384 embedding table is
96M of the 118M total and is untouched. The honest figure is therefore not
"5.5x smaller" -- it is 24 layers of width 1024 replaced by 4 layers of width
384. Counting the per-token projection and MLP work as 12d^2 per layer, that is
301 989 888 against 7 077 888, a **42.7x** reduction in FLOPs per scored pair,
and that is what shows up as latency. The report measures latency rather than
quoting either ratio, because the parameter ratio understates the gain by about
eightfold and the FLOP ratio ignores memory bandwidth, which on CPU is often
what actually binds.

**The loss is listwise KL, not MSE on the raw scores.** What a re-ranker is
judged on is the order it puts a candidate list in, and the teacher's absolute
logits are on a scale the student has no reason to reproduce -- a student that
learned every score with a constant offset would have perfect ranking and a
large MSE. Taking a softmax over each candidate list and matching the resulting
distributions optimises exactly the ranking and is invariant to that offset.
It also weights the examples usefully for free: a list where the teacher is
confident about one article produces a peaked target and a strong gradient,
while a list the teacher considers uniformly mediocre produces a flat target
and contributes little, which is the correct relative importance.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from prodllm.bench.hardware import describe_hardware
from prodllm.config import get_settings

DEFAULT_SCORES = Path("data/processed/rerank_teacher_scores.json")
DEFAULT_STUDENT_BASE = "intfloat/multilingual-e5-small"


def truncate_layers(model: torch.nn.Module, keep: int) -> int:
    """Keep only the first ``keep`` encoder layers, in place.

    Depth-pruning from the bottom rather than taking every k-th layer: the
    lower layers of an encoder carry the lexical and syntactic features a
    relevance head needs most, and a contiguous stack from the embeddings up
    stays a valid network with the residual stream unbroken.

    The config is updated as well as the module list, so that
    ``save_pretrained`` writes a config a plain ``from_pretrained`` can reload
    without this script being in the loop.
    """
    encoder = model.base_model.encoder
    total = len(encoder.layer)
    if keep >= total:
        return total
    encoder.layer = torch.nn.ModuleList(list(encoder.layer)[:keep])
    model.config.num_hidden_layers = keep
    return keep


def listwise_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """KL(teacher || student) over one candidate list's softmax."""
    target = F.softmax(teacher_logits, dim=-1)
    predicted = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(predicted, target, reduction="sum")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path, default=settings.reranker_dir)
    parser.add_argument("--base", default=DEFAULT_STUDENT_BASE)
    parser.add_argument(
        "--keep-layers",
        type=int,
        default=4,
        help="Encoder layers to keep from the base (of 12).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help=(
            "Must match what stage 1 scored the teacher at, or the student is "
            "fitting labels for text it cannot see."
        ),
    )
    parser.add_argument(
        "--lists-per-step",
        type=int,
        default=1,
        help=(
            "Candidate lists per optimizer step. One list is already ~20 "
            "sequences, so this multiplies an already-large batch."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Held-out fraction of the pseudo-query lists, for the loss curve.",
    )
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "torch intra-op threads for this offline job (0 = all cores). "
            "Overrides the serving cap of PRODLLM_TORCH_NUM_THREADS."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Train on the first N lists (smoke test)."
    )
    args = parser.parse_args()

    if not args.scores.is_file():
        print(
            f"ERROR teacher scores not found: {args.scores}\n"
            "Run scripts/rerank_teacher_scores.py first.",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(args.scores.read_text(encoding="utf-8"))
    passages: dict[str, str] = payload["passages"]
    examples: list[dict] = payload["examples"]
    if args.limit:
        examples = examples[: args.limit]

    if payload.get("teacher_max_seq_length") != args.max_seq_length:
        print(
            f"WARNING teacher scored at max_seq_length="
            f"{payload.get('teacher_max_seq_length')} but the student trains at "
            f"{args.max_seq_length}. The student is being asked to reproduce "
            "scores for text it is not shown.",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = max(1, int(len(examples) * args.val_fraction))
    val, train = examples[:split], examples[split:]

    hardware = describe_hardware()
    # All cores by default. This is an offline training job that owns the
    # machine, not a served replica, so the serving thread cap of
    # PRODLLM_TORCH_NUM_THREADS=1 would be the wrong policy here -- it cost a
    # 3.9x slowdown in stage 1 before it was lifted.
    torch.set_num_threads(args.threads or hardware.logical_cores or 1)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    before = sum(p.numel() for p in model.parameters())
    kept = truncate_layers(model, args.keep_layers)
    after = sum(p.numel() for p in model.parameters())

    print(f"hardware     : {hardware.summary()}")
    print(f"threads      : {torch.get_num_threads()}")
    print(
        f"teacher      : {payload['teacher']} "
        f"({payload['teacher_parameters']:,} params)"
    )
    print(
        f"student base : {args.base} ({before:,} params, "
        f"{model.config.num_hidden_layers + (12 - kept)} layers)"
    )
    print(
        f"student      : {kept} layers kept, {after:,} params "
        f"({after / before:.0%} of base, {after / payload['teacher_parameters']:.1%} "
        f"of teacher)"
    )
    print(f"train lists  : {len(train)}  val lists: {len(val)}")
    print(
        f"epochs       : {args.epochs}  lr: {args.lr}  "
        f"max_seq_length: {args.max_seq_length}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_total = max(1, math.ceil(len(train) / args.lists_per_step) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps_total, pct_start=0.1
    )

    def forward_list(example: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Student logits and teacher targets for one candidate list."""
        texts = [passages[key] for key in example["candidates"]]
        encoded = tokenizer(
            [example["query"]] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        student = model(**encoded).logits.reshape(-1)
        teacher = torch.tensor(example["teacher_scores"], dtype=torch.float32)
        return student, teacher

    def validate() -> float:
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for example in val:
                student, teacher = forward_list(example)
                total += float(listwise_kl(student, teacher))
        model.train()
        return total / max(1, len(val))

    history: list[dict] = []
    print(f"\nval loss before training: {validate():.4f}")

    model.train()
    started = time.perf_counter()
    step = 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        running = 0.0
        seen = 0
        for start in range(0, len(train), args.lists_per_step):
            batch = train[start : start + args.lists_per_step]
            optimizer.zero_grad()
            loss = torch.zeros(())
            for example in batch:
                student, teacher = forward_list(example)
                loss = loss + listwise_kl(student, teacher)
            loss = loss / len(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step + 1 < steps_total:
                scheduler.step()
            step += 1
            # .detach() before the float: without it torch warns about reading
            # a value off a tensor that still carries grad history.
            running += float(loss.detach())
            seen += 1

            if seen % 20 == 0:
                elapsed = time.perf_counter() - started
                done = step / steps_total
                eta = (elapsed / done - elapsed) / 60 if done else 0.0
                total_steps = math.ceil(len(train) / args.lists_per_step)
                print(
                    f"  epoch {epoch} step {seen}/{total_steps} "
                    f"train_kl {running / seen:.4f} eta {eta:.0f} min",
                    flush=True,
                )

        val_loss = validate()
        history.append(
            {
                "epoch": epoch,
                "train_kl": running / max(1, seen),
                "val_kl": val_loss,
                "elapsed_min": round((time.perf_counter() - started) / 60, 2),
            }
        )
        print(
            f"epoch {epoch}: train_kl {running / max(1, seen):.4f}  "
            f"val_kl {val_loss:.4f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    card = {
        "schema_version": 1,
        "student_base": args.base,
        "layers_kept": kept,
        "parameters": after,
        "base_parameters": before,
        "teacher": payload["teacher"],
        "teacher_parameters": payload["teacher_parameters"],
        "max_seq_length": args.max_seq_length,
        "loss": "listwise_kl",
        "epochs": args.epochs,
        "lr": args.lr,
        "train_lists": len(train),
        "val_lists": len(val),
        "candidates_per_list": payload["candidates"],
        "seed": args.seed,
        "history": history,
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 2),
        "hardware": hardware.as_dict(),
    }
    (args.out / "distillation.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )

    share = after / payload["teacher_parameters"]
    print(f"\nwrote {args.out}")
    print(f"  parameters : {after:,} ({share:.1%} of teacher)")
    if history:
        print(
            f"  val_kl     : {history[-1]['val_kl']:.4f} "
            f"(from {history[0]['val_kl']:.4f})"
        )
    print(f"  card       : {args.out / 'distillation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
