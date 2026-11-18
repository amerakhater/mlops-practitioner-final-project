"""The shared benchmark harness and the VRAM arithmetic.

The arithmetic gets tested because reports/module-4.md answers "can a 70B model
run on one 80 GB GPU" from it. A quietly wrong constant there produces a
confident, checkable-looking, wrong answer -- the worst failure mode a report
has -- and nothing else in the suite would catch it.
"""

import importlib.util
from pathlib import Path

import pytest

from prodllm.bench import describe_hardware, measure
from prodllm.bench.hardware import GpuInfo, Hardware
from prodllm.bench.timing import _percentile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    """Import a file from scripts/, which is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vram = load_script("vram_table")


# ------------------------------------------------------------------- timing


def test_measure_runs_warmup_then_repeats_and_times_only_the_repeats():
    calls = []
    stats, _ = measure(lambda: calls.append(1), warmup=3, repeats=5)

    assert len(calls) == 8
    assert stats.repeats == 5
    assert stats.warmup == 3


def test_measure_returns_the_last_value_so_a_benchmark_can_be_asserted_on():
    _, result = measure(lambda: "answer", warmup=0, repeats=2)

    assert result == "answer"


def test_measure_rejects_zero_repeats():
    with pytest.raises(ValueError, match="at least 1"):
        measure(lambda: None, repeats=0)


def test_measure_reports_a_zero_stdev_for_a_single_repeat():
    stats, _ = measure(lambda: None, warmup=0, repeats=1)

    assert stats.stdev_ms == 0.0


def test_percentile_returns_a_real_observation_not_an_interpolation():
    """An interpolated p95 at n=10 is a number that was never measured."""
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]

    assert _percentile(samples, 0.95) in samples
    assert _percentile(samples, 0.50) in samples


def test_percentile_edges_stay_in_range():
    samples = [1.0, 2.0, 3.0]

    assert _percentile(samples, 0.0) == 1.0
    assert _percentile(samples, 1.0) == 3.0


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="no samples"):
        _percentile([], 0.5)


# ------------------------------------------------------------------ hardware


def test_describe_hardware_captures_the_fields_a_report_needs():
    hardware = describe_hardware()

    assert isinstance(hardware, Hardware)
    assert hardware.logical_cores >= 1
    assert hardware.python_version
    assert "torch" in hardware.summary()


def test_hardware_separates_a_present_gpu_from_a_reachable_one():
    """The central Module 4 constraint: a CUDA card and a CPU-only torch."""
    hardware = Hardware(
        platform="Windows",
        processor="x86",
        python_version="3.12.0",
        logical_cores=16,
        ram_total_gib=32.0,
        torch_version="2.14.1+cpu",
        torch_cuda_available=False,
        gpus=[
            GpuInfo(
                name="RTX A2000",
                memory_total_mib=4096,
                driver_version="591.66",
                compute_capability="8.6",
            )
        ],
    )

    assert "NOT reachable" in hardware.summary()
    assert "RTX A2000" in hardware.summary()


def test_hardware_round_trips_through_a_dict_for_the_json_sidecar():
    payload = describe_hardware().as_dict()

    assert payload["logical_cores"] >= 1
    assert "torch_cuda_available" in payload


# ------------------------------------------------- parameter-count arithmetic


def test_qwen_1_5b_parameter_count_matches_the_published_figure():
    """Qwen2.5-1.5B is 1.54B parameters; derive it, do not quote it."""
    arch = vram.KNOWN["qwen2.5-1.5b"]

    assert 1.53e9 < arch.parameters < 1.55e9


def test_llama_70b_parameter_count_matches_the_published_figure():
    arch = vram.KNOWN["llama-3.1-70b"]

    assert 70.0e9 < arch.parameters < 71.0e9


def test_tied_embeddings_remove_a_whole_vocab_by_hidden_matrix():
    """On a 1.5B model the untied head would be a seventh of the parameters."""
    tied = vram.KNOWN["qwen2.5-1.5b"]
    untied = vram.Architecture(**{**tied.__dict__, "tie_embeddings": False})

    assert untied.parameters - tied.parameters == tied.vocab * tied.hidden


# ------------------------------------------------------- memory arithmetic


def test_four_bit_costs_more_than_four_bits_because_of_group_scales():
    """4.25 bits/weight at group_size 128, not 4.0 -- that is ~2 GiB on a 70B."""
    assert vram.BYTES["int4"] * 8 == pytest.approx(4.25)


def test_int8_is_exactly_half_of_fp16_and_int4_is_not_exactly_half_of_int8():
    arch = vram.KNOWN["llama-3.1-70b"]

    assert arch.weight_gib("int8") == pytest.approx(arch.weight_gib("fp16") / 2)
    assert arch.weight_gib("int4") > arch.weight_gib("int8") / 2


def test_kv_cache_is_linear_in_tokens():
    arch = vram.KNOWN["llama-3.1-70b"]

    assert arch.kv_cache_gib(2048) == pytest.approx(arch.kv_cache_gib(1024) * 2)


def test_kv_cache_ignores_the_weight_precision():
    """Quantizing weights does nothing for the cache, which is the real point."""
    arch = vram.KNOWN["llama-3.1-70b"]

    assert arch.kv_cache_gib(1024, "fp16") == pytest.approx(
        arch.kv_cache_gib(1024, "fp16")
    )
    assert arch.kv_cache_gib(1024, "int8") == pytest.approx(
        arch.kv_cache_gib(1024, "fp16") / 2
    )


def test_grouped_query_attention_shrinks_the_cache_by_the_head_ratio():
    """8 KV heads against 64 query heads is the 8x that makes long context fit."""
    gqa = vram.KNOWN["llama-3.1-70b"]
    multi_head = vram.Architecture(**{**gqa.__dict__, "kv_heads": gqa.attention_heads})

    ratio = multi_head.kv_bytes_per_token() / gqa.kv_bytes_per_token()
    assert ratio == pytest.approx(gqa.attention_heads / gqa.kv_heads)


def test_a_70b_model_does_not_fit_80gib_at_fp16_but_does_at_four_bit():
    """The module's headline question, asserted so the report cannot drift."""
    arch = vram.KNOWN["llama-3.1-70b"]
    budget = 80.0

    fp16 = arch.weight_gib("fp16") + vram.RUNTIME_OVERHEAD_GIB
    int4 = arch.weight_gib("int4") + vram.RUNTIME_OVERHEAD_GIB

    assert fp16 > budget
    assert int4 < budget / 2


def test_qwen_1_5b_fp16_does_not_fit_this_projects_4gib_card():
    """Why Module 4 is not optional on this hardware rather than a nicety."""
    arch = vram.KNOWN["qwen2.5-1.5b"]
    kv = arch.kv_cache_gib(4096)

    fp16_total = arch.weight_gib("fp16") + kv + vram.RUNTIME_OVERHEAD_GIB
    int4_total = arch.weight_gib("int4") + kv + vram.RUNTIME_OVERHEAD_GIB

    assert fp16_total > 4.0
    assert int4_total < 4.0


def test_architecture_reads_a_real_config_json(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(
        '{"num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 16,'
        ' "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,'
        ' "tie_word_embeddings": true}',
        encoding="utf-8",
    )

    arch = vram.from_config(config)

    assert arch.layers == 2
    assert arch.kv_heads == 2
    assert arch.effective_head_dim == 2
    assert arch.tie_embeddings is True


def test_architecture_defaults_kv_heads_to_attention_heads_when_absent(
    tmp_path: Path,
):
    """A pre-GQA config has no num_key_value_heads; it is multi-head."""
    config = tmp_path / "config.json"
    config.write_text(
        '{"num_hidden_layers": 1, "hidden_size": 8, "intermediate_size": 16,'
        ' "num_attention_heads": 4, "vocab_size": 100}',
        encoding="utf-8",
    )

    assert vram.from_config(config).kv_heads == 4
