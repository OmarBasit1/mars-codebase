"""Calibrate every MARS cost-model / solver coefficient to a model + GPU.

Measures the empirical cost of each KV operation so the cost model, the swap
waste, and the Gurobi solver are grounded on the real hardware:

  * RECOMPUTE / forward  -> prefill latency vs #tokens. Fit BOTH a linear model
    (``cost_a``, ``cost_c`` for ``discard_waste``) and a quadratic model
    (``solver_poly_a/b/c`` for the solver's batch polynomial).
  * SWAP  -> host<->GPU round-trip latency for KV-block-sized tensors -> the
    per-token swap latency (``solver_per_token_swap_latency``, ``cost_swap_a1``).
  * PRESERVE -> ~0 (no operation; the baseline).

Prints a ready-to-paste ``MarsConfig(...)`` block. Works with
``--load-format dummy`` too (compute cost is weight-independent), so no download
is needed. Run with the model you will actually serve.

Usage (from a neutral cwd):
    CUDA_VISIBLE_DEVICES=0 .../python examples/calibrate_cost.py --model <hf_model>
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams, TokensPrompt


def measure_prefill(llm: LLM, n_tokens: int, reps: int = 10) -> float:
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    prompt = TokensPrompt(prompt_token_ids=list(range(10, 10 + n_tokens)))
    llm.generate(prompt, sp, use_tqdm=False)  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate(prompt, sp, use_tqdm=False)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples))


def kv_bytes_per_token_from_config(model: str, tp: int = 1) -> int:
    """KV-cache bytes per token from HF config — no GPU/LLM required."""
    top = AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()
    # VLMs nest the LM config under "text_config"; fall back to top-level for pure LMs.
    d = top.get("text_config", top) if "num_hidden_layers" not in top else top
    n_layers = next((d[k] for k in ("num_hidden_layers", "n_layers", "num_layers") if k in d), None)
    n_heads  = next((d[k] for k in ("num_attention_heads", "n_head", "num_heads") if k in d), None)
    hidden   = next((d[k] for k in ("hidden_size", "n_embd", "d_model") if k in d), None)
    if None in (n_layers, n_heads, hidden):
        raise ValueError(f"Cannot derive KV shape; missing keys. Available: {sorted(d)}")
    n_kv_heads = next((d[k] for k in ("num_key_value_heads",) if k in d), n_heads) // tp
    return 2 * n_layers * n_kv_heads * (hidden // n_heads) * 2  # fp16 = 2 bytes


def measure_swap(bytes_per_tok: int, token_counts: list[int], reps: int = 10) -> list[float]:
    """Median host<->GPU round-trip (D2H + H2D) latency (ms) per token count."""
    out = []
    for ntok in token_counts:
        n_elems = max(1, (bytes_per_tok * ntok) // 2)  # fp16 elements
        x = torch.empty(n_elems, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        samples = []
        gpu_buf = None
        for i in range(reps):
            t0 = time.perf_counter()
            cpu_buf = x.to("cpu")
            gpu_buf = cpu_buf.to("cuda")
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        out.append(float(np.median(samples)))
        del x, cpu_buf, gpu_buf
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--load-format", default="auto")
    args = ap.parse_args()

    # --- SWAP (host<->GPU KV transfer) — run before LLM load to avoid memory pressure ---
    bpt = kv_bytes_per_token_from_config(args.model)
    counts = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    sw = measure_swap(bpt, counts)
    print(f"\n# swap (KV {bpt} bytes/token; D2H+H2D round-trip)")
    for n, s in zip(counts, sw):
        print(f"  tokens={n:>5}  {s:8.3f} ms  ({s / n * 1000:.3f} us/tok)")
    sa, sc = np.linalg.lstsq(np.vstack([counts, np.ones(len(counts))]).T, sw, rcond=None)[0]
    per_tok_swap_ms = max(sa, 0.0)
    print(f"  per-token swap: {per_tok_swap_ms * 1000:.3f} us/tok  (intercept {sc:.3f} ms)")

    # --- PRESERVE ---
    print("\n# preserve: ~0 (no operation)")

    # --- RECOMPUTE / forward — load LLM after swap test ---
    llm = LLM(
        model=args.model, enforce_eager=True, gpu_memory_utilization=0.95,
        max_model_len=args.max_model_len, max_num_batched_tokens=args.max_model_len,
        load_format=args.load_format,
    )

    lengths = [n for n in (8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if n <= args.max_model_len - 1]
    lat = [measure_prefill(llm, n) for n in lengths]
    print("\n# forward / recompute (prefill latency)")
    for n, l in zip(lengths, lat):
        print(f"  N={n:>5}  {l:8.3f} ms  ({l / n:.4f} ms/tok)")
    a1, c1 = np.linalg.lstsq(np.vstack([lengths, np.ones(len(lengths))]).T, lat, rcond=None)[0]
    qa, qb, qc = np.polyfit(lengths, lat, 2)
    print(f"  linear:    cost_a={a1:.6f}  cost_c={c1:.3f}")
    print(f"  quadratic: solver_poly_a={qa:.3e}  solver_poly_b={qb:.4f}  solver_poly_c={qc:.3f}")

    print("\n# === paste into MarsConfig (or additional_config) ===")
    print(f"MarsConfig(")
    print(f"    cost_a={a1:.6f}, cost_c={c1:.3f},")
    print(f"    cost_swap_a1={per_tok_swap_ms * 1000:.4f},  # us/tok")
    print(f"    solver_per_token_swap_latency={per_tok_swap_ms / 1000:.3e},  # s/tok")
    print(f"    solver_poly_a={qa:.3e}, solver_poly_b={qb:.4f}, solver_poly_c={qc:.3f},")
    print(f")")
    print(">>> CALIBRATE_DONE")


if __name__ == "__main__":
    main()
