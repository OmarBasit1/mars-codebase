"""Phase 4 calibration microbench: fit cost-model coefficients to a model/GPU.

Measures single-forward prefill latency vs prompt length and fits
``latency(N) ~ a*N + c`` (ms). Feed the suggested ``cost_a``/``cost_c`` into
``MarsConfig`` (and set ``cost_max_ragged_batch`` to where tokens/sec saturates).

IMPORTANT: the coefficients are MODEL-specific (a 125M model is dominated by
fixed overhead; a multi-billion-param model has a meaningful per-token slope).
Run this with the model you will actually serve (Phase 7) -- the defaults in
MarsConfig stay at the original MARS values until then.

Usage (from a neutral cwd):
    CUDA_VISIBLE_DEVICES=0 .../python examples/calibrate_cost.py --model <hf_model>
"""

import argparse
import time

import numpy as np
from vllm import LLM, SamplingParams, TokensPrompt


def measure(llm: LLM, n_tokens: int, reps: int = 5) -> float:
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    prompt = TokensPrompt(prompt_token_ids=list(range(10, 10 + n_tokens)))
    llm.generate(prompt, sp, use_tqdm=False)  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate(prompt, sp, use_tqdm=False)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.40,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
    )
    lengths = [n for n in (16, 64, 128, 256, 512, 1024) if n <= args.max_model_len - 1]
    latencies = [measure(llm, n) for n in lengths]
    for n, lat in zip(lengths, latencies):
        print(f"N={n:>5}  prefill_latency={lat:8.3f} ms  ({lat / n:.4f} ms/tok)")

    A = np.vstack([lengths, np.ones(len(lengths))]).T
    a, c = np.linalg.lstsq(A, latencies, rcond=None)[0]
    print(f"\nfit: latency(N) ~ {a:.5f}*N + {c:.3f} ms   (model={args.model})")
    print(f"suggested: cost_a={a:.5f}  cost_c={c:.3f}")
    print(">>> CALIBRATE_DONE")


if __name__ == "__main__":
    main()
