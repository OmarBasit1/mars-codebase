"""Convert the old flat MARS workload into the collected-trace format.

The old format is a dict ``request_id -> [segment...]`` where each segment is
``{prompt_tokens, completion_tokens, api_time, api_token_length}`` (every request
in ``diverse_oneapi_merged_exp_uniform.json`` is a single API call: 2 segments).

The new format (consumed by ``mars.bench.run``) is a dict
``{request_id: {name, api_time, multi_turn:[turn...]}}`` where each request is a
single-agent ReAct loop: ``N`` segments -> ``N`` turns; turns ``0..N-2`` carry an
API ``tool`` (``{name:"tool", api_time}``) and the final turn carries none.
``tool_output_len`` is dropped (it is inferred from the next turn's ``prompt_len``).
A sibling ``agent_prefix_len_dict.json`` (``{"agent:0": 0}``) carries the shared
system-prefix length per agent type (0: old prompts already include any system text).

Run (from a neutral cwd):
    .../python -m mars.bench.convert_workload \
        --in .../diverse_oneapi_merged_exp_uniform.json --out .../new_workload.json
    # also writes .../agent_prefix_len_dict.json next to --out
"""

from __future__ import annotations

import argparse
import json
import os

AGENT = "agent:0"


def convert(old: dict[str, list[dict]]) -> dict:
    """Map the old ``{id: [segments]}`` dict to ``{id: agent_invocation}``."""
    jobs = {}
    for rid, segs in old.items():
        turns = []
        for j, s in enumerate(segs):
            turn = {
                "prompt_len": int(s.get("prompt_tokens", 1)),
                "output_len": int(s.get("completion_tokens", 1)),
            }
            # Every segment but the last is followed by an API/tool call.
            if j < len(segs) - 1:
                turn["tool"] = {
                    "name": "tool",
                    "api_time": float(s.get("api_time", 0.0)),
                }
            turns.append(turn)
        jobs[rid] = {"name": AGENT, "api_time": 0.0, "multi_turn": turns}
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="old flat workload JSON")
    ap.add_argument("--out", required=True, help="output trace-format JSON file")
    args = ap.parse_args()

    old = json.load(open(args.inp))
    jobs = convert(old)
    with open(args.out, "w") as f:
        json.dump(jobs, f)
    prefix_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                               "agent_prefix_len_dict.json")
    with open(prefix_path, "w") as f:
        json.dump({AGENT: 0}, f)
    print(f"converted {len(old)} requests -> {len(jobs)} jobs ; wrote {args.out}")
    print(f"wrote prefix dict -> {prefix_path}")


if __name__ == "__main__":
    main()
