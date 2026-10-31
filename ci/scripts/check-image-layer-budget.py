"""Fail before native compilation when a derived base leaves too little depth."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_SAFE_LAYER_LIMIT = 120
CONTAINER_LAYER_MARGIN = 3


def filesystem_instruction_count(recipe):
    return len(re.findall(r"(?im)^(?:RUN|COPY|ADD)\s+", recipe))


def filesystem_stage_budgets(recipe):
    stages = {}
    current = "default"
    stages[current] = 0
    for line in recipe.splitlines():
        match = re.match(r"(?i)^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", line)
        if match:
            parent, name = match.groups()
            current = name or f"stage_{len(stages)}"
            stages[current] = stages.get(parent, 0)
        elif re.match(r"(?i)^(RUN|COPY|ADD)\s+", line):
            stages[current] += 1
    return stages


def layer_budget(base_layers, recipe, limit=DEFAULT_SAFE_LAYER_LIMIT):
    if base_layers < 0 or limit <= 0:
        raise ValueError("Layer counts and limit must be valid")
    stages = filesystem_stage_budgets(recipe)
    instructions = max(stages.values(), default=0)
    estimated = base_layers + instructions + CONTAINER_LAYER_MARGIN
    return {
        "base_filesystem_layers": base_layers,
        "native_filesystem_instruction_budget": instructions,
        "container_layer_margin": CONTAINER_LAYER_MARGIN,
        "estimated_depth": estimated,
        "conservative_limit": limit,
        "within_budget": estimated <= limit,
        "stage_filesystem_budgets": stages,
        "final_stage_filesystem_budget": next(reversed(stages.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--limit", type=int, default=DEFAULT_SAFE_LAYER_LIMIT)
    args = parser.parse_args()
    raw = subprocess.check_output(
        [args.docker, "image", "inspect", args.image], text=True,
    )
    images = json.loads(raw)
    layers = images[0]["RootFS"]["Layers"]
    report = layer_budget(len(layers), args.dockerfile.read_text(), args.limit)
    report["base_image"] = args.image
    print(json.dumps(report, sort_keys=True))
    if not report["within_budget"]:
        print(
            "Native build refused before compilation: the derived base leaves "
            "insufficient layer budget. Use a less-derived base or compact "
            "its layers first. No source, kernel, or daemon setting was changed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
