# SPDX-License-Identifier: Apache-2.0
"""Check the complete compiled 950 BF16 paged FAInfer symbol, not execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED_SYMBOL = (
    "_Z7FAInferIu6__bf16fL6Format0ELS0_0EL9CacheMode1EL9PageShape0EL12MaskCategory4"
    "EL11CacheLayout0ELb0ELb0EEvPhS5_S5_S5_S5_S5_S5_S5_S5_S5_S5_S5_S5_ffjjmm"
)


def is_requested_symbol(symbol: str) -> bool:
    return symbol == EXPECTED_SYMBOL


def check(library: Path) -> dict:
    library = library.resolve(strict=True)
    result = subprocess.run(["strings", str(library)], capture_output=True, text=True, check=True)
    matches = sorted({symbol for symbol in result.stdout.splitlines() if is_requested_symbol(symbol)})
    if not matches:
        raise ValueError("Complete two-boolean/two-uint64-stride FAInfer symbol missing; old ffjj is not accepted")
    return {
        "library": str(library),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "symbols": matches,
        "abi_version": 1,
        "runtime_load_verified": False,
        "device_execution_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.library), indent=2))


if __name__ == "__main__":
    main()
