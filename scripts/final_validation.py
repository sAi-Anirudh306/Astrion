"""Build the final M20 evidence package or validate an existing M21 package."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from time import perf_counter

from src.evaluation.final_package import build_package
from src.evaluation.final_validation import compare_packages, validate_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Import all nine configured experiments into a fresh package")
    parser.add_argument("--output", type=Path, default=Path("results/final"))
    parser.add_argument("--compare-to", type=Path, help="Validate and compare with a preserved authoritative package")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    started = perf_counter()
    if args.build:
        build_package(root, output)
    report = validate_package(output, root=root)
    if args.compare_to:
        report["reproduction_comparison"] = compare_packages((root / args.compare_to).resolve(), output)
    report["command_runtime_seconds"] = perf_counter() - started
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
