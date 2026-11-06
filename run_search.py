"""Run SEME-NAS on a user-supplied benchmark table; no network requests."""
import argparse
import sys

from ssme_nas import RELEASE_STATUS, __version__
from ssme_nas.runner import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="SEME-NAS structural edit memory search.")
    parser.add_argument("--describe", action="store_true", help="Describe the implementation without I/O.")
    parser.add_argument("--config", help="Experiment JSON in configs/.")
    parser.add_argument("--benchmark", help="Local official benchmark file or exported JSON table.")
    parser.add_argument("--output", help="New output directory (must not already exist).")
    parser.add_argument("--concepts", help="Override frozen concepts JSON.")
    parser.add_argument("--starts", help="Override initial populations and seeds JSON.")
    parser.add_argument("--validation-only", action="store_true", help="Seal winners but skip final Test reads.")
    args = parser.parse_args(argv)
    if args.describe:
        print(f"SEME-NAS {__version__}: {RELEASE_STATUS}.")
        print("Native NB101/NB201 adapters, memory compiler, mutation selector, controller and evaluators.")
        print("Frozen concepts and initial populations are included. Supply your benchmark with --benchmark.")
        print("No benchmark evaluation, training, or API request has been started.")
        return 0
    missing = [flag for flag in ("config", "benchmark", "output") if not getattr(args, flag)]
    if missing:
        print("Required arguments: " + ", ".join("--" + flag for flag in missing), file=sys.stderr)
        return 2
    try:
        result = run(args.config, args.benchmark, args.output, concepts_path=args.concepts,
                     starts_path=args.starts, validation_only=args.validation_only,
                     progress=lambda message: print(message, flush=True))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Completed {result['search_runs']} runs. Results: {args.output}/RESULTS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
