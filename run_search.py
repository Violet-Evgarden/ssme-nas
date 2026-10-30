"""Public CLI: inspect release status, but do not run unpublished experiments."""
import argparse
import sys

from ssme_nas.private_core import CoreNotReleasedError, require_private_core


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SSME-NAS pre-publication preview (scientific core withheld)."
    )
    parser.add_argument("--describe", action="store_true", help="Show availability only.")
    parser.add_argument("--config", help="Configuration example; execution is disabled.")
    args = parser.parse_args(argv)
    if args.describe:
        print("SSME-NAS: preview-core-withheld. See docs/RELEASE_STATUS.md.")
        print("No benchmark evaluation, training, or API request has been started.")
        return 0
    try:
        # Intentionally before opening config files, loading data, or requesting labels.
        require_private_core()
    except CoreNotReleasedError as exc:
        print(f"CoreNotReleasedError: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
