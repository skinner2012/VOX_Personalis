"""CLI entry point for zipformer_finetune — delegates to sub-commands."""

import argparse
import sys

from scripts.zipformer_finetune.manifest_to_lhotse import main as manifest_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zipformer fine-tuning helpers (local prep for M1 cloud run)."
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "manifest-to-lhotse",
        help="Convert S1 CSV manifests to lhotse CutSet format.",
    )

    args, remaining = parser.parse_known_args()

    if args.command == "manifest-to-lhotse" or args.command is None:
        sys.argv = [sys.argv[0]] + remaining
        return int(manifest_main())

    parser.print_help()
    return 1
