# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""One public CLI for the article edition."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("verify", "figures", "tables", "quick", "run")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--audit-data",
        type=Path,
        help="Local audit dataset archive or extracted directory, for verify",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Unpacked archive root for reference commands/quick",
    )
    parser.add_argument(
        "--skip-pybullet",
        action="store_true",
        help="Use nominal dynamics in quick smoke runs",
    )
    args = parser.parse_args()
    if args.command == "run" and args.config is None:
        parser.error("run requires --config")
    if args.audit_data is not None and args.command != "verify":
        parser.error("--audit-data is supported only by verify")
    if args.command != "run" and args.config is not None:
        parser.error("--config is supported only by run")
    os.environ.setdefault("MPLBACKEND", "Agg")
    output = args.output or Path("outputs") / (
        args.command + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output = output.resolve()
    try:
        if args.command == "run":
            from . import load_config, run_experiment

            print(run_experiment(load_config(args.config), output))
        else:
            output.mkdir(parents=True, exist_ok=False)
            from tools.reproduce import quick, reference

            if args.command == "quick":
                quick(output, args.skip_pybullet, args.root)
            else:
                reference(args.command, output, args.root, args.audit_data)
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
