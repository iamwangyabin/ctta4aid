from __future__ import annotations

import argparse
from pathlib import Path

from online_aig_tta.cli.common import write_json
from online_aig_tta.config import load_config


def config_path(track: str, setting: str, seed: int) -> Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return Path("configs") / f"caidbench_{track}_{setting}{suffix}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze effective CAIDBench configs beside completed outputs"
    )
    parser.add_argument("--track", required=True, choices=("genimage", "progan"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    args = parser.parse_args()

    for seed in args.seeds:
        for setting in ("single_target", "continual"):
            config = load_config(config_path(args.track, setting, seed))
            destination = Path(config["output_dir"]) / "effective_config.json"
            write_json(destination, config)
            print(destination)


if __name__ == "__main__":
    main()
