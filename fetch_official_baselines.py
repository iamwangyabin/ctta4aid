from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Repository:
    name: str
    url: str
    commit: str
    directory: str
    patch: str | None = None


REPOSITORIES = {
    "tent": Repository(
        name="TENT",
        url="https://github.com/DequanWang/tent.git",
        commit="e9e926a668d85244c66a6d5c006efbd2b82e83e8",
        directory="TENT",
    ),
    "eata": Repository(
        name="EATA",
        url="https://github.com/mr-eggplant/EATA.git",
        commit="f739b3668cc7617e9b9f1979c1a358497a3472c3",
        directory="EATA",
    ),
    "cotta": Repository(
        name="CoTTA",
        url="https://github.com/qinenergy/cotta.git",
        commit="c212a204b32be4005092e4323105a24a29ad2952",
        directory="CoTTA",
    ),
    "rotta": Repository(
        name="RoTTA",
        url="https://github.com/BIT-DA/RoTTA.git",
        commit="67e34c900cdd355fc07e55edd4c577ea7b8ebcc9",
        directory="RoTTA",
    ),
    "lame": Repository(
        name="LAME",
        url="https://github.com/fiveai/LAME.git",
        commit="d2e5f63090bc1c8129bf7cbd781029a5955e1a67",
        directory="LAME",
    ),
    "t2a": Repository(
        name="T2A",
        url="https://github.com/HongHanh2104/T2A-Think-Twice-Before-Adaptation.git",
        commit="33c8ccc64afdda260564123d6c790d030a89ff81",
        directory="T2A",
    ),
    "iapl": Repository(
        name="IAPL",
        url="https://github.com/liyih/IAPL.git",
        commit="a173e7783bbafaa00d60e6e31774a0bc14411a23",
        directory="IAPL",
        patch="patches/iapl-a173e77-compat.patch",
    ),
}


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def fetch(repository: Repository, project_root: Path, destination_root: Path) -> Path:
    destination = destination_root / repository.directory
    if not destination.exists():
        run(["git", "clone", "--no-checkout", repository.url, str(destination)])
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"Destination exists but is not a git checkout: {destination}")

    run(["git", "fetch", "--depth", "1", "origin", repository.commit], cwd=destination)
    run(["git", "checkout", "--detach", repository.commit], cwd=destination)
    actual = run(["git", "rev-parse", "HEAD"], cwd=destination, capture=True)
    if actual != repository.commit:
        raise RuntimeError(f"{repository.name} commit mismatch: {actual} != {repository.commit}")

    if repository.patch:
        patch_path = project_root / repository.patch
        check = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--check", str(patch_path)],
            cwd=destination,
            text=True,
            capture_output=True,
        )
        if check.returncode == 0:
            run(["git", "apply", "--unidiff-zero", str(patch_path)], cwd=destination)
        else:
            reverse = subprocess.run(
                [
                    "git",
                    "apply",
                    "--unidiff-zero",
                    "--reverse",
                    "--check",
                    str(patch_path),
                ],
                cwd=destination,
                text=True,
                capture_output=True,
            )
            if reverse.returncode != 0:
                raise RuntimeError(
                    f"Cannot apply compatibility patch to {repository.name}: {check.stderr}"
                )
    print(f"{repository.name}: {repository.commit} -> {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pinned official baseline repositories")
    parser.add_argument("baseline", choices=["all", *REPOSITORIES], nargs="?", default="all")
    parser.add_argument("--destination-root", default="external")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    destination_root = Path(args.destination_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    names = list(REPOSITORIES) if args.baseline == "all" else [args.baseline]
    for name in names:
        fetch(REPOSITORIES[name], project_root, destination_root)


if __name__ == "__main__":
    main()
