from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.streams import load_locked_manifest  # noqa: E402


MANIFEST_FIELDS = ("batch", "domain", "position", "sample_id")
REAL_COMPONENTS = frozenset({"0_real", "real", "nature"})
FAKE_COMPONENTS = frozenset({"1_fake", "fake", "ai"})


def binary_label_from_sample_id(sample_id: str) -> int:
    """Recover the evaluator-only binary label from a canonical logical path."""

    components = set(PurePosixPath(sample_id).parts)
    is_real = bool(components & REAL_COMPONENTS)
    is_fake = bool(components & FAKE_COMPONENTS)
    if is_real == is_fake:
        raise ValueError(
            "Prior-shift manifest requires exactly one canonical real/fake path "
            f"component: {sample_id}"
        )
    return 0 if is_real else 1


def class_counts(samples_per_domain: int, fake_ratio: Decimal) -> tuple[int, int]:
    if samples_per_domain < 2:
        raise ValueError("samples_per_domain must be at least two")
    if not (Decimal(0) < fake_ratio < Decimal(1)):
        raise ValueError("fake_ratio must be strictly between zero and one")
    exact_fake = Decimal(samples_per_domain) * fake_ratio
    integral_fake = exact_fake.to_integral_value()
    if exact_fake != integral_fake:
        raise ValueError(
            "samples_per_domain * fake_ratio must produce an integer class count"
        )
    fake = int(integral_fake)
    real = samples_per_domain - fake
    if real < 1 or fake < 1:
        raise ValueError("prior-shift manifests must retain both classes")
    return real, fake


def _domain_groups(
    manifest: Sequence[Mapping[str, int | str]],
) -> OrderedDict[str, list[Mapping[str, int | str]]]:
    grouped: OrderedDict[str, list[Mapping[str, int | str]]] = OrderedDict()
    completed: set[str] = set()
    active: str | None = None
    for row in manifest:
        domain = str(row["domain"])
        if domain != active:
            if domain in completed:
                raise ValueError(f"Input manifest revisits completed domain: {domain}")
            if active is not None:
                completed.add(active)
            active = domain
            grouped.setdefault(domain, [])
        grouped[domain].append(row)
    return grouped


def _selection_priority(
    *, salt: str, seed: int, domain: str, sample_id: str
) -> str:
    payload = f"{salt}\0{seed}\0{domain}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_prior_shift_rows(
    manifest: Sequence[Mapping[str, int | str]],
    *,
    samples_per_domain: int,
    fake_ratio: Decimal,
    seed: int,
    batch_size: int,
    selection_salt: str = "ctta4aid-prior-shift-v1",
) -> tuple[list[dict[str, int | str]], dict[str, dict[str, int]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not selection_salt:
        raise ValueError("selection_salt must not be empty")
    requested_real, requested_fake = class_counts(samples_per_domain, fake_ratio)
    output: list[dict[str, int | str]] = []
    summary: dict[str, dict[str, int]] = {}

    for domain, domain_rows in _domain_groups(manifest).items():
        by_label: dict[int, list[Mapping[str, int | str]]] = {0: [], 1: []}
        for row in domain_rows:
            sample_id = str(row["sample_id"])
            by_label[binary_label_from_sample_id(sample_id)].append(row)

        requested = {0: requested_real, 1: requested_fake}
        selected_ids: set[str] = set()
        for label in (0, 1):
            available = by_label[label]
            if len(available) < requested[label]:
                class_name = "real" if label == 0 else "fake"
                raise ValueError(
                    f"Domain {domain!r} has only {len(available)} {class_name} "
                    f"samples; {requested[label]} requested"
                )
            ranked = sorted(
                available,
                key=lambda row: (
                    _selection_priority(
                        salt=selection_salt,
                        seed=seed,
                        domain=domain,
                        sample_id=str(row["sample_id"]),
                    ),
                    int(row["position"]),
                ),
            )
            selected_ids.update(
                str(row["sample_id"]) for row in ranked[: requested[label]]
            )

        selected = [
            row for row in domain_rows if str(row["sample_id"]) in selected_ids
        ]
        if len(selected) != samples_per_domain:
            raise RuntimeError(f"Prior-shift selection failed for domain {domain!r}")
        summary[domain] = {"real": requested_real, "fake": requested_fake}
        for row in selected:
            position = len(output)
            output.append(
                {
                    "batch": position // batch_size,
                    "domain": domain,
                    "position": position,
                    "sample_id": str(row["sample_id"]),
                }
            )
    return output, summary


def write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_ratio(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"Invalid decimal ratio: {value}") from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-length target-prior-shift manifest from an existing "
            "balanced online manifest"
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--fake-ratio", type=_parse_ratio, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--selection-salt", default="ctta4aid-prior-shift-v1"
    )
    args = parser.parse_args()

    manifest = load_locked_manifest(args.input_manifest)
    rows, summary = build_prior_shift_rows(
        manifest,
        samples_per_domain=args.samples_per_domain,
        fake_ratio=args.fake_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        selection_salt=args.selection_salt,
    )
    write_manifest(args.output_manifest, rows)
    digest = hashlib.sha256(args.output_manifest.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "input_manifest": str(args.input_manifest.resolve()),
                "output_manifest": str(args.output_manifest.resolve()),
                "manifest_sha256": digest,
                "fake_ratio": str(args.fake_ratio),
                "samples_per_domain": args.samples_per_domain,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "selection_salt": args.selection_salt,
                "domains": summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
