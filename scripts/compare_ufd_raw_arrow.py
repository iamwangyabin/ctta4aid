from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class RawLayout:
    base: Path
    strip_test_prefix: bool

    def file_for(self, arrow_path: str) -> Path:
        relative = PurePosixPath(arrow_path)
        if self.strip_test_prefix and relative.parts[:1] == ("test",):
            relative = PurePosixPath(*relative.parts[1:])
        return self.base.joinpath(*relative.parts)

    def domain_root(self, domain: str) -> Path:
        if self.strip_test_prefix:
            return self.base / domain
        return self.base / "test" / domain

    def canonical_path(self, raw_path: Path) -> str:
        relative = raw_path.relative_to(self.base).as_posix()
        if self.strip_test_prefix:
            return f"test/{relative}"
        return relative


def resolve_raw_layout(raw_root: Path, arrow_paths: Sequence[str]) -> RawLayout:
    root = raw_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset root does not exist: {root}")
    if not arrow_paths:
        raise ValueError("At least one Arrow path is required to resolve the raw layout")

    bases = [root]
    preferred_wrapper = root / "CNN_synth_testset"
    if preferred_wrapper.is_dir():
        bases.append(preferred_wrapper)
    bases.extend(path for path in sorted(root.iterdir()) if path.is_dir())
    bases = list(dict.fromkeys(bases))

    probes = list(arrow_paths[:128])
    candidates: list[tuple[int, RawLayout]] = []
    for base in bases:
        for strip_test_prefix in (False, True):
            layout = RawLayout(base=base, strip_test_prefix=strip_test_prefix)
            hits = sum(layout.file_for(path).is_file() for path in probes)
            candidates.append((hits, layout))
    hits, layout = max(candidates, key=lambda item: item[0])
    if hits == 0:
        raise FileNotFoundError(
            f"Could not match any Arrow test path beneath raw dataset root: {root}"
        )
    return layout


def _image_payload(row: Mapping[str, Any]) -> bytes:
    payload = row.get("image")
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("Arrow row does not contain a binary image payload")
    return bytes(payload)


def _label_from_path(path: str) -> int | None:
    parts = set(PurePosixPath(path).parts)
    if "0_real" in parts:
        return 0
    if "1_fake" in parts:
        return 1
    return None


def _update_manifest_hash(digest: Any, path: str, payload: bytes) -> None:
    encoded_path = path.encode("utf-8")
    digest.update(len(encoded_path).to_bytes(8, "big"))
    digest.update(encoded_path)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _append_detail(details: list[dict[str, Any]], value: dict[str, Any], limit: int) -> None:
    if len(details) < limit:
        details.append(value)


def compare_domain(
    *,
    domain: str,
    entries: Mapping[str, Any],
    mapping: Mapping[str, Any],
    arrow_dataset: Any,
    raw_layout: RawLayout,
    max_details: int = 20,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    all_expected_paths = sorted(str(path) for path in entries)
    expected_paths = (
        all_expected_paths[:sample_limit] if sample_limit is not None else all_expected_paths
    )
    expected_set = set(all_expected_paths)
    raw_domain_root = raw_layout.domain_root(domain)
    raw_paths = {
        raw_layout.canonical_path(path)
        for path in raw_domain_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    } if raw_domain_root.is_dir() else set()

    missing_raw_paths = sorted(expected_set - raw_paths)
    extra_raw_paths = sorted(raw_paths - expected_set)
    if sample_limit is not None:
        missing_raw_paths = [path for path in expected_paths if path not in raw_paths]
        extra_raw_paths = []

    expected_by_label = {"real": 0, "fake": 0, "unknown": 0}
    details: list[dict[str, Any]] = []
    mapping_missing = 0
    row_path_mismatches = 0
    label_path_mismatches = 0
    missing_raw_files = 0
    byte_mismatches = 0
    exact_byte_matches = 0
    metadata_md5_mismatches = 0
    arrow_manifest = hashlib.sha256()
    raw_manifest = hashlib.sha256()

    for arrow_path in expected_paths:
        label = int(entries[arrow_path])
        label_key = "real" if label == 0 else "fake" if label == 1 else "unknown"
        expected_by_label[label_key] += 1
        inferred_label = _label_from_path(arrow_path)
        if inferred_label != label:
            label_path_mismatches += 1
            _append_detail(
                details,
                {
                    "kind": "label_path_mismatch",
                    "path": arrow_path,
                    "metadata_label": label,
                    "path_label": inferred_label,
                },
                max_details,
            )

        row_index_value = mapping.get(arrow_path)
        if row_index_value is None:
            mapping_missing += 1
            _append_detail(
                details,
                {"kind": "mapping_missing", "path": arrow_path},
                max_details,
            )
            continue
        row = arrow_dataset[int(row_index_value)]
        row_path = str(row.get("image_path"))
        if row_path != arrow_path:
            row_path_mismatches += 1
            _append_detail(
                details,
                {
                    "kind": "row_path_mismatch",
                    "path": arrow_path,
                    "row_path": row_path,
                    "row_index": int(row_index_value),
                },
                max_details,
            )

        raw_path = raw_layout.file_for(arrow_path)
        if not raw_path.is_file():
            missing_raw_files += 1
            _append_detail(
                details,
                {"kind": "raw_file_missing", "path": arrow_path},
                max_details,
            )
            continue

        arrow_bytes = _image_payload(row)
        raw_bytes = raw_path.read_bytes()
        _update_manifest_hash(arrow_manifest, arrow_path, arrow_bytes)
        _update_manifest_hash(raw_manifest, arrow_path, raw_bytes)
        if arrow_bytes == raw_bytes:
            exact_byte_matches += 1
        else:
            byte_mismatches += 1
            _append_detail(
                details,
                {
                    "kind": "byte_mismatch",
                    "path": arrow_path,
                    "arrow_size": len(arrow_bytes),
                    "raw_size": len(raw_bytes),
                    "arrow_sha256": hashlib.sha256(arrow_bytes).hexdigest(),
                    "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                },
                max_details,
            )

        metadata_md5 = row.get("md5")
        if metadata_md5 and str(metadata_md5).lower() != hashlib.md5(raw_bytes).hexdigest():
            metadata_md5_mismatches += 1
            _append_detail(
                details,
                {
                    "kind": "metadata_md5_mismatch",
                    "path": arrow_path,
                    "metadata_md5": str(metadata_md5),
                    "raw_md5": hashlib.md5(raw_bytes).hexdigest(),
                },
                max_details,
            )

    sampled_only = sample_limit is not None and sample_limit < len(all_expected_paths)
    checked_exact = (
        not missing_raw_paths
        and mapping_missing == 0
        and row_path_mismatches == 0
        and label_path_mismatches == 0
        and missing_raw_files == 0
        and byte_mismatches == 0
        and metadata_md5_mismatches == 0
        and exact_byte_matches == len(expected_paths)
    )
    exact = (
        not sampled_only
        and checked_exact
        and not extra_raw_paths
    )
    return {
        "domain": domain,
        "expected_files": len(all_expected_paths),
        "checked_files": len(expected_paths),
        "raw_files": len(raw_paths),
        "sampled_only": sampled_only,
        "expected_by_label": expected_by_label,
        "missing_raw_paths": len(missing_raw_paths),
        "extra_raw_paths": len(extra_raw_paths),
        "mapping_missing": mapping_missing,
        "row_path_mismatches": row_path_mismatches,
        "label_path_mismatches": label_path_mismatches,
        "missing_raw_files": missing_raw_files,
        "exact_byte_matches": exact_byte_matches,
        "byte_mismatches": byte_mismatches,
        "metadata_md5_mismatches": metadata_md5_mismatches,
        "arrow_manifest_sha256": arrow_manifest.hexdigest(),
        "raw_manifest_sha256": raw_manifest.hexdigest(),
        "checked_exact": checked_exact,
        "exact": exact,
        "missing_raw_path_examples": missing_raw_paths[:max_details],
        "extra_raw_path_examples": extra_raw_paths[:max_details],
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare an extracted official UFD/CNNDetection test set with Arrow bytes."
    )
    parser.add_argument("--arrow-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    arrow_root = args.arrow_root.expanduser().resolve()
    split_path = arrow_root / "test.json"
    mapping_path = arrow_root / "mapping.json"
    split_metadata = json.loads(split_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(split_metadata, dict) or not isinstance(mapping, dict):
        raise ValueError("Arrow test.json and mapping.json must both be JSON objects")

    domains = args.domains or sorted(split_metadata)
    missing_domains = [domain for domain in domains if domain not in split_metadata]
    if missing_domains:
        raise ValueError(f"Domains missing from Arrow test metadata: {missing_domains}")
    probe_paths = [
        str(path)
        for domain in domains
        for path in list(split_metadata[domain])[:128]
    ]
    raw_layout = resolve_raw_layout(args.raw_root, probe_paths)

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to read the Arrow image payloads") from exc
    arrow_dataset = load_from_disk(str(arrow_root), keep_in_memory=False)

    by_domain = {
        domain: compare_domain(
            domain=domain,
            entries=split_metadata[domain],
            mapping=mapping,
            arrow_dataset=arrow_dataset,
            raw_layout=raw_layout,
            max_details=args.max_details,
            sample_limit=args.sample_limit,
        )
        for domain in domains
    }
    report = {
        "arrow_root": str(arrow_root),
        "raw_root": str(args.raw_root.expanduser().resolve()),
        "resolved_raw_layout": {
            "base": str(raw_layout.base),
            "strip_test_prefix": raw_layout.strip_test_prefix,
        },
        "domains": domains,
        "domain_count": len(domains),
        "exact_domain_count": sum(result["exact"] for result in by_domain.values()),
        "all_exact": all(result["exact"] for result in by_domain.values()),
        "by_domain": by_domain,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "domain\texpected\tchecked\traw\texact_bytes\tbyte_mismatch\tmissing\t"
        "extra\tchecked_exact\tfull_exact"
    )
    for domain, result in by_domain.items():
        print(
            f"{domain}\t{result['expected_files']}\t{result['checked_files']}\t"
            f"{result['raw_files']}\t"
            f"{result['exact_byte_matches']}\t{result['byte_mismatches']}\t"
            f"{result['missing_raw_paths']}\t{result['extra_raw_paths']}\t"
            f"{result['checked_exact']}\t{result['exact']}"
        )
    if args.require_exact and not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
