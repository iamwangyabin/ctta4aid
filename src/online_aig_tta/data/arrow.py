from __future__ import annotations

import random
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ArrowSampleRecord:
    arrow_path: str
    batch_id: int
    row_in_batch: int
    label: int
    sample_id: str


class CAIDBenchArrowDataset:
    """Read CAIDBench images lazily from Arrow IPC files via its parquet index."""

    def __init__(
        self,
        *,
        root: str | Path,
        generator: str,
        split: str,
        transform: Any = None,
        max_samples_per_class: int | None = None,
        sample_offset_per_class: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"CAIDBench root does not exist: {self.root}")
        index_path = self.root / "index.parquet"
        if not index_path.is_file():
            raise FileNotFoundError(f"CAIDBench index does not exist: {index_path}")

        records = _load_records(index_path, generator, split)
        real_records = [record for record in records if record.label == 0]
        fake_records = [record for record in records if record.label == 1]
        if not real_records:
            raise FileNotFoundError(
                f"No real samples found for CAIDBench generator={generator!r} split={split!r}"
            )
        if not fake_records:
            raise FileNotFoundError(
                f"No fake samples found for CAIDBench generator={generator!r} split={split!r}"
            )

        if max_samples_per_class is not None or sample_offset_per_class > 0:
            rng = random.Random(seed)
            rng.shuffle(real_records)
            rng.shuffle(fake_records)
            end = (
                sample_offset_per_class + max_samples_per_class
                if max_samples_per_class is not None
                else None
            )
            real_records = real_records[sample_offset_per_class:end]
            fake_records = fake_records[sample_offset_per_class:end]

        self.records = sorted(
            [*real_records, *fake_records], key=lambda record: record.sample_id
        )
        self.transform = transform
        self._readers: dict[str, tuple[Any, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        image_bytes, label, sample_id = self.raw_item(index)
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label, sample_id

    def raw_item(self, index: int) -> tuple[bytes, int, str]:
        """Return the original encoded image without a decode/re-encode cycle."""
        record = self.records[index]
        reader = self._reader(record.arrow_path)
        batch = reader.get_batch(record.batch_id)
        if "image" not in batch.schema.names:
            raise ValueError(f"Missing image column in {record.arrow_path}")
        payload = batch.column("image")[record.row_in_batch].as_py()
        image_bytes = payload.get("bytes") if isinstance(payload, dict) else payload
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise ValueError(f"Invalid image payload for {record.sample_id}")
        return bytes(image_bytes), record.label, record.sample_id

    def _reader(self, arrow_path: str) -> Any:
        path = Path(arrow_path).expanduser()
        if not path.is_absolute():
            path = self.root / path
        key = str(path.resolve())
        cached = self._readers.get(key)
        if cached is not None:
            return cached[1]

        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for the caidbench_arrow data format"
            ) from exc
        source = pa.memory_map(key, "r")
        try:
            reader = ipc.open_file(source)
        except Exception:
            source.close()
            raise
        self._readers[key] = (source, reader)
        return reader

    def close(self) -> None:
        for source, reader in self._readers.values():
            close_reader = getattr(reader, "close", None)
            if callable(close_reader):
                with suppress(Exception):
                    close_reader()
            with suppress(Exception):
                source.close()
        self._readers.clear()

    def __getstate__(self) -> dict[str, Any]:
        self.close()
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def _load_records(index_path: Path, generator: str, split: str) -> list[ArrowSampleRecord]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for the caidbench_arrow data format"
        ) from exc

    columns = ["arrow_path", "batch_id", "row_in_batch", "label"]
    table = parquet.read_table(
        index_path,
        columns=columns,
        filters=[("generator_name", "=", generator), ("split", "=", split)],
    )
    if table.num_rows == 0:
        raise FileNotFoundError(
            f"No CAIDBench rows found for generator={generator!r} split={split!r}"
        )
    values = [table[column].to_pylist() for column in columns]
    records = []
    for arrow_path, batch_id, row_in_batch, label in zip(*values, strict=True):
        label = int(label)
        if label not in {0, 1}:
            raise ValueError(f"CAIDBench label must be 0/1, got {label}")
        sample_id = f"{arrow_path}#{int(batch_id)}:{int(row_in_batch)}"
        records.append(
            ArrowSampleRecord(
                arrow_path=str(arrow_path),
                batch_id=int(batch_id),
                row_in_batch=int(row_in_batch),
                label=label,
                sample_id=sample_id,
            )
        )
    return records
