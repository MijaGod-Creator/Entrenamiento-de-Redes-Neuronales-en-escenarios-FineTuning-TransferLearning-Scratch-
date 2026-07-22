import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config.settings import IMAGE_EXTENSIONS, RAFDB_LABEL_MAP


@dataclass
class DatasetStructure:
    zip_path: Path
    extraction_dir: Path
    dataset_root: Path
    label_files: list[Path]
    image_files: list[Path]
    split_dirs: dict[str, Path]


class RAFDBInspector:
    def __init__(self, zip_path: Path, extraction_dir: Path):
        self.zip_path = zip_path
        self.extraction_dir = extraction_dir

    def inspect_zip(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            rows = []
            for info in archive.infolist():
                suffix = Path(info.filename).suffix.lower()
                parts = Path(info.filename).parts
                rows.append(
                    {
                        "path": info.filename,
                        "top_level": parts[0] if parts else "",
                        "suffix": suffix,
                        "size_bytes": info.file_size,
                        "is_image": suffix in IMAGE_EXTENSIONS,
                        "is_text": suffix in {".txt", ".csv", ".json", ".xml"},
                    }
                )
        return pd.DataFrame(rows)

    def extract_if_needed(self) -> None:
        marker = self.extraction_dir / ".extracted_from_archive_2"
        if marker.exists():
            return

        if self.extraction_dir.exists():
            shutil.rmtree(self.extraction_dir)
        self.extraction_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            archive.extractall(self.extraction_dir)
        marker.write_text(str(self.zip_path), encoding="utf-8")

    def detect_structure(self) -> DatasetStructure:
        image_files = sorted(
            p for p in self.extraction_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_files = sorted(
            p
            for p in self.extraction_dir.rglob("*")
            if p.suffix.lower() in {".txt", ".csv", ".json", ".xml"}
        )

        if not image_files:
            raise FileNotFoundError("No se encontraron imagenes dentro del ZIP extraido.")

        dataset_root = self._infer_dataset_root(image_files)
        split_dirs = self._find_split_dirs(dataset_root)

        return DatasetStructure(
            zip_path=self.zip_path,
            extraction_dir=self.extraction_dir,
            dataset_root=dataset_root,
            label_files=label_files,
            image_files=image_files,
            split_dirs=split_dirs,
        )

    def build_metadata(self, structure: DatasetStructure) -> tuple[pd.DataFrame, pd.DataFrame]:
        label_table = self._load_label_tables(structure.label_files)
        label_lookup = {
            (row["split"], row["image"]): row["label"]
            for _, row in label_table.dropna(subset=["label"]).iterrows()
        }

        records = []
        hashes: dict[str, str] = {}
        for image_path in structure.image_files:
            rel = image_path.relative_to(structure.extraction_dir)
            split = self._detect_split(rel)
            folder_label = self._detect_folder_label(rel, split)
            csv_label = label_lookup.get((split, image_path.name))
            label = csv_label if pd.notna(csv_label) else folder_label
            label_int = self._to_int_label(label)

            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            is_damaged = img is None
            height = width = channels = np.nan
            mean_brightness = np.nan
            if img is not None:
                height, width = img.shape[:2]
                channels = 1 if img.ndim == 2 else img.shape[2]
                mean_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

            file_hash = self._sha1(image_path)
            hashes[str(image_path)] = file_hash
            records.append(
                {
                    "path": str(image_path),
                    "relative_path": str(rel).replace("\\", "/"),
                    "image": image_path.name,
                    "split": split,
                    "folder_label": folder_label,
                    "csv_label": csv_label,
                    "label": label_int,
                    "class_name": RAFDB_LABEL_MAP.get(label_int, str(label_int)),
                    "suffix": image_path.suffix.lower(),
                    "size_bytes": image_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "mean_brightness": mean_brightness,
                    "is_damaged": is_damaged,
                    "sha1": file_hash,
                }
            )

        metadata = pd.DataFrame(records)
        metadata["is_duplicate"] = metadata.duplicated("sha1", keep=False)
        metadata["label_mismatch"] = (
            metadata["folder_label"].notna()
            & metadata["csv_label"].notna()
            & (metadata["folder_label"].astype(str) != metadata["csv_label"].astype(str))
        )

        return metadata, label_table

    @staticmethod
    def _infer_dataset_root(image_files: list[Path]) -> Path:
        common = Path(os.path.commonpath([str(p.parent) for p in image_files]))
        candidates = [common, *common.parents]
        for candidate in candidates:
            names = {p.name.lower() for p in candidate.iterdir() if p.is_dir()}
            if {"train", "test"}.issubset(names):
                return candidate
        return common

    @staticmethod
    def _find_split_dirs(dataset_root: Path) -> dict[str, Path]:
        split_dirs = {}
        for child in dataset_root.iterdir():
            if child.is_dir() and child.name.lower() in {"train", "test", "val", "valid", "validation"}:
                canonical = "validation" if child.name.lower() in {"val", "valid"} else child.name.lower()
                split_dirs[canonical] = child
        return split_dirs

    @staticmethod
    def _detect_split(relative_path: Path) -> str:
        parts = [part.lower() for part in relative_path.parts]
        for split in ("train", "test", "validation", "valid", "val"):
            if split in parts:
                return "validation" if split in {"valid", "val"} else split
        return "unknown"

    @staticmethod
    def _detect_folder_label(relative_path: Path, split: str) -> str | None:
        parts = list(relative_path.parts)
        lower_parts = [p.lower() for p in parts]
        if split in lower_parts:
            idx = lower_parts.index(split)
            if idx + 1 < len(parts) - 1:
                return parts[idx + 1]
        return None

    @staticmethod
    def _load_label_tables(label_files: list[Path]) -> pd.DataFrame:
        frames = []
        for file_path in label_files:
            if file_path.suffix.lower() != ".csv":
                continue
            frame = pd.read_csv(file_path)
            normalized = {c.lower().strip(): c for c in frame.columns}
            if "image" not in normalized or "label" not in normalized:
                continue
            split = "train" if "train" in file_path.name.lower() else "test" if "test" in file_path.name.lower() else "unknown"
            partial = frame[[normalized["image"], normalized["label"]]].copy()
            partial.columns = ["image", "label"]
            partial["split"] = split
            partial["source_file"] = str(file_path)
            frames.append(partial)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["image", "label", "split", "source_file"])

    @staticmethod
    def _to_int_label(value) -> int | float:
        try:
            return int(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _sha1(path: Path, block_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(block_size), b""):
                digest.update(block)
        return digest.hexdigest()
