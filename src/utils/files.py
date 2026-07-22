from pathlib import Path
from typing import Iterable


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_dataset_zip(project_root: Path, candidate_names: tuple[str, ...]) -> Path:
    search_roots = [project_root, project_root / "dataset"]
    matches: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue
        for candidate in candidate_names:
            matches.extend(root.rglob(candidate))

    if not matches:
        zip_files = list(project_root.rglob("*.zip"))
        if len(zip_files) == 1:
            return zip_files[0]
        raise FileNotFoundError(
            "No se encontro Archive(2).zip. Coloque el ZIP en la raiz del proyecto "
            "o en la carpeta dataset/."
        )

    return sorted(matches, key=lambda p: len(str(p)))[0]
