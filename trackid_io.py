from __future__ import annotations

import configparser
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from trackid_core import bee_occurrences, rect_bounds


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
STATE_NAME = ".trackid_review_state.json"
TRAJECTORY_STATE_NAME = ".trackid_trajectory_review.json"


def image_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def load_document(image_path: Path) -> dict:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"图片缺少同名 JSON: {image_path.name}")
    with json_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def atomic_json_write(path: Path, data: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_documents(image_paths: list[Path], documents: list[dict]) -> None:
    for image_path, document in zip(image_paths, documents):
        atomic_json_write(image_path.with_suffix(".json"), document)


def load_reviewed(folder: Path) -> set[str]:
    path = folder / STATE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(value) for value in data.get("reviewed", [])}
    except (OSError, json.JSONDecodeError):
        return set()


def save_reviewed(folder: Path, reviewed: set[str]) -> None:
    atomic_json_write(
        folder / STATE_NAME,
        {"version": 1, "reviewed": sorted(reviewed)},
    )


def load_trajectory_reviews(folder: Path) -> dict[str, str]:
    path = folder / TRAJECTORY_STATE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(signature): str(status)
            for signature, status in data.get("statuses", {}).items()
            if status in {"passed", "issue"}
        }
    except (OSError, json.JSONDecodeError):
        return {}


def save_trajectory_reviews(folder: Path, statuses: dict[str, str]) -> None:
    deferred_new_ids = load_deferred_new_ids(folder)
    atomic_json_write(
        folder / TRAJECTORY_STATE_NAME,
        {
            "version": 1,
            "statuses": dict(sorted(statuses.items())),
            "deferred_new_ids": sorted(deferred_new_ids),
        },
    )


def load_deferred_new_ids(folder: Path) -> set[int]:
    path = folder / TRAJECTORY_STATE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            int(track_id)
            for track_id in data.get("deferred_new_ids", [])
            if int(track_id) > 0
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return set()


def save_deferred_new_ids(folder: Path, track_ids: set[int]) -> None:
    atomic_json_write(
        folder / TRAJECTORY_STATE_NAME,
        {
            "version": 1,
            "statuses": dict(sorted(load_trajectory_reviews(folder).items())),
            "deferred_new_ids": sorted(
                track_id for track_id in track_ids if track_id > 0
            ),
        },
    )


def create_section_backup(folder: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = folder.parent / f"{folder.name}_trackid修正备份_{timestamp}"
    counter = 1
    while destination.exists():
        destination = folder.parent / f"{folder.name}_trackid修正备份_{timestamp}_{counter}"
        counter += 1
    shutil.copytree(folder, destination)
    return destination


def export_mot(folder: Path, image_paths: list[Path], documents: list[dict]) -> None:
    mot_dir = folder / "MOT"
    mot_dir.mkdir(exist_ok=True)
    rows = []
    for mot_frame, document in enumerate(documents, start=1):
        records = sorted(
            bee_occurrences(document, mot_frame - 1),
            key=lambda item: item.track_id,
        )
        for record in records:
            x1, y1, x2, y2 = record.rect
            rows.append(
                f"{mot_frame},{record.track_id},{x1:.3f},{y1:.3f},"
                f"{x2 - x1:.3f},{y2 - y1:.3f},1,0,1.0"
            )
    (mot_dir / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (mot_dir / "classes.txt").write_text("bee\n", encoding="utf-8")

    with Image.open(image_paths[0]) as image:
        width, height = image.size
    config = configparser.ConfigParser()
    config.optionxform = str
    config["Sequence"] = {
        "name": folder.name,
        "imDir": "..",
        "frameRate": "6",
        "seqLength": str(len(image_paths)),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": image_paths[0].suffix.lower(),
    }
    with (mot_dir / "seqinfo.ini").open("w", encoding="utf-8", newline="\n") as file:
        config.write(file, space_around_delimiters=False)


def snapshot_files(folder: Path, image_paths: list[Path]) -> dict[Path, bytes | None]:
    paths = [image_path.with_suffix(".json") for image_path in image_paths]
    paths.extend(
        [
            folder / STATE_NAME,
            folder / TRAJECTORY_STATE_NAME,
            folder / "MOT" / "gt.txt",
            folder / "MOT" / "classes.txt",
            folder / "MOT" / "seqinfo.ini",
        ]
    )
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def restore_snapshot(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".undo.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
