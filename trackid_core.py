from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Occurrence:
    track_id: int
    frame_index: int
    shape_index: int
    rect: Rect


@dataclass(frozen=True)
class ReviewEvent:
    track_id: int
    first_frame_index: int
    rect: Rect
    signature: str


@dataclass(frozen=True)
class Candidate:
    track_id: int
    last_frame_index: int
    rect: Rect
    distance: float


@dataclass(frozen=True)
class TrackMetrics:
    occurrence_count: int
    first_frame_index: int
    last_frame_index: int
    missing_frames: tuple[int, ...]
    duplicate_frames: tuple[int, ...]
    anomaly_frames: tuple[int, ...]
    max_step_distance: float
    max_step_ratio: float
    max_area_ratio: float
    reasons: tuple[str, ...]

    @property
    def suspicious(self) -> bool:
        return bool(self.reasons)


def rect_bounds(points: Iterable[Iterable[float]]) -> Rect:
    values = [tuple(map(float, point)) for point in points]
    if not values:
        raise ValueError("矩形点不能为空")
    xs = [point[0] for point in values]
    ys = [point[1] for point in values]
    return min(xs), min(ys), max(xs), max(ys)


def rect_center(rect: Rect) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0


def rect_area(rect: Rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def point_in_rect(point: tuple[float, float], rect: Rect) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def bee_occurrences(document: dict, frame_index: int) -> list[Occurrence]:
    records = []
    for shape_index, shape in enumerate(document.get("shapes", [])):
        if shape.get("shape_type") != "rectangle" or shape.get("label") != "bee":
            continue
        track_id = shape.get("group_id")
        if not isinstance(track_id, int) or track_id <= 0:
            continue
        try:
            rect = rect_bounds(shape.get("points", []))
        except (TypeError, ValueError, IndexError):
            continue
        records.append(Occurrence(track_id, frame_index, shape_index, rect))
    return records


def build_tracks(documents: list[dict]) -> dict[int, list[Occurrence]]:
    tracks: dict[int, list[Occurrence]] = {}
    for frame_index, document in enumerate(documents):
        for occurrence in bee_occurrences(document, frame_index):
            tracks.setdefault(occurrence.track_id, []).append(occurrence)
    return tracks


def event_signature(image_name: str, rect: Rect) -> str:
    coords = ",".join(f"{value:.2f}" for value in rect)
    return f"{image_name}|{coords}"


def review_events(
    documents: list[dict],
    image_names: list[str],
    reviewed_signatures: set[str] | None = None,
    forced_track_ids: set[int] | None = None,
) -> list[ReviewEvent]:
    reviewed = reviewed_signatures or set()
    forced = forced_track_ids or set()
    events = []
    for track_id, occurrences in build_tracks(documents).items():
        first = min(occurrences, key=lambda item: item.frame_index)
        if first.frame_index == 0 and track_id not in forced:
            continue
        signature = event_signature(image_names[first.frame_index], first.rect)
        if signature in reviewed and track_id not in forced:
            continue
        events.append(ReviewEvent(track_id, first.frame_index, first.rect, signature))
    return sorted(events, key=lambda item: (item.first_frame_index, item.track_id))


def new_id_queue_status(
    documents: list[dict],
    image_names: list[str],
    reviewed_signatures: set[str] | None = None,
    forced_track_ids: set[int] | None = None,
) -> str:
    """说明新 ID 队列为空或仍有任务的原因。"""
    if not build_tracks(documents):
        return "no_tracks"
    if review_events(
        documents,
        image_names,
        reviewed_signatures,
        forced_track_ids,
    ):
        return "pending"
    if review_events(documents, image_names, set(), forced_track_ids):
        return "complete"
    return "no_later_ids"


def trajectory_review_events(
    documents: list[dict],
    image_names: list[str],
) -> list[ReviewEvent]:
    """为当前区段中的每个 Track ID 创建一个轨迹复查项目。"""
    events = []
    for track_id, occurrences in build_tracks(documents).items():
        first = min(occurrences, key=lambda item: item.frame_index)
        events.append(
            ReviewEvent(
                track_id,
                first.frame_index,
                first.rect,
                event_signature(image_names[first.frame_index], first.rect),
            )
        )
    return sorted(events, key=lambda item: item.track_id)


def analyze_track(occurrences: list[Occurrence]) -> TrackMetrics:
    """计算只用于辅助复查的轨迹异常指标，不自动修改标注。"""
    if not occurrences:
        raise ValueError("轨迹不能为空")
    ordered = sorted(occurrences, key=lambda item: (item.frame_index, item.shape_index))
    frame_counts: dict[int, int] = {}
    representatives: dict[int, Occurrence] = {}
    for occurrence in ordered:
        frame_counts[occurrence.frame_index] = (
            frame_counts.get(occurrence.frame_index, 0) + 1
        )
        representatives.setdefault(occurrence.frame_index, occurrence)

    first_frame = min(frame_counts)
    last_frame = max(frame_counts)
    missing = tuple(
        frame_index
        for frame_index in range(first_frame, last_frame + 1)
        if frame_index not in frame_counts
    )
    duplicates = tuple(
        frame_index
        for frame_index, count in sorted(frame_counts.items())
        if count > 1
    )

    max_distance = 0.0
    max_step_ratio = 0.0
    max_area_ratio = 1.0
    anomaly_frames = set(missing) | set(duplicates)
    unique_occurrences = [
        representatives[frame_index] for frame_index in sorted(representatives)
    ]
    for left, right in zip(unique_occurrences, unique_occurrences[1:]):
        left_center = rect_center(left.rect)
        right_center = rect_center(right.rect)
        distance = hypot(
            right_center[0] - left_center[0],
            right_center[1] - left_center[1],
        )
        left_diagonal = hypot(
            left.rect[2] - left.rect[0],
            left.rect[3] - left.rect[1],
        )
        right_diagonal = hypot(
            right.rect[2] - right.rect[0],
            right.rect[3] - right.rect[1],
        )
        step_ratio = distance / max(1.0, (left_diagonal + right_diagonal) / 2.0)
        left_area = max(1.0, rect_area(left.rect))
        right_area = max(1.0, rect_area(right.rect))
        area_ratio = max(left_area, right_area) / min(left_area, right_area)
        max_distance = max(max_distance, distance)
        max_step_ratio = max(max_step_ratio, step_ratio)
        max_area_ratio = max(max_area_ratio, area_ratio)
        if step_ratio > 3.0 or area_ratio > 2.5:
            anomaly_frames.add(right.frame_index)

    reasons = []
    if len(unique_occurrences) == 1:
        reasons.append("仅出现一帧")
    if missing:
        reasons.append(f"中间缺失 {len(missing)} 帧")
    if duplicates:
        reasons.append(f"同帧重复 {len(duplicates)} 处")
    if max_step_ratio > 3.0:
        reasons.append(f"位移突变 {max_step_ratio:.1f}×框尺寸")
    if max_area_ratio > 2.5:
        reasons.append(f"框面积突变 {max_area_ratio:.1f}×")

    return TrackMetrics(
        occurrence_count=len(ordered),
        first_frame_index=first_frame,
        last_frame_index=last_frame,
        missing_frames=missing,
        duplicate_frames=duplicates,
        anomaly_frames=tuple(sorted(anomaly_frames)),
        max_step_distance=max_distance,
        max_step_ratio=max_step_ratio,
        max_area_ratio=max_area_ratio,
        reasons=tuple(reasons),
    )


def analyze_tracks(
    tracks: dict[int, list[Occurrence]],
) -> dict[int, TrackMetrics]:
    return {
        track_id: analyze_track(occurrences)
        for track_id, occurrences in tracks.items()
        if occurrences
    }


def candidates_for_event(
    tracks: dict[int, list[Occurrence]],
    event: ReviewEvent,
) -> list[Candidate]:
    target_center = rect_center(event.rect)
    candidates = []
    new_frames = {item.frame_index for item in tracks.get(event.track_id, [])}
    for track_id, occurrences in tracks.items():
        if track_id == event.track_id:
            continue
        old_frames = {item.frame_index for item in occurrences}
        if old_frames & new_frames:
            continue
        previous = [item for item in occurrences if item.frame_index < event.first_frame_index]
        if not previous:
            continue
        last = max(previous, key=lambda item: item.frame_index)
        old_center = rect_center(last.rect)
        distance = hypot(target_center[0] - old_center[0], target_center[1] - old_center[1])
        candidates.append(Candidate(track_id, last.frame_index, last.rect, distance))
    return sorted(candidates, key=lambda item: (item.distance, -item.last_frame_index, item.track_id))


def collision_frames(documents: list[dict], old_id: int, new_id: int) -> list[int]:
    collisions = []
    for frame_index, document in enumerate(documents):
        ids = {
            item.track_id
            for item in bee_occurrences(document, frame_index)
        }
        if old_id in ids and new_id in ids:
            collisions.append(frame_index)
    return collisions


def displace_id_on_frames(
    documents: list[dict],
    track_id: int,
    frame_indices: list[int],
) -> tuple[int, int]:
    """把指定冲突帧中的整组标注临时移到最大 ID 之后。"""
    current_max = max(
        (
            group_id
            for document in documents
            for shape in document.get("shapes", [])
            if isinstance((group_id := shape.get("group_id")), int)
            and group_id > 0
        ),
        default=0,
    )
    displaced_id = current_max + 1
    changes = 0
    for frame_index in frame_indices:
        for shape in documents[frame_index].get("shapes", []):
            if shape.get("group_id") == track_id:
                shape["group_id"] = displaced_id
                changes += 1
    if changes == 0:
        raise ValueError("冲突帧中没有找到需要拆出的旧 ID")
    return displaced_id, changes


def remap_document(document: dict, old_id: int, new_id: int) -> int:
    if old_id <= 0 or new_id <= 0:
        raise ValueError("ID 必须大于 0")
    if old_id == new_id:
        raise ValueError("旧 ID 不能与新 ID 相同")
    merged_id = old_id - 1 if old_id > new_id else old_id
    changes = 0
    for shape in document.get("shapes", []):
        group_id = shape.get("group_id")
        if not isinstance(group_id, int):
            continue
        if group_id == new_id:
            replacement = merged_id
        elif group_id > new_id:
            replacement = group_id - 1
        else:
            replacement = group_id
        if replacement != group_id:
            shape["group_id"] = replacement
            changes += 1
    return changes


def split_track_from_frame(
    documents: list[dict],
    track_id: int,
    start_frame_index: int,
) -> tuple[int, int]:
    """把指定 ID 从某帧开始的轨迹拆到当前最大 ID 之后。"""
    if track_id <= 0:
        raise ValueError("ID 必须大于 0")
    if not 0 <= start_frame_index < len(documents):
        raise ValueError("起始帧超出范围")

    current_max = max(
        (
            group_id
            for document in documents
            for shape in document.get("shapes", [])
            if isinstance((group_id := shape.get("group_id")), int)
            and group_id > 0
        ),
        default=0,
    )
    new_id = current_max + 1
    changes = 0
    for document in documents[start_frame_index:]:
        for shape in document.get("shapes", []):
            if shape.get("group_id") == track_id:
                shape["group_id"] = new_id
                changes += 1
    if changes == 0:
        raise ValueError(f"从指定帧开始没有找到 ID {track_id}")
    return new_id, changes


def change_track_id_in_range(
    documents: list[dict],
    old_track_id: int,
    new_track_id: int,
    start_frame_index: int,
    end_frame_index: int,
) -> int:
    """把闭区间内同一 group_id 的检测框和关联关键点整体换成新 ID。"""
    if old_track_id <= 0 or new_track_id <= 0:
        raise ValueError("ID 必须大于 0")
    if old_track_id == new_track_id:
        raise ValueError("原 ID 不能与新 ID 相同")
    if not 0 <= start_frame_index <= end_frame_index < len(documents):
        raise ValueError("修改帧范围超出区段")

    changes = 0
    for document in documents[start_frame_index:end_frame_index + 1]:
        for shape in document.get("shapes", []):
            if shape.get("group_id") == old_track_id:
                shape["group_id"] = new_track_id
                changes += 1
    if changes == 0:
        raise ValueError(
            f"第 {start_frame_index + 1} 至 {end_frame_index + 1} 张"
            f"没有找到 ID {old_track_id}"
        )
    return changes


def add_bee_rectangle(document: dict, track_id: int, rect: Rect) -> int:
    if track_id <= 0:
        raise ValueError("ID 必须大于 0")
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        raise ValueError("检测框尺寸无效")
    template = next(
        (
            shape
            for shape in document.get("shapes", [])
            if shape.get("label") == "bee"
            and shape.get("shape_type") == "rectangle"
        ),
        None,
    )
    shape = dict(template) if template is not None else {
        "kie_linking": [],
        "label": "bee",
        "score": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
    }
    shape.update(
        {
            "label": "bee",
            "shape_type": "rectangle",
            "group_id": track_id,
            "points": [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
        }
    )
    document.setdefault("shapes", []).append(shape)
    return len(document["shapes"]) - 1


def delete_occurrence_with_points(
    document: dict,
    occurrence: Occurrence,
) -> int:
    kept = []
    deleted = 0
    for shape_index, shape in enumerate(document.get("shapes", [])):
        delete_shape = shape_index == occurrence.shape_index
        delete_shape = delete_shape or (
            shape.get("group_id") == occurrence.track_id
            and shape.get("label") in {"head", "tail"}
        )
        if delete_shape:
            deleted += 1
        else:
            kept.append(shape)
    document["shapes"] = kept
    return deleted


def change_occurrence_track_id(
    document: dict,
    occurrence: Occurrence,
    new_track_id: int,
) -> int:
    """修改当前检测框及其关联 head/tail 的 ID。"""
    if new_track_id <= 0:
        raise ValueError("ID 必须大于 0")
    shapes = document.get("shapes", [])
    if not 0 <= occurrence.shape_index < len(shapes):
        raise ValueError("检测框已不存在")

    changes = 0
    for shape_index, shape in enumerate(shapes):
        is_rectangle = shape_index == occurrence.shape_index
        is_related_point = (
            shape.get("group_id") == occurrence.track_id
            and shape.get("label") in {"head", "tail"}
        )
        if (is_rectangle or is_related_point) and shape.get("group_id") != new_track_id:
            shape["group_id"] = new_track_id
            changes += 1
    return changes


def transform_occurrence_with_points(
    document: dict,
    occurrence: Occurrence,
    new_rect: Rect,
) -> int:
    """修改检测框，并让同 group_id 的 head/tail 保持相对框位置。"""
    x1, y1, x2, y2 = map(float, new_rect)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("检测框尺寸无效")
    shapes = document.get("shapes", [])
    if not 0 <= occurrence.shape_index < len(shapes):
        raise ValueError("检测框已不存在")

    old_x1, old_y1, old_x2, old_y2 = occurrence.rect
    old_width = max(1e-6, old_x2 - old_x1)
    old_height = max(1e-6, old_y2 - old_y1)
    new_width = x2 - x1
    new_height = y2 - y1

    rectangle = shapes[occurrence.shape_index]
    rectangle["points"] = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]
    changes = 1
    for shape in shapes:
        if (
            shape.get("group_id") != occurrence.track_id
            or shape.get("label") not in {"head", "tail"}
        ):
            continue
        transformed = []
        for point in shape.get("points", []):
            px, py = map(float, point[:2])
            relative_x = (px - old_x1) / old_width
            relative_y = (py - old_y1) / old_height
            transformed.append(
                [
                    x1 + relative_x * new_width,
                    y1 + relative_y * new_height,
                ]
            )
        if transformed:
            shape["points"] = transformed
            changes += 1
    return changes


def find_occurrence(
    tracks: dict[int, list[Occurrence]],
    track_id: int,
    frame_index: int,
) -> Occurrence | None:
    for occurrence in tracks.get(track_id, []):
        if occurrence.frame_index == frame_index:
            return occurrence
    return None


def last_occurrence_before(
    tracks: dict[int, list[Occurrence]],
    track_id: int,
    frame_index: int,
) -> Occurrence | None:
    values = [
        occurrence
        for occurrence in tracks.get(track_id, [])
        if occurrence.frame_index < frame_index
    ]
    return max(values, key=lambda item: item.frame_index) if values else None
