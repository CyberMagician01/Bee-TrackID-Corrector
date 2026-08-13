from trackid_core import (
    add_bee_rectangle,
    analyze_track,
    bee_occurrences,
    build_tracks,
    candidates_for_event,
    change_occurrence_track_id,
    change_track_id_in_range,
    collision_frames,
    delete_occurrence_with_points,
    displace_id_on_frames,
    remap_document,
    review_events,
    split_track_from_frame,
    trajectory_review_events,
    transform_occurrence_with_points,
)


def rectangle(track_id, x):
    return {
        "label": "bee",
        "shape_type": "rectangle",
        "group_id": track_id,
        "points": [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]],
    }


def point(track_id):
    return {
        "label": "head",
        "shape_type": "point",
        "group_id": track_id,
        "points": [[5, 5]],
    }


def test_event_candidate_merge_and_renumber():
    documents = [
        {"shapes": [rectangle(1, 0), rectangle(2, 100)]},
        {"shapes": [rectangle(1, 2), rectangle(2, 102)]},
        {"shapes": [rectangle(3, 4), rectangle(2, 104), rectangle(4, 200), point(3)]},
    ]
    events = review_events(documents, ["f0.jpg", "f1.jpg", "f2.jpg"])
    assert [event.track_id for event in events] == [3, 4]

    tracks = build_tracks(documents)
    candidates = candidates_for_event(tracks, events[0])
    assert [candidate.track_id for candidate in candidates] == [1]
    assert collision_frames(documents, 1, 3) == []

    changed = sum(remap_document(document, 1, 3) for document in documents)
    assert changed == 3
    assert documents[2]["shapes"][0]["group_id"] == 1
    assert documents[2]["shapes"][2]["group_id"] == 3
    assert documents[2]["shapes"][3]["group_id"] == 1


def test_forced_new_id_is_reviewed_even_on_first_frame_or_already_reviewed():
    documents = [
        {"shapes": [rectangle(9, 0)]},
        {"shapes": [rectangle(9, 2)]},
    ]
    signature = "f0.jpg|0.00,0.00,10.00,10.00"

    assert review_events(documents, ["f0.jpg", "f1.jpg"], {signature}) == []
    events = review_events(
        documents,
        ["f0.jpg", "f1.jpg"],
        {signature},
        {9},
    )

    assert [event.track_id for event in events] == [9]
    assert events[0].first_frame_index == 0


def test_collision_rejected_by_detection():
    documents = [
        {"shapes": [rectangle(1, 0), rectangle(3, 50)]},
    ]
    assert collision_frames(documents, 1, 3) == [0]


def test_split_track_from_frame_to_last_id():
    documents = [
        {"shapes": [rectangle(1, 0), rectangle(3, 50)]},
        {"shapes": [rectangle(1, 2), point(1)]},
        {"shapes": [rectangle(1, 4), point(1)]},
    ]

    new_id, changed = split_track_from_frame(documents, 1, 1)

    assert new_id == 4
    assert changed == 4
    assert documents[0]["shapes"][0]["group_id"] == 1
    assert documents[1]["shapes"][0]["group_id"] == 4
    assert documents[1]["shapes"][1]["group_id"] == 4
    assert documents[2]["shapes"][0]["group_id"] == 4


def test_change_track_id_in_range_includes_current_and_related_points():
    documents = [
        {"shapes": [rectangle(3, 0), point(3)]},
        {"shapes": [rectangle(3, 2), point(3), rectangle(8, 100)]},
        {"shapes": [rectangle(3, 4), point(3)]},
        {"shapes": [rectangle(3, 6), point(3)]},
    ]

    changed = change_track_id_in_range(documents, 3, 9, 1, 2)

    assert changed == 4
    assert [shape["group_id"] for shape in documents[0]["shapes"]] == [3, 3]
    assert [shape["group_id"] for shape in documents[1]["shapes"]] == [9, 9, 8]
    assert [shape["group_id"] for shape in documents[2]["shapes"]] == [9, 9]
    assert [shape["group_id"] for shape in documents[3]["shapes"]] == [3, 3]


def test_merge_accepts_higher_old_id_after_manual_split():
    documents = [
        {"shapes": [rectangle(60, 0), point(60)]},
        {"shapes": [rectangle(32, 2), point(32), rectangle(40, 100)]},
    ]
    events = review_events(documents, ["f0.jpg", "f1.jpg"])
    tracks = build_tracks(documents)

    candidates = candidates_for_event(tracks, events[0])
    assert [candidate.track_id for candidate in candidates] == [60]

    changed = sum(remap_document(document, 60, 32) for document in documents)
    assert changed == 5
    assert documents[0]["shapes"][0]["group_id"] == 59
    assert documents[0]["shapes"][1]["group_id"] == 59
    assert documents[1]["shapes"][0]["group_id"] == 59
    assert documents[1]["shapes"][1]["group_id"] == 59
    assert documents[1]["shapes"][2]["group_id"] == 39


def test_add_and_delete_detection_box_with_keypoints():
    document = {
        "shapes": [
            rectangle(1, 0),
            point(1),
            {**point(1), "label": "tail"},
        ]
    }
    added_index = add_bee_rectangle(document, 2, (20, 30, 40, 60))
    document["shapes"].append({**point(2), "label": "tail"})

    records = bee_occurrences(document, 0)
    added = next(record for record in records if record.shape_index == added_index)
    assert added.track_id == 2
    assert added.rect == (20.0, 30.0, 40.0, 60.0)

    deleted = delete_occurrence_with_points(document, added)
    assert deleted == 2
    assert [shape["group_id"] for shape in document["shapes"]] == [1, 1, 1]


def test_transform_box_moves_and_scales_linked_keypoints():
    document = {
        "shapes": [
            rectangle(1, 0),
            point(1),
            {**point(1), "label": "tail"},
            rectangle(2, 100),
        ]
    }
    occurrence = bee_occurrences(document, 0)[0]

    changed = transform_occurrence_with_points(
        document,
        occurrence,
        (20, 30, 40, 50),
    )

    assert changed == 3
    assert document["shapes"][0]["points"] == [
        [20.0, 30.0],
        [40.0, 30.0],
        [40.0, 50.0],
        [20.0, 50.0],
    ]
    assert document["shapes"][1]["points"] == [[30.0, 40.0]]
    assert document["shapes"][2]["points"] == [[30.0, 40.0]]
    assert document["shapes"][3] == rectangle(2, 100)


def test_collision_old_box_moves_to_last_id_before_merge():
    documents = [
        {"shapes": [rectangle(1, 0)]},
        {
            "shapes": [
                rectangle(1, 10),
                point(1),
                rectangle(3, 12),
                point(3),
                rectangle(4, 100),
            ]
        },
    ]

    temporary_id, displaced = displace_id_on_frames(documents, 1, [1])
    changed = sum(remap_document(document, 1, 3) for document in documents)

    assert temporary_id == 5
    assert displaced == 2
    assert changed == 5
    assert [shape["group_id"] for shape in documents[1]["shapes"]] == [
        4,
        4,
        1,
        1,
        3,
    ]
    events = review_events(documents, ["f0.jpg", "f1.jpg"])
    assert 4 in [event.track_id for event in events]


def test_change_occurrence_track_id_updates_related_points_only():
    document = {
        "shapes": [
            rectangle(3, 10),
            point(3),
            {
                "label": "tail",
                "shape_type": "point",
                "group_id": 3,
                "points": [[8, 8]],
            },
            rectangle(4, 100),
        ]
    }
    occurrence = bee_occurrences(document, 0)[0]

    changed = change_occurrence_track_id(document, occurrence, 9)

    assert changed == 3
    assert [shape["group_id"] for shape in document["shapes"]] == [9, 9, 9, 4]


def test_trajectory_review_events_include_ids_from_first_frame():
    documents = [
        {"shapes": [rectangle(1, 0)]},
        {"shapes": [rectangle(1, 2), rectangle(2, 50)]},
    ]

    events = trajectory_review_events(documents, ["f0.jpg", "f1.jpg"])

    assert [event.track_id for event in events] == [1, 2]
    assert [event.first_frame_index for event in events] == [0, 1]


def test_trajectory_metrics_detect_missing_jump_and_area_change():
    documents = [
        {"shapes": [rectangle(1, 0)]},
        {"shapes": []},
        {
            "shapes": [
                {
                    **rectangle(1, 100),
                    "points": [[100, 0], [140, 0], [140, 40], [100, 40]],
                }
            ]
        },
    ]
    occurrences = build_tracks(documents)[1]

    metrics = analyze_track(occurrences)

    assert metrics.missing_frames == (1,)
    assert metrics.anomaly_frames == (1, 2)
    assert metrics.max_step_ratio > 3.0
    assert metrics.max_area_ratio > 2.5
    assert metrics.suspicious is True
