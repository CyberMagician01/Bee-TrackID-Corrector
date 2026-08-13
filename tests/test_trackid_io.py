from trackid_io import (
    load_deferred_new_ids,
    load_trajectory_reviews,
    save_deferred_new_ids,
    save_trajectory_reviews,
)


def test_trajectory_state_preserves_deferred_ids(tmp_path):
    save_deferred_new_ids(tmp_path, {12, 18})
    save_trajectory_reviews(tmp_path, {"frame.jpg|1.00,2.00,3.00,4.00": "passed"})

    assert load_deferred_new_ids(tmp_path) == {12, 18}
    assert load_trajectory_reviews(tmp_path) == {
        "frame.jpg|1.00,2.00,3.00,4.00": "passed"
    }


def test_deferred_ids_update_preserves_trajectory_reviews(tmp_path):
    reviews = {"frame.jpg|1.00,2.00,3.00,4.00": "issue"}
    save_trajectory_reviews(tmp_path, reviews)
    save_deferred_new_ids(tmp_path, {21})

    assert load_trajectory_reviews(tmp_path) == reviews
    assert load_deferred_new_ids(tmp_path) == {21}
