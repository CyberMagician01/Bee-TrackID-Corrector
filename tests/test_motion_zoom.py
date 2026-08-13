from app import TrackIdCorrector


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeWidget:
    def __init__(self):
        self.cursor = None

    def configure(self, cursor):
        self.cursor = cursor


class FakeRoot:
    def after(self, _delay, _callback):
        return 1

    def after_cancel(self, _job):
        return None


def test_motion_zoom_can_grow_beyond_canvas_width():
    size, scale, x, y, pan = TrackIdCorrector._motion_fit_geometry(
        (600, 600),
        (1200, 600),
        2.0,
    )
    assert size == (1200, 600)
    assert scale == 1.0
    assert x == -300
    assert y == 0
    assert pan == (0.0, 0.0)


def test_motion_zoom_one_always_shows_the_full_image():
    size, scale, x, y, pan = TrackIdCorrector._motion_fit_geometry(
        (600, 600),
        (1200, 600),
        1.0,
        (200.0, -100.0),
    )
    assert size == (600, 300)
    assert scale == 0.5
    assert x == 0
    assert y == 150
    assert pan == (0.0, 0.0)


def test_motion_pan_is_clamped_to_the_full_image_edges():
    size, scale, x, y, pan = TrackIdCorrector._motion_fit_geometry(
        (600, 600),
        (1200, 600),
        3.0,
        (1000.0, -1000.0),
    )
    assert size == (1800, 900)
    assert scale == 1.5
    assert pan == (400.0, -100.0)
    assert x == -1200
    assert y == 0


def test_initial_view_focuses_target_but_keeps_full_image_available():
    zoom, pan = TrackIdCorrector._motion_initial_view(
        (600, 600),
        (1200, 600),
        (400, 150, 800, 450),
    )
    assert 1.0 < zoom <= 6.0
    assert pan == (0.0, 0.0)


def test_middle_drag_pans_while_draw_mode_stays_enabled():
    value = TrackIdCorrector.__new__(TrackIdCorrector)
    canvas = FakeWidget()
    value.motion_canvas = canvas
    value.raw_canvas = FakeWidget()
    value.motion_transform = (2.0, 0.0, 0.0, (0, 0, 1000, 800), 0)
    value.raw_motion_transform = value.motion_transform
    value.motion_draw_mode = FakeVar(True)
    value.motion_box_start = None
    value.motion_edit_mode = None
    value.motion_drag_last = None
    value.motion_drag_total = 0.0
    value.motion_drag_scale = 1.0
    value.motion_pan_offset = [0.0, 0.0]
    value.motion_left_render_key = None
    value.motion_pan_render_job = None
    value.root = FakeRoot()
    value._stop_motion = lambda: None
    value._render_motion = lambda force_left=False: None

    start = type("Event", (), {"widget": canvas, "x": 100, "y": 100})()
    move = type("Event", (), {"widget": canvas, "x": 140, "y": 120})()

    assert value._start_motion_middle_drag(start) == "break"
    value._drag_motion_view(move)
    assert value.motion_box_start is None
    assert value.motion_pan_offset == [-20.0, -10.0]
    assert value._end_motion_middle_drag(move) == "break"
    assert value.motion_draw_mode.get() is True
    assert canvas.cursor == "crosshair"


def test_search_chooses_the_nearest_id_occurrence():
    occurrences = [
        type("Occurrence", (), {"frame_index": 2})(),
        type("Occurrence", (), {"frame_index": 8})(),
        type("Occurrence", (), {"frame_index": 15})(),
    ]
    selected = TrackIdCorrector._nearest_occurrence(occurrences, 10)
    assert selected.frame_index == 8


def test_ctrl_f_does_not_advance_the_raw_frame():
    value = TrackIdCorrector.__new__(TrackIdCorrector)
    calls = []
    value.next_raw_frame = lambda: calls.append("next")

    assert value._on_motion_next_raw_shortcut(
        type("Event", (), {"state": 0x4})()
    ) is None
    assert calls == []
    assert value._on_motion_next_raw_shortcut(
        type("Event", (), {"state": 0})()
    ) == "break"
    assert calls == ["next"]


def test_raw_frame_navigation_keeps_sampled_frame_corresponding():
    value = TrackIdCorrector.__new__(TrackIdCorrector)
    value.motion_sequence = list(range(15))
    value.motion_source_frames = list(range(8315, 8390, 5))
    value.motion_index = 5
    value.raw_start_frame = 8315
    value.raw_end_frame = 8389
    value.raw_frame_index = 8381
    value._stop_motion = lambda: None
    renders = []
    value.open_motion_window = lambda force_left=False: renders.append(
        force_left
    )

    value.next_raw_frame()

    assert value.raw_frame_index == 8382
    assert value.motion_index == 13
    assert value.motion_source_frames[value.motion_index] == 8380
    assert renders == [False]

    value.previous_raw_frame()

    assert value.raw_frame_index == 8381
    assert value.motion_index == 13
    assert renders == [False, False]


def test_ctrl_box_move_is_clamped_inside_image():
    moved = TrackIdCorrector._move_rect_within_image(
        (10, 20, 30, 50),
        -50,
        100,
        (100, 80),
    )
    assert moved == (0.0, 50, 20.0, 80)


def test_ctrl_box_corner_resize_keeps_minimum_size():
    resized = TrackIdCorrector._resize_rect_within_image(
        (10, 20, 30, 50),
        "nw",
        (100, 100),
        (120, 100),
    )
    assert resized == (26.0, 46.0, 30, 50)


def test_geometry_refresh_keeps_the_same_review_event():
    events = [
        type("Event", (), {"track_id": 90, "first_frame_index": 2})(),
        type("Event", (), {"track_id": 340, "first_frame_index": 7})(),
        type("Event", (), {"track_id": 400, "first_frame_index": 9})(),
    ]
    index = TrackIdCorrector._event_index_for_identity(
        events,
        track_id=340,
        first_frame_index=7,
        fallback_index=0,
    )
    assert index == 1


def test_b_shortcut_toggles_trajectory_visibility():
    value = TrackIdCorrector.__new__(TrackIdCorrector)
    value.motion_show_trajectory = FakeVar(True)
    value.motion_left_render_key = "cached"
    calls = []
    value._render_motion = lambda force_left=False: calls.append(force_left)

    event = type("Event", (), {"state": 0})()
    assert value._on_motion_toggle_trajectory_shortcut(event) == "break"
    assert value.motion_show_trajectory.get() is False
    assert value.motion_left_render_key is None
    assert calls == [True]


def test_h_shortcut_toggles_detection_box_visibility():
    value = TrackIdCorrector.__new__(TrackIdCorrector)
    value.motion_show_boxes = FakeVar(True)
    value.motion_left_render_key = "cached"
    calls = []
    value._render_motion = lambda force_left=False: calls.append(force_left)

    event = type("Event", (), {"state": 0})()
    assert value._on_motion_toggle_boxes_shortcut(event) == "break"
    assert value.motion_show_boxes.get() is False
    assert value.motion_left_render_key is None
    assert calls == [True]
