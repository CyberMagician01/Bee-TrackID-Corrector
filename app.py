from __future__ import annotations

import colorsys
import copy
import ctypes
import json
import re
import sys
import tkinter as tk
from bisect import bisect_right
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from trackid_core import (
    Candidate,
    ReviewEvent,
    TrackMetrics,
    add_bee_rectangle,
    analyze_tracks,
    bee_occurrences,
    build_tracks,
    candidates_for_event,
    change_occurrence_track_id,
    change_track_id_in_range,
    collision_frames,
    delete_occurrence_with_points,
    displace_id_on_frames,
    event_signature,
    find_occurrence,
    last_occurrence_before,
    point_in_rect,
    rect_area,
    rect_center,
    remap_document,
    review_events,
    split_track_from_frame,
    trajectory_review_events,
    transform_occurrence_with_points,
)
from trackid_io import (
    create_section_backup,
    export_mot,
    image_files,
    load_deferred_new_ids,
    load_document,
    load_reviewed,
    load_trajectory_reviews,
    restore_snapshot,
    save_deferred_new_ids,
    save_documents,
    save_reviewed,
    save_trajectory_reviews,
    snapshot_files,
)


SOURCE_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else SOURCE_DIR
SETTINGS_PATH = APP_DIR / "settings.json"
ICON_PATH = SOURCE_DIR.parent / "bee_keypoint_annotator" / "assets" / "bee_annotator_icon.png"
if IS_FROZEN:
    ICON_PATH = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "assets" / "bee_annotator_icon.png"


def resolve_app_path(value: str | Path, fallback: str) -> Path:
    path = Path(str(value or fallback)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


def portable_setting_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_DIR.resolve()))
    except (OSError, ValueError):
        return str(path)


REFERENCE_FIRST = "第一帧同位置"
REFERENCE_PREVIOUS = "上一帧同位置"
REFERENCE_CANDIDATE = "候选ID最后出现"
MODE_NEW_ID = "新 ID 纠正"
MODE_TRAJECTORY = "室外轨迹复查"
TRAJECTORY_FILTER_ALL = "全部"
TRAJECTORY_FILTER_PENDING = "未检查"
TRAJECTORY_FILTER_SUSPICIOUS = "疑似异常"
TRAJECTORY_FILTER_ISSUE = "有问题"


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


class TrackIdQueryDialog(simpledialog._QueryInteger):
    def body(self, master):
        entry = super().body(master)
        entry.bind("<g>", self._confirm_with_g)
        entry.bind("<G>", self._confirm_with_g)
        return entry

    def _confirm_with_g(self, _event):
        self.ok()
        return "break"


class TrackIdCorrector:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("蜜蜂 Track ID 修正器")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1660, max(1080, screen_width - 60))
        window_height = min(960, max(700, screen_height - 90))
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(
            min(1180, window_width),
            min(720, window_height),
        )
        self.icon = None
        if ICON_PATH.exists():
            try:
                self.icon = tk.PhotoImage(file=str(ICON_PATH))
                self.root.iconphoto(True, self.icon)
            except tk.TclError:
                pass

        self.settings = self._load_settings()
        self.task_root: Path | None = None
        self.all_section_dirs: list[Path] = []
        self.section_dirs: list[Path] = []
        self.section: Path | None = None
        self.images: list[Path] = []
        self.documents: list[dict] = []
        self.tracks = {}
        self.reviewed: set[str] = set()
        self.events: list[ReviewEvent] = []
        self.trajectory_all_events: list[ReviewEvent] = []
        self.trajectory_metrics: dict[int, TrackMetrics] = {}
        self.trajectory_reviews: dict[str, str] = {}
        self.deferred_new_ids: set[int] = set()
        self.outdoor_new_id_handoff = False
        self.event_index = -1
        self.display_frame_index = 0
        self.candidates: list[Candidate] = []
        self.backup_path: Path | None = None
        self.undo_stack: list[tuple[str, dict[Path, bytes | None]]] = []
        self.image_cache: OrderedDict[Path, Image.Image] = OrderedDict()
        self.global_photo = None
        self.current_photo = None
        self.reference_photo = None
        self.motion_photo = None
        self.raw_photo = None
        self.global_transform = None
        self.reference_transform = None
        self.reference_frame_index = 0
        self.redraw_job = None
        self.motion_sequence: list[int] = []
        self.motion_index = 0
        self.motion_event_start_index = 0
        self.raw_play_start_frame = 0
        self.motion_job = None
        self.motion_playing = False
        self.motion_window: tk.Toplevel | None = None
        self.motion_canvas: tk.Canvas | None = None
        self.motion_transform = None
        self.raw_motion_transform = None
        self.motion_pan_offset = [0.0, 0.0]
        self.motion_zoom = 1.0
        self.motion_view_initialized = False
        self.motion_focus_frame_index = None
        self.motion_search_track_id: int | None = None
        self.motion_drag_last = None
        self.motion_drag_total = 0.0
        self.motion_drag_scale = 1.0
        self.motion_pan_render_job = None
        self.motion_context_menu: tk.Menu | None = None
        self.motion_search_entry: ttk.Entry | None = None
        self.motion_review_id_combo: ttk.Combobox | None = None
        self.motion_raw_frame_group: ttk.LabelFrame | None = None
        self.motion_review_bar: ttk.Frame | None = None
        self.motion_views: ttk.Frame | None = None
        self.trajectory_timeline_canvas: tk.Canvas | None = None
        self.motion_draw_mode = tk.BooleanVar(value=False)
        self.motion_draw_button: ttk.Button | None = None
        self.motion_box_start = None
        self.motion_preview_item = None
        self.motion_edit_occurrence = None
        self.motion_edit_mode: str | None = None
        self.motion_edit_start_point = None
        self.motion_edit_original_rect = None
        self.motion_edit_preview_rect = None
        self.raw_canvas: tk.Canvas | None = None
        self.motion_play_button: ttk.Button | None = None
        self.motion_event_start_button: ttk.Button | None = None
        self.global_window: tk.Toplevel | None = None
        self.global_canvas: tk.Canvas | None = None
        self.global_raw_canvas: tk.Canvas | None = None
        self.global_raw_photo = None
        self.global_play_button: ttk.Button | None = None
        self.global_playing = False
        self.global_job = None
        self.global_raw_frame_index = 0
        self.global_raw_capture = None
        self.global_raw_video_path: Path | None = None
        self.global_raw_capture_next_frame: int | None = None
        self.global_raw_image_cache: OrderedDict[int, Image.Image] = OrderedDict()
        self.global_left_render_key = None
        self.motion_source_frames: list[int] = []
        self.sample_gap = 5
        self.raw_frame_index = 0
        self.raw_start_frame = 0
        self.raw_end_frame = 0
        self.raw_video_frame_count = 0
        self.raw_video_root = resolve_app_path(
            self.settings.get("raw_video_root", "原视频"),
            "原视频",
        )
        self.raw_video_path: Path | None = None
        self.raw_capture = None
        self.raw_capture_next_frame: int | None = None
        self.raw_image_cache: OrderedDict[int, Image.Image] = OrderedDict()
        self.motion_left_render_key = None

        self.section_var = tk.StringVar()
        self.work_mode_var = tk.StringVar(value=MODE_NEW_ID)
        self.trajectory_filter_var = tk.StringVar(
            value=TRAJECTORY_FILTER_PENDING
        )
        self.trajectory_progress_var = tk.StringVar(value="")
        self.trajectory_reason_var = tk.StringVar(value="")
        self.event_var = tk.StringVar()
        self.old_id_var = tk.StringVar()
        self.reference_mode = tk.StringVar(
            value=self.settings.get("reference_mode", REFERENCE_FIRST)
        )
        self.reference_frame_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择任务目录或区段目录")
        self.progress_var = tk.StringVar(value="")
        self.frame_var = tk.StringVar(value="")
        self.backup_var = tk.StringVar(value="尚未修改")
        self.show_all_ids = tk.BooleanVar(value=True)
        self.show_class_names = tk.BooleanVar(
            value=bool(self.settings.get("show_class_names", False))
        )
        self.motion_speed = tk.StringVar(value="2 FPS")
        self.motion_loop = tk.BooleanVar(value=True)
        self.motion_follow = tk.BooleanVar(value=False)
        self.motion_show_boxes = tk.BooleanVar(value=True)
        self.motion_show_all_ids = tk.BooleanVar(value=False)
        self.motion_show_trajectory = tk.BooleanVar(value=True)
        self.motion_status_var = tk.StringVar(value="请选择候选旧 ID")
        self.motion_frame_var = tk.StringVar(value="")
        self.motion_zoom_var = tk.StringVar(value="缩放 1.00×")
        try:
            marker_radius = int(self.settings.get("motion_marker_radius", 4))
        except (TypeError, ValueError):
            marker_radius = 4
        self.motion_marker_radius_value = min(20, max(1, marker_radius))
        self.motion_marker_size_var = tk.StringVar(
            value=str(self.motion_marker_radius_value)
        )
        self.motion_marker_size_var.trace_add(
            "write", self._on_motion_marker_size_written
        )
        self.motion_search_id_var = tk.StringVar()
        self.motion_review_id_var = tk.StringVar()
        self.global_speed = tk.StringVar(value="10 FPS")
        self.global_loop = tk.BooleanVar(value=True)
        self.global_status_var = tk.StringVar(value="")

        self._build_style()
        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        default_folder = self.settings.get("default_folder")
        if default_folder and Path(default_folder).exists():
            self.load_task(Path(default_folder))

    def _load_settings(self) -> dict:
        defaults = {"default_folder": "", "reference_mode": REFERENCE_FIRST}
        try:
            defaults.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
        return defaults

    def _save_settings(self) -> None:
        self.settings.update(
            {
                "default_folder": str(self.task_root or self.section or ""),
                "reference_mode": self.reference_mode.get(),
                "show_class_names": self.show_class_names.get(),
                "raw_video_root": portable_setting_path(self.raw_video_root),
                "motion_marker_radius": self._motion_marker_radius(),
            }
        )
        self.settings.pop("motion_skip_similar_frames", None)
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(
            json.dumps(self.settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Danger.TButton", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(toolbar, text="打开任务目录", command=self.open_folder).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(toolbar, text="模式：").pack(side=tk.LEFT, padx=(10, 2))
        self.work_mode_combo = ttk.Combobox(
            toolbar,
            textvariable=self.work_mode_var,
            values=[MODE_NEW_ID, MODE_TRAJECTORY],
            state="readonly",
            width=13,
        )
        self.work_mode_combo.pack(side=tk.LEFT, padx=2)
        self.work_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_work_mode_changed
        )
        ttk.Label(toolbar, text="区段：").pack(side=tk.LEFT, padx=(10, 2))
        self.section_combo = ttk.Combobox(
            toolbar, textvariable=self.section_var, state="readonly", width=24
        )
        self.section_combo.pack(side=tk.LEFT, padx=2)
        self.section_combo.bind("<<ComboboxSelected>>", self._on_section_selected)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        self.previous_event_button = ttk.Button(
            toolbar, text="◀ 上一新ID", command=self.previous_event
        )
        self.previous_event_button.pack(side=tk.LEFT, padx=2)
        self.next_event_button = ttk.Button(
            toolbar, text="下一新ID ▶", command=self.next_event
        )
        self.next_event_button.pack(side=tk.LEFT, padx=2)
        self.event_combo = ttk.Combobox(
            toolbar, textvariable=self.event_var, state="readonly", width=29
        )
        self.event_combo.pack(side=tk.LEFT, padx=4)
        self.event_combo.bind("<<ComboboxSelected>>", self._on_event_selected)
        ttk.Label(toolbar, text="筛选：").pack(side=tk.LEFT, padx=(4, 1))
        self.trajectory_filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.trajectory_filter_var,
            values=[
                TRAJECTORY_FILTER_ALL,
                TRAJECTORY_FILTER_PENDING,
                TRAJECTORY_FILTER_SUSPICIOUS,
                TRAJECTORY_FILTER_ISSUE,
            ],
            state="disabled",
            width=8,
        )
        self.trajectory_filter_combo.pack(side=tk.LEFT, padx=1)
        self.trajectory_filter_combo.bind(
            "<<ComboboxSelected>>", self._on_trajectory_filter_changed
        )
        ttk.Button(toolbar, text="撤销 Ctrl+Z", command=self.undo).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(toolbar, text="帮助 F1", command=self.open_help).pack(
            side=tk.RIGHT, padx=2
        )

        action_bar = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        action_bar.grid(row=1, column=0, sticky=tk.EW)
        self.old_id_label = ttk.Label(action_bar, text="对应旧 ID：")
        self.old_id_label.pack(side=tk.LEFT)
        self.old_id_entry = ttk.Entry(
            action_bar, textvariable=self.old_id_var, width=9, font=("Consolas", 12)
        )
        self.old_id_entry.pack(side=tk.LEFT, padx=3)
        self.old_id_entry.bind("<KeyRelease>", self._on_old_id_changed)
        self.merge_button = ttk.Button(
            action_bar,
            text="合并并重编号 C",
            style="Accent.TButton",
            command=self.merge_current,
        )
        self.merge_button.pack(side=tk.LEFT, padx=4)
        self.keep_button = ttk.Button(
            action_bar,
            text="确认为新目标 V",
            command=self.keep_current,
        )
        self.keep_button.pack(side=tk.LEFT, padx=4)
        ttk.Separator(action_bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        ttk.Button(action_bar, text="Q 上一帧", command=self.previous_frame).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(action_bar, text="回到首次出现", command=self.go_event_frame).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(action_bar, text="下一帧 E", command=self.next_frame).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            action_bar,
            text="局部运动窗口 P",
            command=self.toggle_motion,
        ).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(
            action_bar,
            text="全局鸟瞰 G",
            command=self.open_global_window,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            action_bar,
            text="显示全部ID",
            variable=self.show_all_ids,
            command=self._refresh,
        ).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(
            action_bar,
            text="显示类别名",
            variable=self.show_class_names,
            command=self._on_show_class_names_changed,
        ).pack(side=tk.RIGHT, padx=5)

        main = ttk.Frame(self.root)
        main.grid(row=2, column=0, sticky=tk.NSEW, padx=8, pady=(0, 5))

        self.current_frame_group = ttk.LabelFrame(
            main, text="新 ID 画面（默认第一次出现，Q/E 可切换）"
        )
        self.reference_frame_group = ttk.LabelFrame(
            main, text="参考画面（可选择任意帧，点击框选择旧 ID）"
        )
        current_frame = self.current_frame_group
        reference_frame = self.reference_frame_group
        current_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        reference_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0))
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1, uniform="views")
        main.columnconfigure(1, weight=1, uniform="views")

        self.current_canvas = tk.Canvas(
            current_frame, background="#14171b", highlightthickness=0
        )
        self.current_canvas.pack(fill=tk.BOTH, expand=True)
        self.current_canvas.bind("<Configure>", self._schedule_redraw)

        reference_toolbar = ttk.Frame(reference_frame, padding=(5, 4))
        reference_toolbar.pack(fill=tk.X)
        ttk.Button(
            reference_toolbar,
            text="◀ 上一参考帧 [",
            command=self.previous_reference_frame,
        ).pack(side=tk.LEFT, padx=2)
        self.reference_frame_combo = ttk.Combobox(
            reference_toolbar,
            textvariable=self.reference_frame_var,
            state="readonly",
            width=31,
        )
        self.reference_frame_combo.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3
        )
        self.reference_frame_combo.bind(
            "<<ComboboxSelected>>", self._on_reference_frame_selected
        )
        ttk.Button(
            reference_toolbar,
            text="下一参考帧 ] ▶",
            command=self.next_reference_frame,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            reference_toolbar,
            text="候选ID最后出现",
            command=self.go_candidate_reference_frame,
        ).pack(side=tk.LEFT, padx=(7, 2))
        self.reference_canvas = tk.Canvas(
            reference_frame, background="#14171b", highlightthickness=0, cursor="hand2"
        )
        self.reference_canvas.pack(fill=tk.BOTH, expand=True)
        self.reference_canvas.bind("<Configure>", self._schedule_redraw)
        self.reference_canvas.bind("<Button-1>", self._on_reference_click)

        self.candidate_frame = ttk.LabelFrame(
            self.root, text="建议旧 ID（按位置距离排序，双击选择）"
        )
        candidate_frame = self.candidate_frame
        candidate_frame.grid(row=3, column=0, sticky=tk.EW, padx=8, pady=(0, 5))
        self.candidate_tree = ttk.Treeview(
            candidate_frame,
            columns=("id", "distance", "last"),
            show="headings",
            height=3,
            selectmode="browse",
        )
        self.candidate_tree.heading("id", text="旧 ID")
        self.candidate_tree.heading("distance", text="位置距离")
        self.candidate_tree.heading("last", text="最后出现")
        self.candidate_tree.column("id", width=90, anchor=tk.CENTER)
        self.candidate_tree.column("distance", width=130, anchor=tk.CENTER)
        self.candidate_tree.column("last", width=150, anchor=tk.CENTER)
        self.candidate_tree.pack(fill=tk.X, padx=4, pady=3)
        self.candidate_tree.bind("<<TreeviewSelect>>", self._on_candidate_selected)
        self.candidate_tree.bind("<Double-1>", self._on_candidate_selected)

        footer = ttk.Frame(self.root, padding=(10, 0, 10, 7))
        footer.grid(row=4, column=0, sticky=tk.EW)
        ttk.Label(footer, textvariable=self.progress_var, style="Title.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(footer, textvariable=self.frame_var).pack(side=tk.LEFT, padx=18)
        ttk.Label(footer, textvariable=self.backup_var).pack(side=tk.LEFT, padx=18)
        ttk.Label(footer, textvariable=self.status_var).pack(side=tk.RIGHT)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self._update_mode_ui()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<a>", lambda _event: self.previous_event())
        self.root.bind("<d>", lambda _event: self.next_event())
        self.root.bind("<q>", lambda _event: self.previous_frame())
        self.root.bind("<e>", lambda _event: self.next_frame())
        self.root.bind_all(
            "<c>",
            lambda event: self._run_motion_shortcut(
                event, self.merge_current
            ),
        )
        self.root.bind_all(
            "<v>",
            lambda event: self._run_motion_shortcut(
                event, self.keep_current
            ),
        )
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-o>", lambda _event: self.open_folder())
        self.root.bind("<F1>", lambda _event: self.open_help())
        self.root.bind("<p>", lambda _event: self.toggle_motion())
        self.root.bind("<g>", lambda _event: self.open_global_window())
        self.root.bind("<j>", lambda _event: self.previous_trajectory())
        self.root.bind("<k>", lambda _event: self.next_trajectory())
        self.root.bind("<space>", self._on_trajectory_pass_shortcut)
        self.root.bind("<m>", self._on_trajectory_issue_shortcut)
        self.root.bind("<bracketleft>", lambda _event: self.previous_reference_frame())
        self.root.bind("<bracketright>", lambda _event: self.next_reference_frame())

    def open_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="选择标注员任务目录或单个区段目录",
            initialdir=str(self.task_root or Path.home()),
        )
        if selected:
            self.load_task(Path(selected))

    def _is_trajectory_mode(self) -> bool:
        return self.work_mode_var.get() == MODE_TRAJECTORY

    def _outdoor_sections(self) -> list[Path]:
        return [
            section
            for section in self.all_section_dirs
            if section.name.upper().startswith("A-")
        ]

    def _update_mode_ui(self) -> None:
        trajectory_mode = self._is_trajectory_mode()
        self.previous_event_button.configure(
            text="◀ 上一轨迹 J" if trajectory_mode else "◀ 上一新ID"
        )
        self.next_event_button.configure(
            text="下一轨迹 K ▶" if trajectory_mode else "下一新ID ▶"
        )
        self.merge_button.configure(
            state=tk.DISABLED if trajectory_mode else tk.NORMAL
        )
        self.keep_button.configure(
            state=tk.DISABLED if trajectory_mode else tk.NORMAL
        )
        self.old_id_label.configure(
            text="参考 ID：" if trajectory_mode else "对应旧 ID："
        )
        self.trajectory_filter_combo.configure(
            state="readonly" if trajectory_mode else "disabled"
        )
        self.current_frame_group.configure(
            text=(
                "当前轨迹画面（Q/E 可切换）"
                if trajectory_mode
                else "新 ID 画面（默认第一次出现，Q/E 可切换）"
            )
        )
        self.reference_frame_group.configure(
            text=(
                "轨迹参考画面（可选择任意帧）"
                if trajectory_mode
                else "参考画面（可选择任意帧，点击框选择旧 ID）"
            )
        )
        self.candidate_frame.configure(
            text=(
                "附近参考 ID（可点击画面选择）"
                if trajectory_mode
                else "建议旧 ID（按位置距离排序，双击选择）"
            )
        )
        if self.motion_review_bar is not None and self.motion_views is not None:
            if trajectory_mode:
                if not self.motion_review_bar.winfo_manager():
                    self.motion_review_bar.pack(
                        fill=tk.X,
                        padx=7,
                        pady=(0, 5),
                        before=self.motion_views,
                    )
            else:
                self.motion_review_bar.pack_forget()
        if self.motion_event_start_button is not None:
            self.motion_event_start_button.configure(
                text=(
                    "回到轨迹首次出现 X"
                    if trajectory_mode
                    else "回到新 ID 首次出现 X"
                )
            )
        if self.trajectory_timeline_canvas is not None:
            timeline_parent = self.trajectory_timeline_canvas.master
            if trajectory_mode:
                if not timeline_parent.winfo_manager():
                    timeline_parent.pack(fill=tk.X, padx=7, pady=(0, 7))
            else:
                timeline_parent.pack_forget()

    def _on_work_mode_changed(self, _event=None) -> None:
        self.outdoor_new_id_handoff = False
        self._stop_motion()
        if self._is_trajectory_mode():
            self.trajectory_filter_var.set(TRAJECTORY_FILTER_PENDING)
        self._update_mode_ui()
        if not self.all_section_dirs:
            return
        sections = (
            self._outdoor_sections()
            if self._is_trajectory_mode()
            else list(self.all_section_dirs)
        )
        if not sections:
            messagebox.showwarning(
                "没有室外区段",
                "当前任务目录中没有名称以 A- 开头的室外区段。",
            )
            self.work_mode_var.set(MODE_NEW_ID)
            self._update_mode_ui()
            sections = list(self.all_section_dirs)
        self.section_dirs = sections
        self.section_combo["values"] = [path.name for path in sections]
        if self.section not in sections:
            self.section_var.set(sections[0].name)
            self._load_section(sections[0], clear_undo=True)
        else:
            self.section_var.set(self.section.name)
            self._rebuild_queue(preferred_index=0)
        self._update_mode_ui()

    def load_task(self, selected: Path) -> None:
        self.outdoor_new_id_handoff = False
        if self._is_trajectory_mode():
            self.trajectory_filter_var.set(TRAJECTORY_FILTER_PENDING)
        selected = selected.resolve()
        direct_images = image_files(selected)
        if direct_images:
            sections = [selected]
            task_root = selected.parent
        else:
            sections = sorted(
                child for child in selected.iterdir()
                if (
                    child.is_dir()
                    and "_trackid修正备份_" not in child.name
                    and image_files(child)
                )
            )
            task_root = selected
        if not sections:
            messagebox.showerror("无法打开", "目录中没有包含图片和 JSON 的区段。")
            return
        self.task_root = task_root
        self.all_section_dirs = sections
        self.section_dirs = (
            self._outdoor_sections()
            if self._is_trajectory_mode()
            else list(sections)
        )
        if not self.section_dirs:
            self.work_mode_var.set(MODE_NEW_ID)
            self._update_mode_ui()
            self.section_dirs = list(sections)
        self.section_combo["values"] = [
            path.name for path in self.section_dirs
        ]
        self.section_var.set(self.section_dirs[0].name)
        self._load_section(self.section_dirs[0], clear_undo=True)
        self._save_settings()

    def _on_section_selected(self, _event=None) -> None:
        name = self.section_var.get()
        for section in self.section_dirs:
            if section.name == name:
                self._load_section(section, clear_undo=True)
                break

    def _load_section(self, section: Path, clear_undo: bool) -> None:
        self._stop_motion()
        self._stop_global_video()
        self._release_raw_video()
        self._release_global_raw_video()
        try:
            images = image_files(section)
            documents = [load_document(image) for image in images]
        except (OSError, json.JSONDecodeError, FileNotFoundError) as error:
            messagebox.showerror("加载失败", str(error))
            return
        self.section = section
        self.images = images
        self.documents = documents
        self._update_reference_choices()
        self.reviewed = load_reviewed(section)
        self.trajectory_reviews = load_trajectory_reviews(section)
        self.deferred_new_ids = load_deferred_new_ids(section)
        self.image_cache.clear()
        self.backup_path = None
        if clear_undo:
            self.undo_stack.clear()
        self._rebuild_queue(preferred_index=0)
        self.status_var.set(f"已加载：{section}")
        self.backup_var.set("尚未修改")

    def _reload_from_disk(self) -> None:
        if not self.section:
            return
        self.documents = [load_document(image) for image in self.images]
        self.reviewed = load_reviewed(self.section)
        self.trajectory_reviews = load_trajectory_reviews(self.section)
        self.deferred_new_ids = load_deferred_new_ids(self.section)
        self.image_cache.clear()
        self._rebuild_queue(preferred_index=max(0, self.event_index))

    def _rebuild_queue(self, preferred_index: int = 0) -> None:
        if self._is_trajectory_mode():
            self._rebuild_trajectory_queue(preferred_index, activate=True)
            return
        self._rebuild_new_id_queue(preferred_index)

    def _rebuild_new_id_queue(self, preferred_index: int = 0) -> None:
        self.tracks = build_tracks(self.documents)
        self.events = review_events(
            self.documents,
            [image.name for image in self.images],
            self.reviewed,
            self.deferred_new_ids if self.outdoor_new_id_handoff else None,
        )
        values = [
            f"第 {event.first_frame_index + 1} 张 | 新 ID {event.track_id}"
            for event in self.events
        ]
        self.event_combo["values"] = values
        if self.events:
            self.event_index = min(max(0, preferred_index), len(self.events) - 1)
            self.event_var.set(values[self.event_index])
            self._activate_event()
        else:
            self.event_index = -1
            self.event_var.set("全部审核完成")
            self.candidates = []
            self.old_id_var.set("")
            self._fill_candidates()
            self._reset_motion()
            self._refresh()
        self.progress_var.set(
            f"待审核：{len(self.events)}    已确认保留：{len(self.reviewed)}"
        )

    def _trajectory_filtered_events(
        self,
        events: list[ReviewEvent],
    ) -> list[ReviewEvent]:
        selected_filter = self.trajectory_filter_var.get()
        if selected_filter == TRAJECTORY_FILTER_PENDING:
            return [
                event
                for event in events
                if event.signature not in self.trajectory_reviews
            ]
        if selected_filter == TRAJECTORY_FILTER_SUSPICIOUS:
            return [
                event
                for event in events
                if self.trajectory_metrics[event.track_id].suspicious
            ]
        if selected_filter == TRAJECTORY_FILTER_ISSUE:
            return [
                event
                for event in events
                if self.trajectory_reviews.get(event.signature) == "issue"
            ]
        return events

    def _trajectory_event_text(self, event: ReviewEvent) -> str:
        metrics = self.trajectory_metrics[event.track_id]
        status = self.trajectory_reviews.get(event.signature)
        status_text = {
            "passed": "✓通过",
            "issue": "⚠问题",
        }.get(status, "未检查")
        warning = f" | 疑似 {len(metrics.reasons)} 项" if metrics.suspicious else ""
        return (
            f"ID {event.track_id} | 第 {metrics.first_frame_index + 1}-"
            f"{metrics.last_frame_index + 1} 张 | {metrics.occurrence_count} 框"
            f" | {status_text}{warning}"
        )

    def _trajectory_review_id_text(self, event: ReviewEvent) -> str:
        status = self.trajectory_reviews.get(event.signature)
        status_text = {
            "passed": "通过",
            "issue": "有问题",
        }.get(status, "未检查")
        return f"ID {event.track_id}｜{status_text}"

    def _sync_motion_review_id_choices(self) -> None:
        if (
            self.motion_review_id_combo is None
            or not self.motion_review_id_combo.winfo_exists()
        ):
            return
        values = [
            self._trajectory_review_id_text(event)
            for event in self.trajectory_all_events
        ]
        self.motion_review_id_combo["values"] = values
        event = self.current_event()
        if event is None:
            self.motion_review_id_var.set("")
            return
        selected = next(
            (
                index
                for index, item in enumerate(self.trajectory_all_events)
                if item.track_id == event.track_id
            ),
            -1,
        )
        if selected >= 0:
            self.motion_review_id_combo.current(selected)

    def _on_motion_review_id_selected(self, _event=None) -> None:
        if (
            not self._is_trajectory_mode()
            or self.motion_review_id_combo is None
        ):
            return
        selected = self.motion_review_id_combo.current()
        if not 0 <= selected < len(self.trajectory_all_events):
            return
        target = self.trajectory_all_events[selected]
        self.trajectory_filter_var.set(
            TRAJECTORY_FILTER_PENDING
            if target.signature not in self.trajectory_reviews
            else TRAJECTORY_FILTER_ALL
        )
        self._rebuild_trajectory_queue(
            preferred_index=0,
            activate=True,
            active_track_id=target.track_id,
        )
        self.root.after_idle(self.release_motion_search_focus)

    def _update_trajectory_progress(self) -> None:
        total = len(self.trajectory_all_events)
        passed = sum(
            self.trajectory_reviews.get(event.signature) == "passed"
            for event in self.trajectory_all_events
        )
        issues = sum(
            self.trajectory_reviews.get(event.signature) == "issue"
            for event in self.trajectory_all_events
        )
        pending = total - passed - issues
        section_text = self.section.name if self.section else "-"
        text = (
            f"轨迹：{total}  未检查：{pending}  通过：{passed}  "
            f"问题：{issues}  |  {section_text}"
        )
        self.progress_var.set(text)
        self.trajectory_progress_var.set(text)

    def _rebuild_trajectory_queue(
        self,
        preferred_index: int = 0,
        activate: bool = True,
        active_track_id: int | None = None,
    ) -> None:
        self.tracks = build_tracks(self.documents)
        self.trajectory_metrics = analyze_tracks(self.tracks)
        self.trajectory_all_events = trajectory_review_events(
            self.documents,
            [image.name for image in self.images],
        )
        self.events = self._trajectory_filtered_events(
            self.trajectory_all_events
        )
        values = [self._trajectory_event_text(event) for event in self.events]
        self.event_combo["values"] = values
        if self.events:
            if active_track_id is None:
                self.event_index = min(
                    max(0, preferred_index), len(self.events) - 1
                )
            else:
                self.event_index = next(
                    (
                        index
                        for index, event in enumerate(self.events)
                        if event.track_id == active_track_id
                    ),
                    min(max(0, preferred_index), len(self.events) - 1),
                )
            self.event_var.set(values[self.event_index])
            if activate:
                self._activate_event()
            else:
                current = self.current_event()
                self.candidates = (
                    candidates_for_event(self.tracks, current)
                    if current is not None
                    else []
                )
                self._fill_candidates()
        else:
            self.event_index = -1
            self.event_var.set("当前筛选条件下没有轨迹")
            self.candidates = []
            self.old_id_var.set("")
            self._fill_candidates()
            if activate:
                self._reset_motion()
                self._refresh()
        self._update_trajectory_progress()
        self._sync_motion_review_id_choices()

    def _on_trajectory_filter_changed(self, _event=None) -> None:
        if self._is_trajectory_mode():
            self._rebuild_trajectory_queue(0, activate=True)

    def _refresh_queue_after_geometry_change(
        self,
        active_track_id: int | None,
        active_first_frame: int | None,
        selected_old_id: int | None,
        fallback_index: int,
    ) -> None:
        """框坐标变化不会改变 ID 关系，因此刷新数据但不重置当前视口。"""
        if self._is_trajectory_mode():
            self._rebuild_trajectory_queue(
                fallback_index,
                activate=False,
                active_track_id=active_track_id,
            )
            if selected_old_id is not None:
                self.old_id_var.set(str(selected_old_id))
            return
        self.tracks = build_tracks(self.documents)
        self.events = review_events(
            self.documents,
            [image.name for image in self.images],
            self.reviewed,
            self.deferred_new_ids if self.outdoor_new_id_handoff else None,
        )
        values = [
            f"第 {event.first_frame_index + 1} 张 | 新 ID {event.track_id}"
            for event in self.events
        ]
        self.event_combo["values"] = values
        if self.events:
            self.event_index = self._event_index_for_identity(
                self.events,
                active_track_id,
                active_first_frame,
                fallback_index,
            )
            self.event_var.set(values[self.event_index])
            current = self.current_event()
            self.candidates = candidates_for_event(self.tracks, current)
            if selected_old_id is not None:
                self.old_id_var.set(str(selected_old_id))
            elif self.candidates:
                self.old_id_var.set(str(self.candidates[0].track_id))
            else:
                self.old_id_var.set("")
            self._fill_candidates()
        else:
            self.event_index = -1
            self.event_var.set("全部审核完成")
            self.candidates = []
            self.old_id_var.set("")
            self._fill_candidates()
        self.progress_var.set(
            f"待审核：{len(self.events)}    已确认保留：{len(self.reviewed)}"
        )

    @staticmethod
    def _event_index_for_identity(
        events,
        track_id: int | None,
        first_frame_index: int | None,
        fallback_index: int,
    ) -> int:
        return next(
            (
                index
                for index, event in enumerate(events)
                if event.track_id == track_id
                and event.first_frame_index == first_frame_index
            ),
            min(max(0, fallback_index), len(events) - 1),
        )

    def current_event(self) -> ReviewEvent | None:
        if 0 <= self.event_index < len(self.events):
            return self.events[self.event_index]
        return None

    def _activate_event(self) -> None:
        event = self.current_event()
        if not event:
            return
        self._stop_global_video()
        self.motion_pan_offset = [0.0, 0.0]
        self.motion_zoom = 1.0
        self.motion_view_initialized = False
        self.motion_focus_frame_index = None
        self.motion_search_track_id = None
        self.motion_search_id_var.set("")
        self.motion_edit_occurrence = None
        self.motion_edit_mode = None
        self.motion_edit_preview_rect = None
        self.motion_zoom_var.set("缩放 1.00×")
        self.display_frame_index = event.first_frame_index
        self._sync_global_raw_to_display()
        if self._is_trajectory_mode():
            self.candidates = []
            self.old_id_var.set("")
            default_reference = event.first_frame_index
        else:
            self.candidates = candidates_for_event(self.tracks, event)
            self.old_id_var.set(
                str(self.candidates[0].track_id) if self.candidates else ""
            )
            default_reference = (
                self.candidates[0].last_frame_index
                if self.candidates
                else max(0, event.first_frame_index - 1)
            )
        self._set_reference_frame(default_reference, refresh=False)
        self._fill_candidates()
        self._reset_motion()
        self._refresh()
        self._sync_motion_review_id_choices()

    def _fill_candidates(self) -> None:
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        for candidate in self.candidates[:30]:
            item = self.candidate_tree.insert(
                "",
                tk.END,
                values=(
                    candidate.track_id,
                    f"{candidate.distance:.1f} px",
                    f"第 {candidate.last_frame_index + 1} 张",
                ),
            )
            if str(candidate.track_id) == self.old_id_var.get():
                self.candidate_tree.selection_set(item)

    def _on_event_selected(self, _event=None) -> None:
        selected = self.event_combo.current()
        if selected >= 0:
            self.event_index = selected
            self._activate_event()

    def previous_event(self) -> None:
        if self._is_trajectory_mode():
            self.previous_trajectory()
            return
        if self.events:
            self.event_index = (self.event_index - 1) % len(self.events)
            self.event_combo.current(self.event_index)
            self._activate_event()

    def next_event(self) -> None:
        if self._is_trajectory_mode():
            self.next_trajectory()
            return
        if self.events:
            self.event_index = (self.event_index + 1) % len(self.events)
            self.event_combo.current(self.event_index)
            self._activate_event()

    def _advance_trajectory_section(self, direction: int) -> bool:
        if not self.section_dirs or self.section not in self.section_dirs:
            return False
        current_index = self.section_dirs.index(self.section)
        target_index = current_index + direction
        while 0 <= target_index < len(self.section_dirs):
            target = self.section_dirs[target_index]
            self.section_var.set(target.name)
            self._load_section(target, clear_undo=True)
            if self.events:
                self.event_index = 0 if direction > 0 else len(self.events) - 1
                self.event_combo.current(self.event_index)
                self._activate_event()
                return True
            target_index += direction
        if direction > 0 and self._all_outdoor_trajectories_reviewed():
            return self._begin_outdoor_new_id_handoff()
        self.status_var.set(
            "已经是最后一个室外区段"
            if direction > 0
            else "已经是第一个室外区段"
        )
        return False

    def _all_outdoor_trajectories_reviewed(self) -> bool:
        sections = self._outdoor_sections()
        if not sections:
            return False
        try:
            for section in sections:
                if section == self.section and self._is_trajectory_mode():
                    events = self.trajectory_all_events
                    statuses = self.trajectory_reviews
                else:
                    images = image_files(section)
                    documents = [load_document(image) for image in images]
                    events = trajectory_review_events(
                        documents,
                        [image.name for image in images],
                    )
                    statuses = load_trajectory_reviews(section)
                if any(event.signature not in statuses for event in events):
                    return False
        except (OSError, json.JSONDecodeError, FileNotFoundError):
            return False
        return True

    def _begin_outdoor_new_id_handoff(self) -> bool:
        sections = self._outdoor_sections()
        if not sections:
            return False
        self._stop_motion()
        self.work_mode_var.set(MODE_NEW_ID)
        self.outdoor_new_id_handoff = True
        self.section_dirs = sections
        self.section_combo["values"] = [path.name for path in sections]
        self._update_mode_ui()
        for section in sections:
            self.section_var.set(section.name)
            self._load_section(section, clear_undo=True)
            if self.events:
                self.status_var.set(
                    "室外轨迹已全部复查完成；"
                    f"开始审核 {section.name} 的待处理新 ID"
                )
                return True
        self.outdoor_new_id_handoff = False
        self.status_var.set("室外轨迹已全部复查完成，没有待审核的新 ID")
        return True

    def _advance_outdoor_new_id_section(self) -> bool:
        if (
            not self.outdoor_new_id_handoff
            or self.section not in self.section_dirs
        ):
            return False
        start = self.section_dirs.index(self.section) + 1
        for section in self.section_dirs[start:]:
            self.section_var.set(section.name)
            self._load_section(section, clear_undo=True)
            if self.events:
                self.status_var.set(
                    f"继续审核 {section.name} 的待处理新 ID"
                )
                return True
        self.outdoor_new_id_handoff = False
        self.status_var.set("所有室外新增 ID 已审核完成")
        return True

    def previous_trajectory(self) -> None:
        if not self._is_trajectory_mode():
            return
        if self.events and self.event_index > 0:
            self.event_index -= 1
            self.event_combo.current(self.event_index)
            self._activate_event()
            return
        self._advance_trajectory_section(-1)

    def next_trajectory(self) -> None:
        if not self._is_trajectory_mode():
            return
        if self.events and self.event_index < len(self.events) - 1:
            self.event_index += 1
            self.event_combo.current(self.event_index)
            self._activate_event()
            return
        self._advance_trajectory_section(1)

    @staticmethod
    def _shortcut_is_typing(event) -> bool:
        widget = getattr(event, "widget", None)
        return widget is not None and widget.winfo_class() in {
            "TEntry",
            "Entry",
            "TCombobox",
            "TSpinbox",
            "Spinbox",
            "Text",
        }

    def _on_trajectory_pass_shortcut(self, event):
        if not self._is_trajectory_mode() or self._shortcut_is_typing(event):
            return None
        self.mark_trajectory_review("passed")
        return "break"

    def _on_trajectory_issue_shortcut(self, event):
        if not self._is_trajectory_mode() or self._shortcut_is_typing(event):
            return None
        self.mark_trajectory_review("issue")
        return "break"

    def mark_trajectory_review(self, status: str) -> None:
        event = self.current_event()
        if (
            not self._is_trajectory_mode()
            or event is None
            or self.section is None
        ):
            return
        old_index = self.event_index
        self.trajectory_reviews[event.signature] = status
        save_trajectory_reviews(self.section, self.trajectory_reviews)
        self._rebuild_trajectory_queue(
            old_index,
            activate=False,
            active_track_id=event.track_id,
        )
        current_index = next(
            (
                index
                for index, item in enumerate(self.events)
                if item.track_id == event.track_id
            ),
            None,
        )
        self.status_var.set(
            f"轨迹 ID {event.track_id} 已标记为"
            f"{'通过' if status == 'passed' else '有问题'}"
        )
        if current_index is not None:
            self.event_index = current_index
            self.event_combo.current(current_index)
            self.next_trajectory()
        elif self.events:
            self.event_index = min(old_index, len(self.events) - 1)
            self.event_combo.current(self.event_index)
            self._activate_event()
        else:
            self._advance_trajectory_section(1)

    def previous_frame(self) -> None:
        if self.images:
            self._stop_global_video()
            self.display_frame_index = max(0, self.display_frame_index - 1)
            self._sync_global_raw_to_display()
            self._refresh()

    def next_frame(self) -> None:
        if self.images:
            self._stop_global_video()
            self.display_frame_index = min(
                len(self.images) - 1, self.display_frame_index + 1
            )
            self._sync_global_raw_to_display()
            self._refresh()

    def go_event_frame(self) -> None:
        event = self.current_event()
        if event:
            self._stop_global_video()
            self.display_frame_index = event.first_frame_index
            self._sync_global_raw_to_display()
            self._refresh()

    def _on_candidate_selected(self, _event=None) -> None:
        selection = self.candidate_tree.selection()
        if not selection:
            return
        values = self.candidate_tree.item(selection[0], "values")
        if values:
            selected_id = str(values[0])
            # 保存框后会重建候选列表并恢复原选中项。该程序化选择也会
            # 触发 TreeviewSelect；候选 ID 未变化时不能重置局部运动帧。
            if selected_id == self.old_id_var.get().strip():
                return
            self.old_id_var.set(selected_id)
            self.go_candidate_reference_frame(refresh=False)
            self._reset_motion()
            self._refresh()

    def _on_old_id_changed(self, _event=None) -> None:
        self._reset_motion()
        self._refresh()

    def _reference_choice_text(self, index: int) -> str:
        if not self.images:
            return ""
        index = min(max(0, index), len(self.images) - 1)
        return (
            f"第 {index + 1:02d}/{len(self.images):02d} 张"
            f"  |  {self.images[index].stem}"
        )

    def _update_reference_choices(self) -> None:
        values = [
            self._reference_choice_text(index)
            for index in range(len(self.images))
        ]
        self.reference_frame_combo["values"] = values
        if values:
            self._set_reference_frame(0, refresh=False)
        else:
            self.reference_frame_var.set("")

    def _set_reference_frame(self, index: int, refresh: bool = True) -> None:
        if not self.images:
            self.reference_frame_index = 0
            self.reference_frame_var.set("")
            return
        self.reference_frame_index = min(max(0, index), len(self.images) - 1)
        self.reference_frame_combo.current(self.reference_frame_index)
        self.reference_frame_var.set(
            self._reference_choice_text(self.reference_frame_index)
        )
        if refresh:
            self._render_reference()

    def _on_reference_frame_selected(self, _event=None) -> None:
        selected = self.reference_frame_combo.current()
        if selected >= 0:
            self._set_reference_frame(selected)

    def previous_reference_frame(self) -> None:
        self._set_reference_frame(self.reference_frame_index - 1)

    def next_reference_frame(self) -> None:
        self._set_reference_frame(self.reference_frame_index + 1)

    def go_candidate_reference_frame(self, refresh: bool = True) -> None:
        event = self.current_event()
        old_id = self._selected_old_id()
        occurrence = (
            last_occurrence_before(self.tracks, old_id, event.first_frame_index)
            if event is not None and old_id is not None
            else None
        )
        if occurrence is not None:
            self._set_reference_frame(occurrence.frame_index, refresh=refresh)
        elif event is not None:
            self._set_reference_frame(
                max(0, event.first_frame_index - 1), refresh=refresh
            )

    def _on_show_class_names_changed(self) -> None:
        self._save_settings()
        self._refresh()

    def keep_current(self) -> None:
        if self._is_trajectory_mode():
            self.status_var.set("轨迹复查模式请用 Space 标记通过，M 标记问题")
            return
        event = self.current_event()
        if not event or not self.section:
            return
        snapshot = snapshot_files(self.section, self.images)
        self.reviewed.add(event.signature)
        if self.outdoor_new_id_handoff:
            self.deferred_new_ids.discard(event.track_id)
            save_deferred_new_ids(self.section, self.deferred_new_ids)
        save_reviewed(self.section, self.reviewed)
        self.undo_stack.append((f"保留新 ID {event.track_id}", snapshot))
        old_index = self.event_index
        self.status_var.set(f"已确认 ID {event.track_id} 是新目标")
        self._rebuild_queue(preferred_index=old_index)
        if not self.events:
            self._advance_outdoor_new_id_section()

    def merge_current(self) -> None:
        if self._is_trajectory_mode():
            self.status_var.set("轨迹复查模式不会执行新 ID 合并")
            return
        event = self.current_event()
        if not event or not self.section:
            return
        try:
            old_id = int(self.old_id_var.get().strip())
        except ValueError:
            messagebox.showwarning("ID 无效", "请输入整数形式的旧 ID。")
            return
        new_id = event.track_id
        if old_id <= 0 or old_id == new_id:
            messagebox.showwarning(
                "ID 无效",
                f"旧 ID 必须大于 0，且不能与新 ID {new_id} 相同。",
            )
            return
        if old_id not in self.tracks:
            messagebox.showwarning("ID 不存在", f"当前区段中没有 ID {old_id}。")
            return
        collisions = collision_frames(self.documents, old_id, new_id)
        final_displaced_id = None
        if collisions:
            shown = "、".join(str(index + 1) for index in collisions[:10])
            if len(collisions) > 10:
                shown += f" 等 {len(collisions)} 张"
            current_max = max(
                (
                    group_id
                    for document in self.documents
                    for shape in document.get("shapes", [])
                    if isinstance((group_id := shape.get("group_id")), int)
                    and group_id > 0
                ),
                default=0,
            )
            final_displaced_id = current_max
            merged_id = old_id - 1 if old_id > new_id else old_id
            confirmed = messagebox.askyesno(
                "确认冲突合并",
                f"检测到旧 ID {old_id} 与当前 ID {new_id} 在第 "
                f"{shown} 同时出现。\n\n"
                "确认后将执行：\n"
                f"1. 仅把这些冲突帧中的原 ID {old_id} 框及其 "
                f"head/tail 改为末尾 ID {final_displaced_id}；\n"
                f"2. 末尾 ID {final_displaced_id} 会加入待修正队列；\n"
                f"3. 将当前 ID {new_id} 合并为旧 ID {old_id}"
                f"（连续重编号后显示为 {merged_id}）；\n"
                f"4. 其他大于 {new_id} 的 ID 按现有规则减 1。\n\n"
                "是否确认执行？",
                icon="warning",
                parent=self.motion_window or self.root,
            )
            if not confirmed:
                self.status_var.set("已取消冲突合并")
                return

        if self.backup_path is None:
            try:
                self.backup_path = create_section_backup(self.section)
                self.backup_var.set(f"备份：{self.backup_path.name}")
            except OSError as error:
                messagebox.showerror("备份失败", str(error))
                return

        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_deferred_new_ids = set(self.deferred_new_ids)
        collision_occurrences = [
            occurrence
            for occurrence in self.tracks.get(old_id, [])
            if occurrence.frame_index in set(collisions)
        ]
        try:
            displaced_changes = 0
            if collisions:
                _temporary_id, displaced_changes = displace_id_on_frames(
                    self.documents,
                    old_id,
                    collisions,
                )
                for occurrence in collision_occurrences:
                    self.reviewed.discard(
                        event_signature(
                            self.images[occurrence.frame_index].name,
                            occurrence.rect,
                        )
                    )
            changed = sum(
                remap_document(document, old_id, new_id)
                for document in self.documents
            )
            if self.outdoor_new_id_handoff:
                self.deferred_new_ids = {
                    track_id - 1 if track_id > new_id else track_id
                    for track_id in self.deferred_new_ids
                    if track_id != new_id
                }
                if final_displaced_id is not None:
                    self.deferred_new_ids.add(final_displaced_id)
                save_deferred_new_ids(
                    self.section,
                    self.deferred_new_ids,
                )
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.deferred_new_ids = original_deferred_new_ids
            restore_snapshot(snapshot)
            messagebox.showerror("修改失败", str(error))
            return

        merged_id = old_id - 1 if old_id > new_id else old_id
        undo_label = f"合并 {new_id} → {merged_id}"
        if final_displaced_id is not None:
            undo_label += f"，冲突旧框 → {final_displaced_id}"
        self.undo_stack.append((undo_label, snapshot))
        old_index = self.event_index
        if final_displaced_id is None:
            self.status_var.set(
                f"已合并 {new_id} → {merged_id}；修改 {changed} 个 group_id；"
                f"原大于 {new_id} 的 ID 已减 1"
            )
        else:
            self.status_var.set(
                f"已将 {len(collisions)} 张冲突帧中的旧 ID {old_id} "
                f"拆为末尾 ID {final_displaced_id}"
                f"（修改 {displaced_changes} 个关联标注），并合并 "
                f"{new_id} → {merged_id}"
            )
        self._rebuild_queue(preferred_index=old_index)
        if not self.events:
            self._advance_outdoor_new_id_section()

    def undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("没有可撤销的操作")
            return
        label, snapshot = self.undo_stack.pop()
        try:
            restore_snapshot(snapshot)
            self._reload_from_disk()
            self.status_var.set(f"已撤销：{label}")
        except OSError as error:
            messagebox.showerror("撤销失败", str(error))

    def _get_image(self, index: int) -> Image.Image:
        path = self.images[index]
        image = self.image_cache.get(path)
        if image is not None:
            self.image_cache.move_to_end(path)
            return image.copy()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        self.image_cache[path] = image
        while len(self.image_cache) > 6:
            self.image_cache.popitem(last=False)
        return image.copy()

    @staticmethod
    @lru_cache(maxsize=12)
    def _font(size: int) -> ImageFont.ImageFont:
        for path in (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _draw_boxes(
        self,
        image: Image.Image,
        frame_index: int,
        selected_id: int | None,
        offset: tuple[float, float] = (0.0, 0.0),
        crop_rect=None,
        scale_hint: float = 1.0,
    ) -> None:
        draw = ImageDraw.Draw(image)
        records = bee_occurrences(self.documents[frame_index], frame_index)
        font = self._font(max(10, int(14 * scale_hint)))
        for record in records:
            if crop_rect and not intersects(record.rect, crop_rect):
                continue
            x1 = record.rect[0] - offset[0]
            y1 = record.rect[1] - offset[1]
            x2 = record.rect[2] - offset[0]
            y2 = record.rect[3] - offset[1]
            color = (255, 45, 45) if record.track_id == selected_id else track_color(record.track_id)
            width = 4 if record.track_id == selected_id else 2
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            show_id = self.show_all_ids.get() or record.track_id == selected_id
            if show_id or self.show_class_names.get():
                parts = []
                if self.show_class_names.get():
                    parts.append("bee")
                if show_id:
                    parts.append(str(record.track_id))
                text = " ".join(parts)
                text_box = draw.textbbox((x1, y1), text, font=font, stroke_width=2)
                draw.rectangle(text_box, fill=(0, 0, 0))
                draw.text(
                    (x1, y1),
                    text,
                    fill=color,
                    font=font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

    def _fit_to_canvas(self, canvas: tk.Canvas, image: Image.Image):
        width = max(2, canvas.winfo_width())
        height = max(2, canvas.winfo_height())
        scale = min(width / image.width, height / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        x = (width - size[0]) / 2
        y = (height - size[1]) / 2
        return resized, scale, x, y

    @staticmethod
    def _motion_fit_geometry(canvas_size, image_size, zoom, pan=(0.0, 0.0)):
        canvas_width, canvas_height = canvas_size
        image_width, image_height = image_size
        fit_scale = min(
            canvas_width / image_width,
            canvas_height / image_height,
        )
        scale = fit_scale * max(1.0, zoom)
        size = (
            max(1, int(round(image_width * scale))),
            max(1, int(round(image_height * scale))),
        )
        max_pan_x = max(0.0, (size[0] - canvas_width) / (2.0 * scale))
        max_pan_y = max(0.0, (size[1] - canvas_height) / (2.0 * scale))
        pan_x = min(max(float(pan[0]), -max_pan_x), max_pan_x)
        pan_y = min(max(float(pan[1]), -max_pan_y), max_pan_y)
        x = (canvas_width - size[0]) / 2 - pan_x * scale
        y = (canvas_height - size[1]) / 2 - pan_y * scale
        return size, scale, x, y, (pan_x, pan_y)

    @staticmethod
    def _motion_initial_view(canvas_size, image_size, focus_crop):
        canvas_width, canvas_height = canvas_size
        image_width, image_height = image_size
        focus_width = max(1.0, focus_crop[2] - focus_crop[0])
        focus_height = max(1.0, focus_crop[3] - focus_crop[1])
        full_scale = min(
            canvas_width / image_width,
            canvas_height / image_height,
        )
        focus_scale = min(
            canvas_width / focus_width,
            canvas_height / focus_height,
        ) * 0.92
        zoom = min(6.0, max(1.0, focus_scale / full_scale))
        pan = (
            (focus_crop[0] + focus_crop[2]) / 2 - image_width / 2,
            (focus_crop[1] + focus_crop[3]) / 2 - image_height / 2,
        )
        return zoom, pan

    def _fit_motion_to_canvas(
        self,
        canvas: tk.Canvas,
        image: Image.Image,
        resample=Image.Resampling.LANCZOS,
    ):
        width = max(2, canvas.winfo_width())
        height = max(2, canvas.winfo_height())
        size, scale, x, y, pan = self._motion_fit_geometry(
            (width, height),
            image.size,
            self.motion_zoom,
            self.motion_pan_offset,
        )
        self.motion_pan_offset = [pan[0], pan[1]]
        resized = image.resize(size, resample)
        return resized, scale, x, y

    def _crop_around(self, image_size: tuple[int, int], rect) -> tuple[int, int, int, int]:
        width, height = image_size
        cx, cy = rect_center(rect)
        box_width = max(1.0, rect[2] - rect[0])
        box_height = max(1.0, rect[3] - rect[1])
        crop_width = min(width, max(280.0, box_width * 7.0))
        crop_height = min(height, max(220.0, box_height * 6.0))
        x1 = min(max(0.0, cx - crop_width / 2), width - crop_width)
        y1 = min(max(0.0, cy - crop_height / 2), height - crop_height)
        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x1 + crop_width)),
            int(round(y1 + crop_height)),
        )

    def open_global_window(self) -> None:
        if (
            self.global_window is not None
            and self.global_window.winfo_exists()
        ):
            self.global_window.deiconify()
            self.global_window.lift()
            self.global_window.focus_force()
            self._render_global()
            return
        window = tk.Toplevel(self.root)
        window.title("全局鸟瞰图")
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        width = min(1560, max(1080, screen_width - 100))
        height = min(880, max(620, screen_height - 120))
        window.geometry(f"{width}x{height}")
        window.minsize(980, 560)
        if self.icon is not None:
            try:
                window.iconphoto(True, self.icon)
            except tk.TclError:
                pass
        controls = ttk.Frame(window, padding=(7, 5))
        controls.pack(fill=tk.X)
        self.global_play_button = ttk.Button(
            controls, text="▶ 播放", command=self.toggle_global_video
        )
        self.global_play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="Q 上一抽帧", command=self.previous_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="回到新 ID 首次出现", command=self.go_event_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="下一抽帧 E", command=self.next_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="◀ 原帧", command=self.previous_global_raw_frame
        ).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(
            controls, text="原帧 ▶", command=self.next_global_raw_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(controls, text="速度：").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(
            controls,
            textvariable=self.global_speed,
            values=["5 FPS", "10 FPS", "15 FPS", "20 FPS"],
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls, text="循环", variable=self.global_loop
        ).pack(side=tk.LEFT, padx=7)
        ttk.Label(controls, textvariable=self.global_status_var).pack(
            side=tk.RIGHT, padx=5
        )

        views = ttk.Frame(window)
        views.pack(fill=tk.BOTH, expand=True, padx=7, pady=(0, 7))
        views.rowconfigure(0, weight=1)
        views.columnconfigure(0, weight=1, uniform="global")
        views.columnconfigure(1, weight=1, uniform="global")
        sampled_frame = ttk.LabelFrame(
            views, text="带检测框和 Track ID 的抽帧（点击框选择旧 ID）"
        )
        sampled_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 3))
        raw_frame = ttk.LabelFrame(views, text="原视频全局画面")
        raw_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(3, 0))
        self.global_canvas = tk.Canvas(
            sampled_frame,
            background="#14171b",
            highlightthickness=0,
            cursor="hand2",
        )
        self.global_canvas.pack(fill=tk.BOTH, expand=True)
        self.global_canvas.bind("<Configure>", self._schedule_redraw)
        self.global_canvas.bind("<Button-1>", self._on_global_click)
        self.global_raw_canvas = tk.Canvas(
            raw_frame,
            background="#14171b",
            highlightthickness=0,
        )
        self.global_raw_canvas.pack(fill=tk.BOTH, expand=True)
        self.global_raw_canvas.bind("<Configure>", self._schedule_redraw)
        window.bind("<q>", lambda _event: self.previous_frame())
        window.bind("<e>", lambda _event: self.next_frame())
        window.protocol("WM_DELETE_WINDOW", self._close_global_window)
        self.global_window = window
        self._sync_global_raw_to_display()
        self._render_global(force_left=True)

    def _close_global_window(self) -> None:
        self._stop_global_video()
        if self.global_window is not None and self.global_window.winfo_exists():
            self.global_window.destroy()
        self.global_window = None
        self.global_canvas = None
        self.global_raw_canvas = None
        self.global_photo = None
        self.global_raw_photo = None
        self.global_play_button = None
        self.global_transform = None
        self.global_left_render_key = None
        self._release_global_raw_video()

    def _global_source_frames(self) -> list[int]:
        return [
            self._source_frame_number(index) for index in range(len(self.images))
        ]

    def _global_raw_bounds(self) -> tuple[int, int]:
        sources = self._global_source_frames()
        if not sources:
            return 0, 0
        gaps = [
            right - left
            for left, right in zip(sources, sources[1:])
            if right > left
        ]
        gap = max(1, sorted(gaps)[len(gaps) // 2]) if gaps else 5
        return sources[0], sources[-1] + gap - 1

    def _sync_global_raw_to_display(self) -> None:
        if self.images:
            self.global_raw_frame_index = self._source_frame_number(
                self.display_frame_index
            )

    def _sync_display_from_global_raw(self) -> None:
        sources = self._global_source_frames()
        if not sources:
            return
        self.display_frame_index = max(
            0,
            min(
                len(sources) - 1,
                bisect_right(sources, self.global_raw_frame_index) - 1,
            ),
        )

    def _release_global_raw_video(self) -> None:
        if self.global_raw_capture is not None:
            self.global_raw_capture.release()
        self.global_raw_capture = None
        self.global_raw_video_path = None
        self.global_raw_capture_next_frame = None
        self.global_raw_image_cache.clear()

    def _ensure_global_raw_video(self) -> bool:
        wanted = self._resolve_raw_video_path()
        if wanted is None:
            return False
        if (
            self.global_raw_capture is not None
            and self.global_raw_video_path == wanted
            and self.global_raw_capture.isOpened()
        ):
            return True
        self._release_global_raw_video()
        capture = cv2.VideoCapture(str(wanted))
        if not capture.isOpened():
            capture.release()
            return False
        self.global_raw_capture = capture
        self.global_raw_video_path = wanted
        return True

    def _get_global_raw_image(self, frame_index: int) -> Image.Image | None:
        cached = self.global_raw_image_cache.get(frame_index)
        if cached is not None:
            self.global_raw_image_cache.move_to_end(frame_index)
            return cached.copy()
        if not self._ensure_global_raw_video():
            return None
        if self.global_raw_capture_next_frame != frame_index:
            self.global_raw_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.global_raw_capture.read()
        if not ok:
            return None
        self.global_raw_capture_next_frame = frame_index + 1
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.global_raw_image_cache[frame_index] = image
        while len(self.global_raw_image_cache) > 12:
            self.global_raw_image_cache.popitem(last=False)
        return image.copy()

    def _stop_global_video(self) -> None:
        if self.global_job is not None:
            self.root.after_cancel(self.global_job)
            self.global_job = None
        self.global_playing = False
        if (
            self.global_play_button is not None
            and self.global_play_button.winfo_exists()
        ):
            self.global_play_button.configure(text="▶ 播放")

    def toggle_global_video(self) -> None:
        if not self.images:
            return
        if self.global_playing:
            self._stop_global_video()
            return
        start, end = self._global_raw_bounds()
        if self.global_raw_frame_index >= end:
            self.global_raw_frame_index = start
            self._sync_display_from_global_raw()
        self.global_playing = True
        if self.global_play_button is not None:
            self.global_play_button.configure(text="⏸ 暂停")
        self._schedule_global_tick()

    def _global_delay_ms(self) -> int:
        try:
            fps = float(self.global_speed.get().split()[0])
        except (ValueError, IndexError):
            fps = 10.0
        return max(35, int(round(1000 / max(1.0, fps))))

    def _schedule_global_tick(self) -> None:
        if self.global_playing:
            self.global_job = self.root.after(
                self._global_delay_ms(), self._global_tick
            )

    def _global_tick(self) -> None:
        self.global_job = None
        if not self.global_playing or not self.images:
            return
        start, end = self._global_raw_bounds()
        if self.global_raw_frame_index >= end:
            if self.global_loop.get():
                self.global_raw_frame_index = start
            else:
                self._stop_global_video()
                return
        else:
            self.global_raw_frame_index += 1
        self._sync_display_from_global_raw()
        self._render_global()
        self._schedule_global_tick()

    def previous_global_raw_frame(self) -> None:
        self._stop_global_video()
        start, _end = self._global_raw_bounds()
        self.global_raw_frame_index = max(
            start, self.global_raw_frame_index - 1
        )
        self._sync_display_from_global_raw()
        self._render_global()

    def next_global_raw_frame(self) -> None:
        self._stop_global_video()
        _start, end = self._global_raw_bounds()
        self.global_raw_frame_index = min(
            end, self.global_raw_frame_index + 1
        )
        self._sync_display_from_global_raw()
        self._render_global()

    def _render_global(self, force_left: bool = False) -> None:
        canvas = self.global_canvas
        raw_canvas = self.global_raw_canvas
        if (
            canvas is None
            or not canvas.winfo_exists()
            or raw_canvas is None
            or not raw_canvas.winfo_exists()
        ):
            return
        event = self.current_event()
        if not self.images:
            canvas.delete("all")
            raw_canvas.delete("all")
            canvas.create_text(
                canvas.winfo_width() / 2,
                canvas.winfo_height() / 2,
                text="没有待审核的新 ID",
                fill="white",
                font=("Microsoft YaHei UI", 18),
            )
            return
        selected_id = event.track_id if event is not None else None
        left_key = (
            self.display_frame_index,
            canvas.winfo_width(),
            canvas.winfo_height(),
            selected_id,
            self.show_all_ids.get(),
            self.show_class_names.get(),
        )
        if force_left or left_key != self.global_left_render_key:
            canvas.delete("all")
            image = self._get_image(self.display_frame_index)
            self._draw_boxes(image, self.display_frame_index, selected_id)
            resized, scale, x, y = self._fit_to_canvas(canvas, image)
            self.global_photo = ImageTk.PhotoImage(resized)
            canvas.create_image(x, y, image=self.global_photo, anchor=tk.NW)
            self.global_transform = (scale, x, y)
            self.global_left_render_key = left_key

        raw_canvas.delete("all")
        raw_image = self._get_global_raw_image(self.global_raw_frame_index)
        if raw_image is None:
            raw_canvas.create_text(
                max(1, raw_canvas.winfo_width()) / 2,
                max(1, raw_canvas.winfo_height()) / 2,
                text="无法读取当前区段对应的原视频",
                fill="white",
                font=("Microsoft YaHei UI", 12),
            )
        else:
            draw = ImageDraw.Draw(raw_image)
            font = self._font(16)
            source_frame = self._source_frame_number(self.display_frame_index)
            title = (
                f"原视频帧 {self.global_raw_frame_index}  "
                f"对应抽帧 {source_frame}"
            )
            title_box = draw.textbbox((8, 8), title, font=font, stroke_width=2)
            draw.rectangle(title_box, fill=(0, 0, 0))
            draw.text(
                (8, 8),
                title,
                fill=(255, 255, 255),
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            resized, _scale, x, y = self._fit_to_canvas(
                raw_canvas, raw_image
            )
            self.global_raw_photo = ImageTk.PhotoImage(resized)
            raw_canvas.create_image(
                x, y, image=self.global_raw_photo, anchor=tk.NW
            )
        self.global_status_var.set(
            f"抽帧 {self.display_frame_index + 1}/{len(self.images)}"
            f"  原帧 {self.global_raw_frame_index}"
            f"  审核新 ID：{selected_id if selected_id is not None else '-'}"
            f"  已选旧 ID：{self._selected_old_id() if self._selected_old_id() is not None else '-'}"
        )

    def _render_current_local(self) -> None:
        canvas = self.current_canvas
        canvas.delete("all")
        event = self.current_event()
        if not self.images or not event:
            return
        occurrence = find_occurrence(
            self.tracks, event.track_id, self.display_frame_index
        )
        frame_index = self.display_frame_index
        focus_rect = occurrence.rect if occurrence else event.rect
        image = self._get_image(frame_index)
        crop = self._crop_around(image.size, focus_rect)
        cropped = image.crop(crop)
        self._draw_boxes(
            cropped,
            frame_index,
            event.track_id,
            offset=(crop[0], crop[1]),
            crop_rect=crop,
            scale_hint=1.4,
        )
        resized, _scale, x, y = self._fit_to_canvas(canvas, cropped)
        self.current_photo = ImageTk.PhotoImage(resized)
        canvas.create_image(x, y, image=self.current_photo, anchor=tk.NW)
        if occurrence is None:
            canvas.create_text(
                12,
                12,
                text="该 ID 在当前帧不存在",
                fill="#ff7777",
                anchor=tk.NW,
                font=("Microsoft YaHei UI", 11, "bold"),
            )

    def _selected_old_id(self) -> int | None:
        try:
            return int(self.old_id_var.get().strip())
        except ValueError:
            return None

    def _reset_motion(self) -> None:
        self._stop_motion()
        self.motion_view_initialized = False
        self.motion_focus_frame_index = None
        self.motion_search_track_id = None
        self.motion_search_id_var.set("")
        event = self.current_event()
        if not event or not self.images:
            self.motion_sequence = []
            self.motion_source_frames = []
            self.motion_index = 0
            self.motion_event_start_index = 0
            self.motion_status_var.set(
                "没有可复查的轨迹"
                if self._is_trajectory_mode()
                else "没有待审核的新 ID"
            )
            self.motion_frame_var.set("")
            self._render_trajectory_timeline()
            return
        old_id = self._selected_old_id()
        self._set_motion_sequence_around(event.first_frame_index)
        self.motion_event_start_index = self.motion_index
        self.raw_play_start_frame = self.motion_source_frames[self.motion_index]
        self.raw_frame_index = self.raw_play_start_frame
        self.motion_left_render_key = None
        if self._is_trajectory_mode():
            metrics = self.trajectory_metrics.get(event.track_id)
            reasons = (
                "；".join(metrics.reasons)
                if metrics and metrics.reasons
                else "未发现规则异常"
            )
            self.motion_status_var.set(
                f"复查轨迹 ID {event.track_id}：{reasons}"
            )
        else:
            self.motion_status_var.set(
                f"旧 {old_id if old_id is not None else '-'} → 新 {event.track_id}"
            )

    def _source_frame_number(self, image_index: int) -> int:
        match = re.search(r"_frame_(\d+)", self.images[image_index].stem)
        return int(match.group(1)) if match else image_index * self.sample_gap

    def _resolve_raw_video_path(self) -> Path | None:
        if not self.section or not self.raw_video_root.exists():
            return None
        sequence = self.section.name.split("_区段_", 1)[0]
        matches = sorted(self.raw_video_root.rglob(f"{sequence}.mp4"))
        return matches[0] if matches else None

    def _release_raw_video(self) -> None:
        if self.raw_capture is not None:
            self.raw_capture.release()
        self.raw_capture = None
        self.raw_capture_next_frame = None
        self.raw_video_path = None
        self.raw_video_frame_count = 0
        self.raw_image_cache.clear()

    def choose_raw_video_root(self) -> None:
        selected = filedialog.askdirectory(
            title="选择包含 A/B 原视频的目录",
            initialdir=str(
                self.raw_video_root
                if self.raw_video_root.exists()
                else Path.home()
            ),
            parent=self.motion_window or self.root,
        )
        if not selected:
            return
        self.raw_video_root = Path(selected)
        self._release_raw_video()
        self._save_settings()
        self._render_motion(force_left=True)

    def _ensure_raw_video(self) -> bool:
        wanted = self._resolve_raw_video_path()
        if wanted is None:
            return False
        if (
            self.raw_capture is not None
            and self.raw_video_path == wanted
            and self.raw_capture.isOpened()
        ):
            return True
        self._release_raw_video()
        capture = cv2.VideoCapture(str(wanted))
        if not capture.isOpened():
            capture.release()
            return False
        self.raw_capture = capture
        self.raw_video_path = wanted
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.raw_video_frame_count = max(0, frame_count)
        if frame_count > 0:
            self.raw_end_frame = min(self.raw_end_frame, frame_count - 1)
        return True

    def _get_raw_image(self, frame_index: int) -> Image.Image | None:
        cached = self.raw_image_cache.get(frame_index)
        if cached is not None:
            self.raw_image_cache.move_to_end(frame_index)
            return cached.copy()
        if not self._ensure_raw_video():
            return None
        if self.raw_capture_next_frame != frame_index:
            self.raw_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.raw_capture.read()
        if not ok:
            return None
        self.raw_capture_next_frame = frame_index + 1
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.raw_image_cache[frame_index] = image
        while len(self.raw_image_cache) > 64:
            self.raw_image_cache.popitem(last=False)
        return image.copy()

    def _sync_motion_index_from_raw(self) -> None:
        if not self.motion_source_frames:
            self.motion_index = 0
            return
        self.motion_index = max(
            0,
            min(
                len(self.motion_source_frames) - 1,
                bisect_right(self.motion_source_frames, self.raw_frame_index) - 1,
            ),
        )

    def _stop_motion(self) -> None:
        if self.motion_job is not None:
            self.root.after_cancel(self.motion_job)
            self.motion_job = None
        self.motion_playing = False
        if (
            self.motion_play_button is not None
            and self.motion_play_button.winfo_exists()
        ):
            self.motion_play_button.configure(text="▶ 播放 P")

    def open_motion_window(self, force_left: bool = False) -> None:
        if (
            self.motion_window is not None
            and self.motion_window.winfo_exists()
        ):
            if not self.motion_window.winfo_viewable():
                self.motion_window.deiconify()
                self.motion_window.lift()
                self.motion_window.focus_force()
            self._render_motion(force_left=force_left)
            return

        window = tk.Toplevel(self.root)
        window.title("局部运动对比")
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        width = min(1560, max(1080, screen_width - 100))
        height = min(880, max(620, screen_height - 120))
        x = max(20, (screen_width - width) // 2)
        y = max(20, (screen_height - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.minsize(980, 560)
        if self.icon is not None:
            try:
                window.iconphoto(True, self.icon)
            except tk.TclError:
                pass

        controls = ttk.Frame(window, padding=(7, 6))
        controls.pack(fill=tk.X)
        self.motion_play_button = ttk.Button(
            controls, text="▶ 播放 P", command=self.toggle_motion
        )
        self.motion_play_button.pack(side=tk.LEFT, padx=2)
        self.motion_event_start_button = ttk.Button(
            controls,
            text="回到新 ID 首次出现 X",
            command=self.go_motion_event_start,
        )
        self.motion_event_start_button.pack(side=tk.LEFT, padx=(2, 8))
        self.motion_draw_button = ttk.Button(
            controls,
            text="□ 画框 R",
            command=self.toggle_motion_draw_mode,
        )
        self.motion_draw_button.pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(
            controls, text="◀ 抽帧 A", command=self.previous_motion_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="抽帧 S ▶", command=self.next_motion_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="◀ 原帧 D", command=self.previous_raw_frame
        ).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(
            controls, text="原帧 F ▶", command=self.next_raw_frame
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(controls, text="速度：").pack(side=tk.LEFT, padx=(10, 2))
        speed_combo = ttk.Combobox(
            controls,
            textvariable=self.motion_speed,
            values=["0.5 FPS", "1 FPS", "2 FPS", "4 FPS"],
            state="readonly",
            width=7,
        )
        speed_combo.pack(side=tk.LEFT)
        options = ttk.Frame(window, padding=(7, 0, 7, 5))
        options.pack(fill=tk.X)
        ttk.Checkbutton(
            options,
            text="循环",
            variable=self.motion_loop,
        ).pack(side=tk.LEFT, padx=7)
        ttk.Checkbutton(
            options,
            text="跟随目标",
            variable=self.motion_follow,
            command=self._on_motion_follow_changed,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(
            options,
            text="显示类别名",
            variable=self.show_class_names,
            command=self._on_show_class_names_changed,
        ).pack(side=tk.LEFT, padx=7)
        ttk.Checkbutton(
            options,
            text="显示检测框 H",
            variable=self.motion_show_boxes,
            command=self._on_motion_show_boxes_changed,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(
            options,
            text="显示全部 ID Z",
            variable=self.motion_show_all_ids,
            command=self._on_motion_show_all_ids_changed,
        ).pack(side=tk.LEFT, padx=7)
        ttk.Checkbutton(
            options,
            text="显示轨迹 B",
            variable=self.motion_show_trajectory,
            command=self._on_motion_show_trajectory_changed,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            options,
            text="原视频目录",
            command=self.choose_raw_video_root,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            options,
            text="视野回到目标",
            command=self.reset_motion_pan,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Label(options, text="查找 ID:").pack(side=tk.LEFT, padx=(8, 2))
        self.motion_search_entry = ttk.Entry(
            options,
            textvariable=self.motion_search_id_var,
            width=7,
        )
        self.motion_search_entry.pack(side=tk.LEFT, padx=2)
        self.motion_search_entry.bind(
            "<Return>",
            lambda _event: (self.locate_motion_track_id(), "break")[1],
        )
        self.motion_search_entry.bind(
            "<Escape>",
            lambda _event: (
                self.release_motion_search_focus(),
                "break",
            )[1],
        )
        ttk.Button(
            options,
            text="定位",
            command=self.locate_motion_track_id,
        ).pack(side=tk.LEFT, padx=(2, 3))
        ttk.Label(
            options,
            textvariable=self.motion_zoom_var,
        ).pack(side=tk.LEFT, padx=(7, 3))
        ttk.Label(options, text="点大小:").pack(side=tk.LEFT, padx=(5, 1))
        marker_size_spinbox = ttk.Spinbox(
            options,
            from_=1,
            to=20,
            increment=1,
            width=3,
            textvariable=self.motion_marker_size_var,
            command=self._on_motion_marker_size_changed,
        )
        marker_size_spinbox.pack(side=tk.LEFT, padx=(0, 3))
        marker_size_spinbox.bind(
            "<Return>", self._on_motion_marker_size_changed
        )
        marker_size_spinbox.bind(
            "<FocusOut>", self._on_motion_marker_size_changed
        )
        ttk.Label(
            options,
            textvariable=self.motion_frame_var,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=tk.RIGHT, padx=(8, 4))
        ttk.Label(
            options, textvariable=self.motion_status_var
        ).pack(side=tk.RIGHT, padx=(4, 8))

        self.motion_review_bar = ttk.Frame(window, padding=(7, 0, 7, 5))
        ttk.Label(
            self.motion_review_bar,
            text="选择检查 ID：",
        ).pack(side=tk.LEFT, padx=(0, 2))
        self.motion_review_id_combo = ttk.Combobox(
            self.motion_review_bar,
            textvariable=self.motion_review_id_var,
            state="readonly",
            width=17,
        )
        self.motion_review_id_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.motion_review_id_combo.bind(
            "<<ComboboxSelected>>",
            self._on_motion_review_id_selected,
        )
        ttk.Button(
            self.motion_review_bar,
            text="◀ 上一轨迹 J",
            command=self.previous_trajectory,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            self.motion_review_bar,
            text="下一轨迹 K ▶",
            command=self.next_trajectory,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            self.motion_review_bar,
            text="✓ 通过 Space",
            style="Accent.TButton",
            command=lambda: self.mark_trajectory_review("passed"),
        ).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(
            self.motion_review_bar,
            text="⚠ 有问题 M",
            command=lambda: self.mark_trajectory_review("issue"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(
            self.motion_review_bar,
            textvariable=self.trajectory_reason_var,
        ).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(
            self.motion_review_bar,
            textvariable=self.trajectory_progress_var,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=4)

        views = ttk.Frame(window)
        self.motion_views = views
        views.pack(fill=tk.BOTH, expand=True, padx=7, pady=(0, 7))
        views.columnconfigure(0, weight=1, uniform="motion")
        views.columnconfigure(1, weight=1, uniform="motion")
        views.rowconfigure(0, weight=1)
        sampled_frame = ttk.LabelFrame(
            views, text="抽帧画面（左键选/画框，中键拖拽，Ctrl+左键编辑框）"
        )
        sampled_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 3))
        raw_frame = ttk.LabelFrame(views, text="原视频同步放大画面")
        self.motion_raw_frame_group = raw_frame
        raw_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(3, 0))
        self.motion_canvas = tk.Canvas(
            sampled_frame,
            background="#14171b",
            highlightthickness=0,
            cursor="hand2",
            takefocus=True,
        )
        self.motion_canvas.pack(fill=tk.BOTH, expand=True)
        self.motion_canvas.bind("<Configure>", self._schedule_redraw)
        self.motion_canvas.bind("<ButtonPress-1>", self._start_motion_drag)
        self.motion_canvas.bind("<B1-Motion>", self._drag_motion_view)
        self.motion_canvas.bind("<ButtonRelease-1>", self._end_motion_drag)
        self.motion_canvas.bind("<ButtonPress-2>", self._start_motion_middle_drag)
        self.motion_canvas.bind("<B2-Motion>", self._drag_motion_view)
        self.motion_canvas.bind("<ButtonRelease-2>", self._end_motion_middle_drag)
        self.motion_canvas.bind("<Button-3>", self._on_motion_right_click)
        self.motion_canvas.bind("<MouseWheel>", self._on_motion_mousewheel)
        self.raw_canvas = tk.Canvas(
            raw_frame,
            background="#14171b",
            highlightthickness=0,
            cursor="hand2",
            takefocus=True,
        )
        self.raw_canvas.pack(fill=tk.BOTH, expand=True)
        self.raw_canvas.bind("<Configure>", self._schedule_redraw)
        self.raw_canvas.bind("<ButtonPress-1>", self._start_motion_drag)
        self.raw_canvas.bind("<B1-Motion>", self._drag_motion_view)
        self.raw_canvas.bind("<ButtonRelease-1>", self._end_motion_drag)
        self.raw_canvas.bind("<ButtonPress-2>", self._start_motion_middle_drag)
        self.raw_canvas.bind("<B2-Motion>", self._drag_motion_view)
        self.raw_canvas.bind("<ButtonRelease-2>", self._end_motion_middle_drag)
        self.raw_canvas.bind("<MouseWheel>", self._on_motion_mousewheel)
        timeline_frame = ttk.LabelFrame(
            window,
            text="轨迹时间轴（点击任意格跳转；红边表示疑似异常）",
            padding=(5, 3),
        )
        self.trajectory_timeline_canvas = tk.Canvas(
            timeline_frame,
            height=44,
            background="#14171b",
            highlightthickness=0,
            cursor="hand2",
        )
        self.trajectory_timeline_canvas.pack(fill=tk.X, expand=True)
        self.trajectory_timeline_canvas.bind(
            "<Configure>", lambda _event: self._render_trajectory_timeline()
        )
        self.trajectory_timeline_canvas.bind(
            "<Button-1>", self._on_trajectory_timeline_click
        )
        if self._is_trajectory_mode():
            self.motion_review_bar.pack(
                fill=tk.X, padx=7, pady=(0, 5), before=views
            )
            timeline_frame.pack(fill=tk.X, padx=7, pady=(0, 7))
        window.bind(
            "<p>",
            lambda event: self._run_motion_shortcut(
                event, self.toggle_motion
            ),
        )
        window.bind(
            "<a>",
            lambda event: self._run_motion_shortcut(
                event, self.previous_motion_frame
            ),
        )
        window.bind(
            "<s>",
            lambda event: self._run_motion_shortcut(
                event, self.next_motion_frame
            ),
        )
        window.bind(
            "<d>",
            lambda event: self._run_motion_shortcut(
                event, self.previous_raw_frame
            ),
        )
        window.bind(
            "<f>",
            self._on_motion_next_raw_shortcut,
        )
        window.bind(
            "<z>",
            self._on_motion_toggle_ids_shortcut,
        )
        window.bind(
            "<h>",
            self._on_motion_toggle_boxes_shortcut,
        )
        window.bind(
            "<b>",
            self._on_motion_toggle_trajectory_shortcut,
        )
        window.bind(
            "<x>",
            lambda event: self._run_motion_shortcut(
                event, self.go_motion_event_start
            ),
        )
        window.bind(
            "<r>",
            lambda event: self._run_motion_shortcut(
                event, self.toggle_motion_draw_mode
            ),
        )
        window.bind("<Escape>", self._on_motion_escape_shortcut)
        window.bind(
            "<Control-z>",
            lambda event: self._run_motion_shortcut(event, self.undo),
        )
        window.bind(
            "<Control-f>",
            lambda _event: (
                self.focus_motion_search(),
                "break",
            )[1],
        )
        window.bind(
            "<j>",
            lambda event: self._run_motion_shortcut(
                event, self.previous_trajectory
            ),
        )
        window.bind(
            "<k>",
            lambda event: self._run_motion_shortcut(
                event, self.next_trajectory
            ),
        )
        window.bind("<space>", self._on_trajectory_pass_shortcut)
        window.bind("<m>", self._on_trajectory_issue_shortcut)
        window.protocol("WM_DELETE_WINDOW", self._close_motion_window)
        self.motion_window = window
        self._update_mode_ui()
        self._sync_motion_review_id_choices()
        self._render_motion(force_left=True)

    def _close_motion_window(self) -> None:
        self._stop_motion()
        if self.motion_pan_render_job is not None:
            self.root.after_cancel(self.motion_pan_render_job)
            self.motion_pan_render_job = None
        if self.motion_window is not None and self.motion_window.winfo_exists():
            self.motion_window.destroy()
        self.motion_window = None
        self.motion_canvas = None
        self.raw_canvas = None
        self.motion_play_button = None
        self.motion_event_start_button = None
        self.motion_draw_button = None
        self.motion_search_entry = None
        self.motion_review_id_combo = None
        self.motion_raw_frame_group = None
        self.motion_review_bar = None
        self.motion_views = None
        self.trajectory_timeline_canvas = None
        self.motion_photo = None
        self.motion_transform = None
        self.raw_motion_transform = None
        self.motion_drag_last = None
        self.motion_draw_mode.set(False)
        self.motion_box_start = None
        self.motion_preview_item = None
        self.motion_edit_occurrence = None
        self.motion_edit_mode = None
        self.motion_edit_start_point = None
        self.motion_edit_original_rect = None
        self.motion_edit_preview_rect = None
        self.raw_photo = None
        self.motion_left_render_key = None

    def toggle_motion_draw_mode(self) -> None:
        enabled = not self.motion_draw_mode.get()
        self.motion_draw_mode.set(enabled)
        self.motion_box_start = None
        if (
            self.motion_preview_item is not None
            and self.motion_canvas is not None
            and self.motion_canvas.winfo_exists()
        ):
            self.motion_canvas.delete(self.motion_preview_item)
        self.motion_preview_item = None
        if self.motion_draw_button is not None:
            self.motion_draw_button.configure(
                text="■ 退出画框 R" if enabled else "□ 画框 R"
            )
        if self.motion_canvas is not None and self.motion_canvas.winfo_exists():
            self.motion_canvas.configure(cursor="crosshair" if enabled else "hand2")
        self.motion_status_var.set(
            "画框模式：左键拖框，松开后输入 ID"
            if enabled
            else "已退出画框模式"
        )

    def cancel_motion_draw(self) -> None:
        if not self.motion_draw_mode.get() and self.motion_box_start is None:
            return
        self.motion_draw_mode.set(False)
        self.motion_box_start = None
        if (
            self.motion_preview_item is not None
            and self.motion_canvas is not None
            and self.motion_canvas.winfo_exists()
        ):
            self.motion_canvas.delete(self.motion_preview_item)
        self.motion_preview_item = None
        if self.motion_draw_button is not None:
            self.motion_draw_button.configure(text="□ 画框 R")
        if self.motion_canvas is not None and self.motion_canvas.winfo_exists():
            self.motion_canvas.configure(cursor="hand2")
        self.motion_status_var.set("已取消画框")

    def toggle_motion(self) -> None:
        if not self.motion_sequence:
            self._reset_motion()
        if not self.motion_sequence:
            return
        if (
            self.motion_window is None
            or not self.motion_window.winfo_exists()
        ):
            self.open_motion_window()
        if self.motion_playing:
            self._stop_motion()
            return
        if self.raw_frame_index >= self.raw_end_frame:
            self.raw_frame_index = self.raw_play_start_frame
            self._sync_motion_index_from_raw()
        self.motion_playing = True
        if self.motion_play_button is not None:
            self.motion_play_button.configure(text="⏸ 暂停 P")
        if self.motion_window is not None:
            self.motion_window.lift()
        self._render_motion()
        self._schedule_motion_tick()

    def _motion_delay_ms(self) -> int:
        try:
            fps = float(self.motion_speed.get().split()[0])
        except (ValueError, IndexError):
            fps = 2.0
        raw_fps = max(0.1, fps * self.sample_gap)
        return max(35, int(round(1000 / raw_fps)))

    def _schedule_motion_tick(self) -> None:
        if self.motion_playing:
            self.motion_job = self.root.after(
                self._motion_delay_ms(), self._motion_tick
            )

    def _motion_tick(self) -> None:
        self.motion_job = None
        if not self.motion_playing or not self.motion_sequence:
            return
        if self.raw_frame_index >= self.raw_end_frame:
            if self.motion_loop.get():
                self.raw_frame_index = self.raw_play_start_frame
            else:
                self._stop_motion()
                return
        else:
            self.raw_frame_index += 1
        self._sync_motion_index_from_raw()
        self._render_motion()
        self._schedule_motion_tick()

    def go_motion_event_start(self) -> None:
        self._stop_motion()
        self._reset_motion()
        if not self.motion_sequence:
            return
        self.motion_index = self.motion_event_start_index
        self.raw_frame_index = self.raw_play_start_frame
        self.open_motion_window(force_left=True)

    def toggle_motion_all_ids(self) -> None:
        self.motion_show_all_ids.set(not self.motion_show_all_ids.get())
        self._on_motion_show_all_ids_changed()

    def toggle_motion_boxes(self) -> None:
        self.motion_show_boxes.set(not self.motion_show_boxes.get())
        self._on_motion_show_boxes_changed()

    def _on_motion_show_boxes_changed(self) -> None:
        if not self.motion_show_boxes.get():
            self.motion_edit_occurrence = None
            self.motion_edit_mode = None
            self.motion_edit_preview_rect = None
        self.motion_left_render_key = None
        self._render_motion(force_left=True)

    def _on_motion_show_all_ids_changed(self) -> None:
        self.motion_left_render_key = None
        self._render_motion(force_left=True)

    def toggle_motion_trajectory(self) -> None:
        self.motion_show_trajectory.set(not self.motion_show_trajectory.get())
        self._on_motion_show_trajectory_changed()

    def _on_motion_show_trajectory_changed(self) -> None:
        self.motion_left_render_key = None
        self._render_motion(force_left=True)

    def _motion_marker_radius(self) -> int:
        return self.motion_marker_radius_value

    def _on_motion_marker_size_written(self, *_args) -> None:
        try:
            radius = int(float(self.motion_marker_size_var.get()))
        except (ValueError, tk.TclError):
            return
        if not 1 <= radius <= 20:
            return
        self.motion_marker_radius_value = radius
        self.settings["motion_marker_radius"] = radius
        self._save_settings()
        self._render_motion()

    def _on_motion_marker_size_changed(self, _event=None) -> None:
        try:
            radius = int(float(self.motion_marker_size_var.get()))
        except (ValueError, tk.TclError):
            radius = self.motion_marker_radius_value
        radius = min(20, max(1, radius))
        self.motion_marker_size_var.set(str(radius))

    def _render_trajectory_timeline(self) -> None:
        canvas = self.trajectory_timeline_canvas
        event = self.current_event()
        if canvas is not None and canvas.winfo_exists():
            canvas.delete("all")
        if (
            canvas is None
            or not canvas.winfo_exists()
            or not self._is_trajectory_mode()
            or event is None
            or not self.images
        ):
            self.trajectory_reason_var.set("")
            return
        width = max(2, canvas.winfo_width())
        height = max(2, canvas.winfo_height())
        padding = 7.0
        slot_width = max(1.0, (width - padding * 2) / len(self.images))
        current_frame = (
            self.motion_sequence[self.motion_index]
            if self.motion_sequence
            else event.first_frame_index
        )
        present_frames = {
            occurrence.frame_index
            for occurrence in self.tracks.get(event.track_id, [])
        }
        metrics = self.trajectory_metrics.get(event.track_id)
        anomaly_frames = set(metrics.anomaly_frames) if metrics else set()
        for frame_index in range(len(self.images)):
            x1 = padding + frame_index * slot_width
            x2 = padding + (frame_index + 1) * slot_width - 2
            fill = "#19c37d" if frame_index in present_frames else "#444a50"
            if frame_index == current_frame:
                fill = "#ffd43b"
            outline = "#ff4545" if frame_index in anomaly_frames else "#17191d"
            canvas.create_rectangle(
                x1,
                5,
                max(x1 + 1, x2),
                height - 5,
                fill=fill,
                outline=outline,
                width=3 if frame_index in anomaly_frames else 1,
            )
            if slot_width >= 24:
                canvas.create_text(
                    (x1 + x2) / 2,
                    height / 2,
                    text=str(frame_index + 1),
                    fill="#111111" if frame_index == current_frame else "#ffffff",
                    font=("Microsoft YaHei UI", 8, "bold"),
                )
        reasons = "；".join(metrics.reasons) if metrics and metrics.reasons else "未发现规则异常"
        self.trajectory_reason_var.set(
            f"ID {event.track_id}：{reasons}"
        )

    def _on_trajectory_timeline_click(self, event) -> str:
        self.release_motion_search_focus()
        if not self._is_trajectory_mode() or not self.images:
            return "break"
        canvas = self.trajectory_timeline_canvas
        if canvas is None:
            return "break"
        padding = 7.0
        usable_width = max(1.0, canvas.winfo_width() - padding * 2)
        frame_index = int(
            (event.x - padding) / usable_width * len(self.images)
        )
        frame_index = min(max(0, frame_index), len(self.images) - 1)
        self._stop_motion()
        if frame_index in self.motion_sequence:
            self.motion_index = self.motion_sequence.index(frame_index)
        else:
            self._set_motion_sequence_around(frame_index)
        self.raw_frame_index = self.motion_source_frames[self.motion_index]
        self.motion_focus_frame_index = frame_index
        self.motion_left_render_key = None
        self._render_motion(force_left=True)
        return "break"

    def previous_motion_frame(self) -> None:
        self._stop_motion()
        if self.motion_sequence:
            self.motion_index = max(0, self.motion_index - 1)
            self.raw_frame_index = self.motion_source_frames[self.motion_index]
            self.open_motion_window(force_left=True)

    def next_motion_frame(self) -> None:
        self._stop_motion()
        if self.motion_sequence:
            self.motion_index = min(
                len(self.motion_sequence) - 1, self.motion_index + 1
            )
            self.raw_frame_index = self.motion_source_frames[self.motion_index]
            self.open_motion_window(force_left=True)

    def previous_raw_frame(self) -> None:
        self._stop_motion()
        if self.motion_sequence:
            self.raw_frame_index = max(
                self.raw_start_frame,
                self.raw_frame_index - 1,
            )
            self._sync_motion_index_from_raw()
            self.open_motion_window()

    def next_raw_frame(self) -> None:
        self._stop_motion()
        if self.motion_sequence:
            self.raw_frame_index = min(
                self.raw_end_frame,
                self.raw_frame_index + 1,
            )
            self._sync_motion_index_from_raw()
            self.open_motion_window()

    def _motion_fixed_crop(self, event: ReviewEvent, image_size) -> tuple[int, int, int, int]:
        old_id = self._selected_old_id()
        old_last = (
            last_occurrence_before(self.tracks, old_id, event.first_frame_index)
            if old_id is not None
            else None
        )
        start_anchor = old_last.frame_index if old_last else event.first_frame_index
        start = max(0, start_anchor - 2)
        end = min(len(self.images) - 1, event.first_frame_index + 4)
        occurrences = [
            occurrence
            for track_id in (old_id, event.track_id)
            if track_id is not None
            for occurrence in self.tracks.get(track_id, [])
            if start <= occurrence.frame_index <= end
        ]
        if not occurrences:
            return self._crop_around(image_size, event.rect)
        x1 = min(item.rect[0] for item in occurrences)
        y1 = min(item.rect[1] for item in occurrences)
        x2 = max(item.rect[2] for item in occurrences)
        y2 = max(item.rect[3] for item in occurrences)
        max_width = max(item.rect[2] - item.rect[0] for item in occurrences)
        max_height = max(item.rect[3] - item.rect[1] for item in occurrences)
        image_width, image_height = image_size
        crop_width = min(
            image_width,
            max(360.0, x2 - x1 + 180.0, max_width * 8.0),
        )
        crop_height = min(
            image_height,
            max(280.0, y2 - y1 + 150.0, max_height * 7.0),
        )
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        crop_x1 = min(max(0.0, center_x - crop_width / 2), image_width - crop_width)
        crop_y1 = min(max(0.0, center_y - crop_height / 2), image_height - crop_height)
        return (
            int(round(crop_x1)),
            int(round(crop_y1)),
            int(round(crop_x1 + crop_width)),
            int(round(crop_y1 + crop_height)),
        )

    def _motion_crop(self, event: ReviewEvent, frame_index: int, image_size):
        if not self.motion_follow.get():
            return self._motion_fixed_crop(event, image_size)
        old_id = self._selected_old_id()
        active_id = (
            old_id
            if frame_index < event.first_frame_index
            else event.track_id
        )
        occurrence = (
            find_occurrence(self.tracks, active_id, frame_index)
            if active_id is not None
            else None
        )
        return self._crop_around(
            image_size,
            occurrence.rect if occurrence else event.rect,
        )

    def _initialize_motion_view(
        self,
        event: ReviewEvent,
        frame_index: int,
        image_size,
    ) -> None:
        if self.motion_canvas is None:
            return
        width = self.motion_canvas.winfo_width()
        height = self.motion_canvas.winfo_height()
        if width < 100 or height < 100:
            return
        focus_crop = self._motion_crop(event, frame_index, image_size)
        zoom, pan = self._motion_initial_view(
            (width, height),
            image_size,
            focus_crop,
        )
        self.motion_zoom = zoom
        self.motion_pan_offset = [pan[0], pan[1]]
        self.motion_view_initialized = True
        self.motion_focus_frame_index = frame_index
        self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")

    def _on_motion_follow_changed(self) -> None:
        self.motion_view_initialized = False
        self.motion_focus_frame_index = None
        self.motion_left_render_key = None
        self._render_motion(force_left=True)

    def reset_motion_pan(self) -> None:
        self.motion_pan_offset = [0.0, 0.0]
        self.motion_zoom = 1.0
        self.motion_view_initialized = False
        self.motion_focus_frame_index = None
        self.motion_zoom_var.set("缩放 1.00×（整图）")
        self.motion_left_render_key = None
        self._render_motion(force_left=True)

    def _on_motion_mousewheel(self, event) -> str:
        if event.delta == 0:
            return "break"
        self._stop_motion()
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self.motion_zoom = min(6.0, max(1.0, self.motion_zoom * factor))
        self.motion_view_initialized = True
        self.motion_focus_frame_index = self.motion_sequence[self.motion_index]
        suffix = "（整图）" if self.motion_zoom <= 1.0001 else ""
        self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×{suffix}")
        self.motion_left_render_key = None
        self._render_motion(force_left=True)
        return "break"

    def focus_motion_search(self) -> None:
        if (
            self.motion_search_entry is not None
            and self.motion_search_entry.winfo_exists()
        ):
            self.motion_search_entry.focus_set()
            self.motion_search_entry.selection_range(0, tk.END)

    def release_motion_search_focus(self) -> None:
        search_entry = getattr(self, "motion_search_entry", None)
        if (
            search_entry is not None
            and search_entry.winfo_exists()
        ):
            search_entry.selection_clear()
        motion_canvas = getattr(self, "motion_canvas", None)
        motion_window = getattr(self, "motion_window", None)
        if (
            motion_canvas is not None
            and hasattr(motion_canvas, "focus_set")
            and (
                not hasattr(motion_canvas, "winfo_exists")
                or motion_canvas.winfo_exists()
            )
        ):
            motion_canvas.focus_set()
        elif (
            motion_window is not None
            and hasattr(motion_window, "focus_set")
            and (
                not hasattr(motion_window, "winfo_exists")
                or motion_window.winfo_exists()
            )
        ):
            motion_window.focus_set()

    def _run_motion_shortcut(self, event, action):
        if self._shortcut_is_typing(event):
            return None
        action()
        return "break"

    def _on_motion_escape_shortcut(self, event):
        if self._shortcut_is_typing(event):
            self.release_motion_search_focus()
        else:
            self.cancel_motion_draw()
        return "break"

    def _on_motion_next_raw_shortcut(self, event):
        if event.state & 0x4 or self._shortcut_is_typing(event):
            return None
        self.next_raw_frame()
        return "break"

    def _on_motion_toggle_ids_shortcut(self, event):
        if event.state & 0x4 or self._shortcut_is_typing(event):
            return None
        self.toggle_motion_all_ids()
        return "break"

    def _on_motion_toggle_boxes_shortcut(self, event):
        if event.state & 0x4 or self._shortcut_is_typing(event):
            return None
        self.toggle_motion_boxes()
        return "break"

    def _on_motion_toggle_trajectory_shortcut(self, event):
        if event.state & 0x4 or self._shortcut_is_typing(event):
            return None
        self.toggle_motion_trajectory()
        return "break"

    @staticmethod
    def _nearest_occurrence(occurrences, frame_index: int):
        return min(
            occurrences,
            key=lambda item: abs(item.frame_index - frame_index),
        )

    def _set_motion_sequence_around(self, frame_index: int) -> None:
        """加载当前区段的全部抽帧，并定位到指定帧。"""
        self.motion_sequence = list(range(len(self.images)))
        self.motion_source_frames = [
            self._source_frame_number(index) for index in self.motion_sequence
        ]
        gaps = [
            right - left
            for left, right in zip(
                self.motion_source_frames, self.motion_source_frames[1:]
            )
            if right > left
        ]
        self.sample_gap = max(1, sorted(gaps)[len(gaps) // 2]) if gaps else 5
        self.motion_index = min(max(0, frame_index), len(self.motion_sequence) - 1)
        self.raw_start_frame = self.motion_source_frames[0]
        self.raw_play_start_frame = self.motion_source_frames[self.motion_index]
        self.raw_end_frame = self.motion_source_frames[-1] + self.sample_gap - 1
        self.raw_frame_index = self.raw_play_start_frame

    def locate_motion_track_id(self) -> None:
        try:
            track_id = int(self.motion_search_id_var.get().strip())
        except ValueError:
            self.motion_status_var.set("请输入有效的数字 ID")
            self.focus_motion_search()
            return
        occurrences = self.tracks.get(track_id, [])
        if not occurrences:
            self.motion_status_var.set(f"当前区段不存在 ID {track_id}")
            self.focus_motion_search()
            return
        current_frame = (
            self.motion_sequence[self.motion_index]
            if self.motion_sequence
            else self.display_frame_index
        )
        occurrence = self._nearest_occurrence(occurrences, current_frame)
        if occurrence.frame_index not in self.motion_sequence:
            self._set_motion_sequence_around(occurrence.frame_index)
        else:
            self.motion_index = self.motion_sequence.index(occurrence.frame_index)
            self.raw_frame_index = self.motion_source_frames[self.motion_index]

        image = self._get_image(occurrence.frame_index)
        focus_crop = self._crop_around(image.size, occurrence.rect)
        width = max(2, self.motion_canvas.winfo_width()) if self.motion_canvas else 2
        height = max(2, self.motion_canvas.winfo_height()) if self.motion_canvas else 2
        zoom, pan = self._motion_initial_view(
            (width, height),
            image.size,
            focus_crop,
        )
        self.motion_search_track_id = track_id
        self.motion_zoom = zoom
        self.motion_pan_offset = [pan[0], pan[1]]
        self.motion_view_initialized = True
        self.motion_focus_frame_index = occurrence.frame_index
        self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
        self.motion_left_render_key = None
        self._render_motion(force_left=True)
        self.motion_status_var.set(
            f"已定位 ID {track_id}：第 {occurrence.frame_index + 1} 张"
        )
        self.release_motion_search_focus()

    def _start_motion_drag(self, event) -> None:
        self.release_motion_search_focus()
        if (
            event.widget == self.motion_canvas
            and event.state & 0x4
            and self.motion_transform is not None
        ):
            self._start_motion_box_edit(event)
            return
        if (
            event.widget == self.motion_canvas
            and self.motion_draw_mode.get()
            and self.motion_transform is not None
        ):
            self._stop_motion()
            self.motion_box_start = (event.x, event.y)
            if self.motion_preview_item is not None:
                self.motion_canvas.delete(self.motion_preview_item)
            self.motion_preview_item = self.motion_canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="#00ff7f",
                width=2,
                dash=(6, 3),
            )
            return
        self._start_motion_pan_drag(event)

    def _start_motion_middle_drag(self, event) -> str:
        self.release_motion_search_focus()
        self._start_motion_pan_drag(event)
        return "break"

    def _start_motion_pan_drag(self, event) -> None:
        transform = (
            self.motion_transform
            if event.widget == self.motion_canvas
            else self.raw_motion_transform
        )
        if transform is None:
            return
        self._stop_motion()
        self.motion_drag_last = (event.x, event.y)
        self.motion_drag_total = 0.0
        self.motion_drag_scale = max(0.001, transform[0])
        event.widget.configure(cursor="fleur")

    def _drag_motion_view(self, event) -> None:
        if (
            event.widget == self.motion_canvas
            and self.motion_edit_mode is not None
        ):
            if event.state & 0x4:
                self._drag_motion_box_edit(event)
            return
        if (
            event.widget == self.motion_canvas
            and self.motion_box_start is not None
            and self.motion_preview_item is not None
        ):
            self.motion_canvas.coords(
                self.motion_preview_item,
                self.motion_box_start[0],
                self.motion_box_start[1],
                event.x,
                event.y,
            )
            return
        if self.motion_drag_last is None:
            return
        delta_x = event.x - self.motion_drag_last[0]
        delta_y = event.y - self.motion_drag_last[1]
        self.motion_drag_last = (event.x, event.y)
        self.motion_drag_total += abs(delta_x) + abs(delta_y)
        self.motion_pan_offset[0] -= delta_x / self.motion_drag_scale
        self.motion_pan_offset[1] -= delta_y / self.motion_drag_scale
        self.motion_left_render_key = None
        if self.motion_pan_render_job is None:
            self.motion_pan_render_job = self.root.after(
                16, self._render_motion_pan
            )

    def _render_motion_pan(self) -> None:
        self.motion_pan_render_job = None
        if not self._move_rendered_motion_view():
            self._render_motion(force_left=True)

    def _move_rendered_motion_view(self) -> bool:
        """拖拽时只移动现有画面，松手后再执行完整高质量重绘。"""
        if (
            self.motion_canvas is None
            or self.raw_canvas is None
            or self.motion_transform is None
        ):
            return False
        canvas = self.motion_canvas
        if not canvas.find_withtag("motion_pan_content"):
            return False

        old_scale, old_x, old_y, crop, frame_index = self.motion_transform
        image_size = (crop[2] - crop[0], crop[3] - crop[1])
        _size, scale, x, y, pan = self._motion_fit_geometry(
            (max(2, canvas.winfo_width()), max(2, canvas.winfo_height())),
            image_size,
            self.motion_zoom,
            self.motion_pan_offset,
        )
        self.motion_pan_offset = [pan[0], pan[1]]
        canvas.move("motion_pan_content", x - old_x, y - old_y)
        self.motion_transform = (scale, x, y, crop, frame_index)
        self._draw_motion_edit_selection()

        if (
            self.raw_motion_transform is not None
            and self.raw_canvas.find_withtag("raw_pan_content")
        ):
            (
                _raw_old_scale,
                raw_old_x,
                raw_old_y,
                raw_crop,
                raw_frame_index,
            ) = self.raw_motion_transform
            raw_size = (
                raw_crop[2] - raw_crop[0],
                raw_crop[3] - raw_crop[1],
            )
            _size, raw_scale, raw_x, raw_y, raw_pan = (
                self._motion_fit_geometry(
                    (
                        max(2, self.raw_canvas.winfo_width()),
                        max(2, self.raw_canvas.winfo_height()),
                    ),
                    raw_size,
                    self.motion_zoom,
                    self.motion_pan_offset,
                )
            )
            self.motion_pan_offset = [raw_pan[0], raw_pan[1]]
            self.raw_canvas.move(
                "raw_pan_content",
                raw_x - raw_old_x,
                raw_y - raw_old_y,
            )
            self.raw_motion_transform = (
                raw_scale,
                raw_x,
                raw_y,
                raw_crop,
                raw_frame_index,
            )
        return True

    def _end_motion_drag(self, event) -> None:
        if (
            event.widget == self.motion_canvas
            and self.motion_edit_mode is not None
        ):
            self._end_motion_box_edit()
            return
        if (
            event.widget == self.motion_canvas
            and self.motion_box_start is not None
            and self.motion_transform is not None
        ):
            start = self.motion_box_start
            self.motion_box_start = None
            if self.motion_preview_item is not None:
                self.motion_canvas.delete(self.motion_preview_item)
            self.motion_preview_item = None
            scale, offset_x, offset_y, crop, frame_index = self.motion_transform
            crop_width = crop[2] - crop[0]
            crop_height = crop[3] - crop[1]

            def original_point(canvas_x, canvas_y):
                local_x = min(max(0.0, (canvas_x - offset_x) / scale), crop_width)
                local_y = min(max(0.0, (canvas_y - offset_y) / scale), crop_height)
                return local_x + crop[0], local_y + crop[1]

            first = original_point(*start)
            second = original_point(event.x, event.y)
            rect = (
                min(first[0], second[0]),
                min(first[1], second[1]),
                max(first[0], second[0]),
                max(first[1], second[1]),
            )
            if rect[2] - rect[0] < 3 or rect[3] - rect[1] < 3:
                self.motion_status_var.set("检测框太小，未添加")
                return
            self._add_motion_box(frame_index, rect)
            return

        dragged = self.motion_drag_total >= 4.0
        self.motion_drag_last = None
        event.widget.configure(
            cursor="hand2"
        )
        if self.motion_pan_render_job is not None:
            self.root.after_cancel(self.motion_pan_render_job)
            self.motion_pan_render_job = None
        if dragged:
            self._render_motion(force_left=True)
        elif event.widget == self.motion_canvas:
            self._on_motion_click(event)

    def _end_motion_middle_drag(self, event) -> str:
        dragged = self.motion_drag_total >= 4.0
        self.motion_drag_last = None
        cursor = (
            "crosshair"
            if event.widget == self.motion_canvas and self.motion_draw_mode.get()
            else "hand2"
        )
        event.widget.configure(cursor=cursor)
        if self.motion_pan_render_job is not None:
            self.root.after_cancel(self.motion_pan_render_job)
            self.motion_pan_render_job = None
        if dragged:
            self._render_motion(force_left=True)
        return "break"

    def _motion_record_at_event(self, event):
        current = self.current_event()
        if self.motion_transform is None or current is None:
            return None
        point = self._canvas_to_image(event, self.motion_transform)
        if point is None:
            return None
        crop = self.motion_transform[3]
        frame_index = self.motion_transform[4]
        original_point = (point[0] + crop[0], point[1] + crop[1])
        records = [
            record
            for record in bee_occurrences(
                self.documents[frame_index], frame_index
            )
            if point_in_rect(original_point, record.rect)
        ]
        if not records:
            return None
        return min(records, key=lambda record: rect_area(record.rect))

    def _motion_canvas_point_to_image(self, event):
        if self.motion_transform is None:
            return None
        point = self._canvas_to_image(event, self.motion_transform)
        if point is None:
            return None
        crop = self.motion_transform[3]
        frame_index = self.motion_transform[4]
        image = self._get_image(frame_index)
        return (
            min(max(0.0, point[0] + crop[0]), float(image.width)),
            min(max(0.0, point[1] + crop[1]), float(image.height)),
        )

    def _motion_edit_canvas_rect(self, rect):
        if self.motion_transform is None:
            return None
        scale, offset_x, offset_y, crop, _frame_index = self.motion_transform
        return (
            offset_x + (rect[0] - crop[0]) * scale,
            offset_y + (rect[1] - crop[1]) * scale,
            offset_x + (rect[2] - crop[0]) * scale,
            offset_y + (rect[3] - crop[1]) * scale,
        )

    @staticmethod
    def _motion_handle_points(canvas_rect):
        x1, y1, x2, y2 = canvas_rect
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return {
            "nw": (x1, y1),
            "n": (center_x, y1),
            "ne": (x2, y1),
            "e": (x2, center_y),
            "se": (x2, y2),
            "s": (center_x, y2),
            "sw": (x1, y2),
            "w": (x1, center_y),
        }

    def _draw_motion_edit_selection(self) -> None:
        canvas = self.motion_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        canvas.delete("motion_edit")
        selected = self.motion_edit_occurrence
        if (
            not self.motion_show_boxes.get()
            or selected is None
            or self.motion_transform is None
            or selected.frame_index != self.motion_transform[4]
        ):
            return
        rect = self.motion_edit_preview_rect or selected.rect
        canvas_rect = self._motion_edit_canvas_rect(rect)
        if canvas_rect is None:
            return
        canvas.create_rectangle(
            *canvas_rect,
            outline="#ffb000",
            width=3,
            dash=(7, 3),
            tags="motion_edit",
        )
        radius = 5
        for x, y in self._motion_handle_points(canvas_rect).values():
            canvas.create_rectangle(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#ffb000",
                outline="#1b1b1b",
                width=1,
                tags="motion_edit",
            )

    def _motion_edit_handle_at(self, event) -> str | None:
        selected = self.motion_edit_occurrence
        if (
            selected is None
            or self.motion_transform is None
            or selected.frame_index != self.motion_transform[4]
        ):
            return None
        rect = self.motion_edit_preview_rect or selected.rect
        canvas_rect = self._motion_edit_canvas_rect(rect)
        if canvas_rect is None:
            return None
        for name, (x, y) in self._motion_handle_points(canvas_rect).items():
            if abs(event.x - x) <= 9 and abs(event.y - y) <= 9:
                return name
        return None

    def _start_motion_box_edit(self, event) -> None:
        point = self._motion_canvas_point_to_image(event)
        if point is None:
            return
        self._stop_motion()
        handle = self._motion_edit_handle_at(event)
        if handle is None:
            selected = self._motion_record_at_event(event)
            if selected is None:
                self.motion_edit_occurrence = None
                self.motion_edit_preview_rect = None
                self.motion_edit_mode = None
                self.motion_left_render_key = None
                self._render_motion(force_left=True)
                self.motion_status_var.set("Ctrl 点击未选中检测框")
                return
            self.motion_edit_occurrence = selected
            self.motion_edit_original_rect = selected.rect
            self.motion_edit_preview_rect = selected.rect
            self.motion_edit_mode = "move"
        else:
            self.motion_edit_original_rect = (
                self.motion_edit_preview_rect
                or self.motion_edit_occurrence.rect
            )
            self.motion_edit_mode = handle
        self.motion_edit_start_point = point
        self.motion_canvas.configure(
            cursor="fleur" if self.motion_edit_mode == "move" else "sizing"
        )
        self.motion_left_render_key = None
        self._render_motion(force_left=True)
        self.motion_status_var.set(
            f"正在编辑 ID {self.motion_edit_occurrence.track_id}；松开鼠标自动保存"
        )

    @staticmethod
    def _move_rect_within_image(rect, delta_x, delta_y, image_size):
        image_width, image_height = image_size
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        x1 = min(max(0.0, rect[0] + delta_x), image_width - width)
        y1 = min(max(0.0, rect[1] + delta_y), image_height - height)
        return x1, y1, x1 + width, y1 + height

    @staticmethod
    def _resize_rect_within_image(
        rect,
        handle: str,
        point,
        image_size,
        minimum_size: float = 4.0,
    ):
        image_width, image_height = image_size
        x1, y1, x2, y2 = rect
        point_x = min(max(0.0, point[0]), float(image_width))
        point_y = min(max(0.0, point[1]), float(image_height))
        if "w" in handle:
            x1 = min(point_x, x2 - minimum_size)
        if "e" in handle:
            x2 = max(point_x, x1 + minimum_size)
        if "n" in handle:
            y1 = min(point_y, y2 - minimum_size)
        if "s" in handle:
            y2 = max(point_y, y1 + minimum_size)
        return x1, y1, x2, y2

    def _drag_motion_box_edit(self, event) -> None:
        selected = self.motion_edit_occurrence
        point = self._motion_canvas_point_to_image(event)
        if (
            selected is None
            or point is None
            or self.motion_edit_original_rect is None
            or self.motion_edit_start_point is None
        ):
            return
        image = self._get_image(selected.frame_index)
        if self.motion_edit_mode == "move":
            delta_x = point[0] - self.motion_edit_start_point[0]
            delta_y = point[1] - self.motion_edit_start_point[1]
            self.motion_edit_preview_rect = self._move_rect_within_image(
                self.motion_edit_original_rect,
                delta_x,
                delta_y,
                image.size,
            )
        else:
            self.motion_edit_preview_rect = self._resize_rect_within_image(
                self.motion_edit_original_rect,
                self.motion_edit_mode,
                point,
                image.size,
            )
        self._draw_motion_edit_selection()

    def _end_motion_box_edit(self) -> None:
        selected = self.motion_edit_occurrence
        original_rect = self.motion_edit_original_rect
        new_rect = self.motion_edit_preview_rect
        self.motion_edit_mode = None
        self.motion_edit_start_point = None
        self.motion_edit_original_rect = None
        if self.motion_canvas is not None:
            self.motion_canvas.configure(
                cursor="crosshair" if self.motion_draw_mode.get() else "hand2"
            )
        if (
            selected is None
            or original_rect is None
            or new_rect is None
            or all(
                abs(old_value - new_value) < 0.05
                for old_value, new_value in zip(original_rect, new_rect)
            )
        ):
            self._draw_motion_edit_selection()
            return
        self._save_motion_box_transform(selected, new_rect)

    def _save_motion_box_transform(self, selected, new_rect) -> None:
        if not self.section or not self._ensure_edit_backup():
            self.motion_edit_preview_rect = selected.rect
            self._draw_motion_edit_selection()
            return
        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_trajectory_reviews = dict(self.trajectory_reviews)
        old_index = self.event_index
        active_event = self.current_event()
        active_track_id = active_event.track_id if active_event else None
        active_first_frame = (
            active_event.first_frame_index if active_event else None
        )
        selected_old_id = self._selected_old_id()
        old_signature = event_signature(
            self.images[selected.frame_index].name,
            selected.rect,
        )
        was_reviewed = old_signature in self.reviewed
        try:
            changes = transform_occurrence_with_points(
                self.documents[selected.frame_index],
                selected,
                new_rect,
            )
            if was_reviewed:
                self.reviewed.discard(old_signature)
                self.reviewed.add(
                    event_signature(
                        self.images[selected.frame_index].name,
                        new_rect,
                    )
                )
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
            self._clear_trajectory_reviews_for_track_ids(selected.track_id)
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.trajectory_reviews = original_trajectory_reviews
            restore_snapshot(snapshot)
            self.motion_edit_preview_rect = selected.rect
            messagebox.showerror(
                "调整检测框失败",
                str(error),
                parent=self.motion_window or self.root,
            )
            self._draw_motion_edit_selection()
            return

        self.undo_stack.append(
            (
                f"第 {selected.frame_index + 1} 张调整 ID "
                f"{selected.track_id} 检测框",
                snapshot,
            )
        )
        track_id = selected.track_id
        frame_index = selected.frame_index
        edit_zoom = self.motion_zoom
        edit_pan = list(self.motion_pan_offset)
        self._refresh_queue_after_geometry_change(
            active_track_id,
            active_first_frame,
            selected_old_id,
            max(0, old_index),
        )
        if frame_index not in self.motion_sequence:
            self._set_motion_sequence_around(frame_index)
        else:
            self.motion_index = self.motion_sequence.index(frame_index)
            self.raw_frame_index = self.motion_source_frames[self.motion_index]
        self.motion_edit_occurrence = find_occurrence(
            self.tracks,
            track_id,
            frame_index,
        )
        self.motion_zoom = edit_zoom
        self.motion_pan_offset = edit_pan
        self.motion_view_initialized = True
        self.motion_focus_frame_index = frame_index
        self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
        self.motion_edit_preview_rect = None
        self.motion_left_render_key = None
        self._render_global()
        self._render_current_local()
        self._render_reference()
        self._render_motion(force_left=True)
        message = (
            f"已调整第 {frame_index + 1} 张 ID {track_id} 检测框"
            f"；同步更新 {max(0, changes - 1)} 个 head/tail"
        )
        self.status_var.set(message)
        self.motion_status_var.set(message)

    def _ensure_edit_backup(self) -> bool:
        if not self.section:
            return False
        if self.backup_path is not None:
            return True
        try:
            self.backup_path = create_section_backup(self.section)
            self.backup_var.set(f"备份：{self.backup_path.name}")
            return True
        except OSError as error:
            messagebox.showerror(
                "备份失败", str(error), parent=self.motion_window or self.root
            )
            return False

    def _clear_trajectory_reviews_for_track_ids(
        self,
        *track_ids: int,
    ) -> None:
        """轨迹被编辑后恢复为未检查，避免保留已经失效的复查结论。"""
        if not self._is_trajectory_mode() or self.section is None:
            return
        affected = {track_id for track_id in track_ids if track_id > 0}
        signatures = {
            event.signature
            for event in self.trajectory_all_events
            if event.track_id in affected
        }
        changed = False
        for signature in signatures:
            if self.trajectory_reviews.pop(signature, None) is not None:
                changed = True
        if changed:
            save_trajectory_reviews(self.section, self.trajectory_reviews)

    def _add_motion_box(self, frame_index: int, rect) -> None:
        current = self.current_event()
        suggested_id = current.track_id if current is not None else None
        if suggested_id is None:
            suggested_id = self._selected_old_id()
        dialog = TrackIdQueryDialog(
            "填写 Track ID",
            "请输入这个检测框目前应使用的 ID（按 G 确认）：",
            initialvalue=suggested_id,
            minvalue=1,
            parent=self.motion_window or self.root,
        )
        track_id = dialog.result
        if track_id is None:
            self.motion_status_var.set("已取消添加检测框")
            return
        if any(
            record.track_id == track_id
            for record in bee_occurrences(
                self.documents[frame_index], frame_index
            )
        ):
            self.motion_status_var.set(
                f"当前第 {frame_index + 1} 张已存在 ID {track_id}，未添加"
            )
            return
        if not self.section or not self._ensure_edit_backup():
            return

        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_trajectory_reviews = dict(self.trajectory_reviews)
        original_deferred_new_ids = set(self.deferred_new_ids)
        old_index = self.event_index
        active_event = self.current_event()
        active_track_id = active_event.track_id if active_event else None
        active_first_frame = (
            active_event.first_frame_index if active_event else None
        )
        selected_old_id = self._selected_old_id()
        edit_zoom = self.motion_zoom
        edit_pan = list(self.motion_pan_offset)
        is_new_track = track_id not in self.tracks
        try:
            add_bee_rectangle(
                self.documents[frame_index],
                track_id,
                rect,
            )
            if is_new_track:
                signature = event_signature(
                    self.images[frame_index].name,
                    rect,
                )
                if self._is_trajectory_mode():
                    self.deferred_new_ids.add(track_id)
                    self.reviewed.discard(signature)
                else:
                    self.reviewed.add(signature)
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
            self._clear_trajectory_reviews_for_track_ids(track_id)
            if self._is_trajectory_mode() and is_new_track:
                save_deferred_new_ids(
                    self.section,
                    self.deferred_new_ids,
                )
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.trajectory_reviews = original_trajectory_reviews
            self.deferred_new_ids = original_deferred_new_ids
            restore_snapshot(snapshot)
            messagebox.showerror(
                "添加失败", str(error), parent=self.motion_window or self.root
            )
            return

        self.undo_stack.append(
            (f"第 {frame_index + 1} 张添加 ID {track_id} 检测框", snapshot)
        )
        if self._is_trajectory_mode():
            self._refresh_queue_after_geometry_change(
                active_track_id,
                active_first_frame,
                selected_old_id,
                max(0, old_index),
            )
            self.motion_index = min(
                max(0, frame_index), max(0, len(self.motion_sequence) - 1)
            )
            self.raw_frame_index = self.motion_source_frames[self.motion_index]
            self.motion_zoom = edit_zoom
            self.motion_pan_offset = edit_pan
            self.motion_view_initialized = True
            self.motion_focus_frame_index = frame_index
            self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
            self.motion_left_render_key = None
            self._render_global()
            self._render_current_local()
            self._render_reference()
            self._render_motion(force_left=True)
        else:
            self._rebuild_queue(preferred_index=max(0, old_index))
        self.status_var.set(
            f"已在第 {frame_index + 1} 张添加 ID {track_id} 检测框"
        )
        self.cancel_motion_draw()
        self.motion_status_var.set(
            f"已添加 ID {track_id}；已自动退出画框模式"
        )

    def _delete_motion_box(self, selected) -> None:
        if not self.section or not self._ensure_edit_backup():
            return
        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_trajectory_reviews = dict(self.trajectory_reviews)
        old_index = self.event_index
        active_event = self.current_event()
        active_track_id = active_event.track_id if active_event else None
        active_first_frame = (
            active_event.first_frame_index if active_event else None
        )
        selected_old_id = self._selected_old_id()
        frame_index = selected.frame_index
        edit_zoom = self.motion_zoom
        edit_pan = list(self.motion_pan_offset)
        try:
            deleted = delete_occurrence_with_points(
                self.documents[selected.frame_index],
                selected,
            )
            self.reviewed.discard(
                event_signature(
                    self.images[selected.frame_index].name,
                    selected.rect,
                )
            )
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
            self._clear_trajectory_reviews_for_track_ids(selected.track_id)
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.trajectory_reviews = original_trajectory_reviews
            restore_snapshot(snapshot)
            messagebox.showerror(
                "删除失败", str(error), parent=self.motion_window or self.root
            )
            return

        self.undo_stack.append(
            (
                f"第 {selected.frame_index + 1} 张删除 ID "
                f"{selected.track_id} 检测框",
                snapshot,
            )
        )
        if self._is_trajectory_mode():
            self._refresh_queue_after_geometry_change(
                active_track_id,
                active_first_frame,
                selected_old_id,
                max(0, old_index),
            )
            if self.current_event() is not None and self.motion_sequence:
                self.motion_index = min(
                    max(0, frame_index), len(self.motion_sequence) - 1
                )
                self.raw_frame_index = self.motion_source_frames[self.motion_index]
                self.motion_zoom = edit_zoom
                self.motion_pan_offset = edit_pan
                self.motion_view_initialized = True
                self.motion_focus_frame_index = frame_index
                self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
                self.motion_edit_occurrence = None
                self.motion_edit_preview_rect = None
                self.motion_left_render_key = None
                self._render_global()
                self._render_current_local()
                self._render_reference()
                self._render_motion(force_left=True)
        else:
            self._rebuild_queue(preferred_index=max(0, old_index))
        self.status_var.set(
            f"已直接删除第 {selected.frame_index + 1} 张的 ID "
            f"{selected.track_id} 检测框及 {max(0, deleted - 1)} 个关联关键点"
        )

    def _change_motion_box_id(self, selected) -> None:
        dialog = TrackIdQueryDialog(
            "手动修改 Track ID",
            f"当前帧 ID 为 {selected.track_id}，请输入修改后的 ID（按 G 确认）：",
            initialvalue=selected.track_id,
            minvalue=1,
            parent=self.motion_window or self.root,
        )
        new_track_id = dialog.result
        if new_track_id is None:
            self.motion_status_var.set("已取消修改 ID")
            return
        if new_track_id == selected.track_id:
            self.motion_status_var.set("输入的 ID 没有变化")
            return

        target_exists = any(
            record.track_id == new_track_id
            and record.shape_index != selected.shape_index
            for record in bee_occurrences(
                self.documents[selected.frame_index],
                selected.frame_index,
            )
        )
        if target_exists:
            confirmed = messagebox.askyesno(
                "确认同帧重复 ID",
                f"第 {selected.frame_index + 1} 张已经存在 ID {new_track_id}。\n\n"
                f"继续后，这一帧会同时存在两个 ID {new_track_id} 检测框。"
                "是否仍要修改？",
                icon="warning",
                parent=self.motion_window or self.root,
            )
            if not confirmed:
                self.motion_status_var.set("已取消修改 ID")
                return

        if not self.section or not self._ensure_edit_backup():
            return
        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_trajectory_reviews = dict(self.trajectory_reviews)
        original_deferred_new_ids = set(self.deferred_new_ids)
        old_index = self.event_index
        active_event = self.current_event()
        active_track_id = active_event.track_id if active_event else None
        active_first_frame = (
            active_event.first_frame_index if active_event else None
        )
        selected_old_id = self._selected_old_id()
        frame_index = selected.frame_index
        edit_zoom = self.motion_zoom
        edit_pan = list(self.motion_pan_offset)
        is_new_track = new_track_id not in self.tracks
        try:
            changes = change_occurrence_track_id(
                self.documents[frame_index],
                selected,
                new_track_id,
            )
            if self._is_trajectory_mode() and is_new_track:
                self.deferred_new_ids.add(new_track_id)
                self.reviewed.discard(
                    event_signature(
                        self.images[frame_index].name,
                        selected.rect,
                    )
                )
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
            self._clear_trajectory_reviews_for_track_ids(
                selected.track_id,
                new_track_id,
            )
            if self._is_trajectory_mode() and is_new_track:
                save_deferred_new_ids(
                    self.section,
                    self.deferred_new_ids,
                )
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.trajectory_reviews = original_trajectory_reviews
            self.deferred_new_ids = original_deferred_new_ids
            restore_snapshot(snapshot)
            messagebox.showerror(
                "修改 ID 失败",
                str(error),
                parent=self.motion_window or self.root,
            )
            return

        self.undo_stack.append(
            (
                f"第 {frame_index + 1} 张修改 ID "
                f"{selected.track_id} → {new_track_id}",
                snapshot,
            )
        )
        self._refresh_queue_after_geometry_change(
            active_track_id,
            active_first_frame,
            selected_old_id,
            max(0, old_index),
        )
        if frame_index not in self.motion_sequence:
            self._set_motion_sequence_around(frame_index)
        else:
            self.motion_index = self.motion_sequence.index(frame_index)
            self.raw_frame_index = self.motion_source_frames[self.motion_index]
        self.motion_zoom = edit_zoom
        self.motion_pan_offset = edit_pan
        self.motion_view_initialized = True
        self.motion_focus_frame_index = frame_index
        self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
        self.motion_edit_occurrence = None
        self.motion_edit_mode = None
        self.motion_edit_preview_rect = None
        self.motion_left_render_key = None
        self._render_global()
        self._render_current_local()
        self._render_reference()
        self._render_motion(force_left=True)
        message = (
            f"已将第 {frame_index + 1} 张的 ID {selected.track_id} "
            f"改为 {new_track_id}；同步修改 "
            f"{max(0, changes - 1)} 个 head/tail"
        )
        self.status_var.set(message)
        self.motion_status_var.set(message)

    def _change_motion_track_id_range(self, selected, direction: str) -> None:
        if not self.section:
            return
        if direction == "forward":
            start_frame_index = selected.frame_index
            end_frame_index = len(self.documents) - 1
            direction_text = "从当前帧起向后"
        elif direction == "backward":
            start_frame_index = 0
            end_frame_index = selected.frame_index
            direction_text = "到当前帧为止向前"
        else:
            raise ValueError(f"未知批量修改方向：{direction}")

        current = self.current_event()
        suggested_id = (
            current.track_id
            if current is not None and current.track_id != selected.track_id
            else selected.track_id
        )
        dialog = TrackIdQueryDialog(
            f"{direction_text}批量修改 Track ID",
            (
                f"将 ID {selected.track_id} {direction_text}的全部记录改为指定 ID。\n"
                "当前帧包含在修改范围内；检测框和关联 head/tail 会一起修改。\n\n"
                "请输入修改后的 ID（按 G 确认）："
            ),
            initialvalue=suggested_id,
            minvalue=1,
            parent=self.motion_window or self.root,
        )
        new_track_id = dialog.result
        if new_track_id is None:
            self.motion_status_var.set("已取消批量修改 ID")
            return
        if new_track_id == selected.track_id:
            self.motion_status_var.set("输入的 ID 没有变化")
            return

        affected_occurrences = [
            record
            for frame_index in range(start_frame_index, end_frame_index + 1)
            for record in bee_occurrences(
                self.documents[frame_index],
                frame_index,
            )
            if record.track_id == selected.track_id
        ]
        affected_frames = sorted(
            {record.frame_index for record in affected_occurrences}
        )
        if not affected_frames:
            self.motion_status_var.set(
                f"{direction_text}没有找到 ID {selected.track_id}"
            )
            return

        collisions = [
            frame_index
            for frame_index in collision_frames(
                self.documents,
                selected.track_id,
                new_track_id,
            )
            if start_frame_index <= frame_index <= end_frame_index
        ]
        if collisions:
            shown_frames = "、".join(
                str(frame_index + 1) for frame_index in collisions[:12]
            )
            if len(collisions) > 12:
                shown_frames += f" 等 {len(collisions)} 帧"
            confirmed = messagebox.askyesno(
                "确认批量修改中的同帧重复 ID",
                (
                    f"目标 ID {new_track_id} 已经出现在受影响范围的"
                    f"第 {shown_frames} 张。\n\n"
                    f"继续后，这些帧会同时存在两个 ID {new_track_id} 检测框。"
                    "是否仍要批量修改？"
                ),
                icon="warning",
                parent=self.motion_window or self.root,
            )
            if not confirmed:
                self.motion_status_var.set("已取消批量修改 ID")
                return

        if not self._ensure_edit_backup():
            return
        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_trajectory_reviews = dict(self.trajectory_reviews)
        original_deferred_new_ids = set(self.deferred_new_ids)
        old_index = self.event_index
        active_event = self.current_event()
        active_track_id = active_event.track_id if active_event else None
        active_first_frame = (
            active_event.first_frame_index if active_event else None
        )
        selected_old_id = self._selected_old_id()
        frame_index = selected.frame_index
        edit_zoom = self.motion_zoom
        edit_pan = list(self.motion_pan_offset)
        is_new_track = new_track_id not in self.tracks
        try:
            changes = change_track_id_in_range(
                self.documents,
                selected.track_id,
                new_track_id,
                start_frame_index,
                end_frame_index,
            )
            if self._is_trajectory_mode() and is_new_track:
                self.deferred_new_ids.add(new_track_id)
                for record in affected_occurrences:
                    self.reviewed.discard(
                        event_signature(
                            self.images[record.frame_index].name,
                            record.rect,
                        )
                    )
            old_id_still_exists = any(
                shape.get("group_id") == selected.track_id
                for document in self.documents
                for shape in document.get("shapes", [])
            )
            if not old_id_still_exists:
                self.deferred_new_ids.discard(selected.track_id)
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
            self._clear_trajectory_reviews_for_track_ids(
                selected.track_id,
                new_track_id,
            )
            save_deferred_new_ids(
                self.section,
                self.deferred_new_ids,
            )
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.trajectory_reviews = original_trajectory_reviews
            self.deferred_new_ids = original_deferred_new_ids
            restore_snapshot(snapshot)
            messagebox.showerror(
                "批量修改 ID 失败",
                str(error),
                parent=self.motion_window or self.root,
            )
            return

        self.undo_stack.append(
            (
                f"{direction_text}批量修改 ID "
                f"{selected.track_id} → {new_track_id}",
                snapshot,
            )
        )
        self._refresh_queue_after_geometry_change(
            active_track_id,
            active_first_frame,
            selected_old_id,
            max(0, old_index),
        )
        if frame_index not in self.motion_sequence:
            self._set_motion_sequence_around(frame_index)
        else:
            self.motion_index = self.motion_sequence.index(frame_index)
            self.raw_frame_index = self.motion_source_frames[self.motion_index]
        self.motion_zoom = edit_zoom
        self.motion_pan_offset = edit_pan
        self.motion_view_initialized = True
        self.motion_focus_frame_index = frame_index
        if self.motion_search_track_id == selected.track_id:
            self.motion_search_track_id = new_track_id
            self.motion_search_id_var.set(str(new_track_id))
        self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
        self.motion_edit_occurrence = None
        self.motion_edit_mode = None
        self.motion_edit_preview_rect = None
        self.motion_left_render_key = None
        self._render_global()
        self._render_current_local()
        self._render_reference()
        self._render_motion(force_left=True)
        message = (
            f"已将 ID {selected.track_id} {direction_text}改为 "
            f"{new_track_id}；影响 {len(affected_frames)} 帧，"
            f"修改 {changes} 个 group_id"
        )
        self.status_var.set(message)
        self.motion_status_var.set(message)

    def _on_motion_right_click(self, event) -> str:
        self.release_motion_search_focus()
        selected = self._motion_record_at_event(event)
        if selected is None:
            self.motion_status_var.set("右键没有点中检测框")
            return "break"
        next_id = max(self.tracks, default=0) + 1
        menu = tk.Menu(self.motion_window or self.root, tearoff=False)
        menu.add_command(
            label=(
                f"删除当前帧的 ID {selected.track_id} 检测框（立即执行）"
            ),
            command=lambda: self._delete_motion_box(selected),
        )
        menu.add_separator()
        menu.add_command(
            label=f"手动修改当前帧 ID {selected.track_id}…",
            command=lambda: self._change_motion_box_id(selected),
        )
        menu.add_separator()
        menu.add_command(
            label=(
                f"从当前帧起向后：将 ID {selected.track_id} "
                "全部改为指定 ID…"
            ),
            command=lambda: self._change_motion_track_id_range(
                selected,
                "forward",
            ),
        )
        menu.add_command(
            label=(
                f"到当前帧为止向前：将 ID {selected.track_id} "
                "全部改为指定 ID…"
            ),
            command=lambda: self._change_motion_track_id_range(
                selected,
                "backward",
            ),
        )
        menu.add_separator()
        menu.add_command(
            label=(
                f"删除此段原 ID：从第 {selected.frame_index + 1} 张起将 "
                f"{selected.track_id} 改为末尾新 ID {next_id}"
            ),
            command=lambda: self._split_track_to_last_id(selected),
        )
        self.motion_context_menu = menu
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _split_track_to_last_id(self, selected) -> None:
        if not self.section:
            return
        expected_id = max(
            (
                shape.get("group_id")
                for document in self.documents
                for shape in document.get("shapes", [])
                if isinstance(shape.get("group_id"), int)
            ),
            default=0,
        ) + 1
        confirmed = messagebox.askyesno(
            "拆分并赋予末尾新 ID",
            (
                f"将 ID {selected.track_id} 从第 "
                f"{selected.frame_index + 1} 张开始的全部记录，改为末尾新 "
                f"ID {expected_id}。\n\n"
                "此前帧中的原 ID 保留；检测框和关联的 head/tail 会一起修改。\n"
                "是否继续？"
            ),
            parent=self.motion_window or self.root,
        )
        if not confirmed:
            return

        if self.backup_path is None:
            try:
                self.backup_path = create_section_backup(self.section)
                self.backup_var.set(f"备份：{self.backup_path.name}")
            except OSError as error:
                messagebox.showerror(
                    "备份失败", str(error), parent=self.motion_window or self.root
                )
                return

        snapshot = snapshot_files(self.section, self.images)
        original_documents = copy.deepcopy(self.documents)
        original_reviewed = set(self.reviewed)
        original_trajectory_reviews = dict(self.trajectory_reviews)
        original_deferred_new_ids = set(self.deferred_new_ids)
        old_index = self.event_index
        active_event = self.current_event()
        active_track_id = active_event.track_id if active_event else None
        active_first_frame = (
            active_event.first_frame_index if active_event else None
        )
        selected_old_id = self._selected_old_id()
        frame_index = selected.frame_index
        edit_zoom = self.motion_zoom
        edit_pan = list(self.motion_pan_offset)
        try:
            new_id, changed = split_track_from_frame(
                self.documents,
                selected.track_id,
                selected.frame_index,
            )
            signature = event_signature(
                self.images[selected.frame_index].name,
                selected.rect,
            )
            if self._is_trajectory_mode():
                self.deferred_new_ids.add(new_id)
                self.reviewed.discard(signature)
            else:
                self.reviewed.add(signature)
            save_documents(self.images, self.documents)
            export_mot(self.section, self.images, self.documents)
            save_reviewed(self.section, self.reviewed)
            self._clear_trajectory_reviews_for_track_ids(
                selected.track_id,
                new_id,
            )
            if self._is_trajectory_mode():
                save_deferred_new_ids(
                    self.section,
                    self.deferred_new_ids,
                )
        except Exception as error:
            self.documents = original_documents
            self.reviewed = original_reviewed
            self.trajectory_reviews = original_trajectory_reviews
            self.deferred_new_ids = original_deferred_new_ids
            restore_snapshot(snapshot)
            messagebox.showerror(
                "拆分失败", str(error), parent=self.motion_window or self.root
            )
            return

        self.undo_stack.append(
            (
                f"拆分 ID {selected.track_id} → {new_id}"
                f"（第 {selected.frame_index + 1} 张起）",
                snapshot,
            )
        )
        self.status_var.set(
            f"已将 ID {selected.track_id} 从第 "
            f"{selected.frame_index + 1} 张起拆为末尾新 ID {new_id}；"
            f"修改 {changed} 个 group_id"
        )
        if self._is_trajectory_mode():
            self._refresh_queue_after_geometry_change(
                active_track_id,
                active_first_frame,
                selected_old_id,
                max(0, old_index),
            )
            if self.current_event() is not None and self.motion_sequence:
                self.motion_index = min(
                    max(0, frame_index), len(self.motion_sequence) - 1
                )
                self.raw_frame_index = self.motion_source_frames[self.motion_index]
                self.motion_zoom = edit_zoom
                self.motion_pan_offset = edit_pan
                self.motion_view_initialized = True
                self.motion_focus_frame_index = frame_index
                self.motion_zoom_var.set(f"缩放 {self.motion_zoom:.2f}×")
                self.motion_edit_occurrence = None
                self.motion_edit_preview_rect = None
                self.motion_left_render_key = None
                self._render_global()
                self._render_current_local()
                self._render_reference()
                self._render_motion(force_left=True)
        else:
            self._rebuild_queue(preferred_index=max(0, old_index))

    def _draw_trajectory_path(
        self,
        draw: ImageDraw.ImageDraw,
        crop,
        event: ReviewEvent,
        frame_index: int,
    ) -> None:
        occurrences = sorted(
            self.tracks.get(event.track_id, []),
            key=lambda item: (item.frame_index, item.shape_index),
        )
        metrics = self.trajectory_metrics.get(event.track_id)
        anomaly_frames = set(metrics.anomaly_frames) if metrics else set()
        small_font = self._font(11)
        for left, right in zip(occurrences, occurrences[1:]):
            left_center = rect_center(left.rect)
            right_center = rect_center(right.rect)
            color = (
                (255, 70, 70)
                if right.frame_index in anomaly_frames
                or right.frame_index - left.frame_index > 1
                else (
                    (0, 210, 255)
                    if right.frame_index <= frame_index
                    else (255, 175, 0)
                )
            )
            draw.line(
                (
                    left_center[0] - crop[0],
                    left_center[1] - crop[1],
                    right_center[0] - crop[0],
                    right_center[1] - crop[1],
                ),
                fill=color,
                width=3,
            )
        for occurrence in occurrences:
            center_x, center_y = rect_center(occurrence.rect)
            x = center_x - crop[0]
            y = center_y - crop[1]
            is_current = occurrence.frame_index == frame_index
            is_anomaly = occurrence.frame_index in anomaly_frames
            radius = 5 if is_current else 3
            color = (
                (255, 60, 60)
                if is_anomaly
                else ((255, 230, 0) if is_current else (0, 220, 255))
            )
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(0, 0, 0),
                width=1,
            )
            draw.text(
                (x + 4, y + 3),
                str(occurrence.frame_index + 1),
                fill=color,
                font=small_font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

    def _draw_motion_overlay(
        self,
        image: Image.Image,
        frame_index: int,
        crop,
        event: ReviewEvent,
    ) -> None:
        draw = ImageDraw.Draw(image)
        old_id = self._selected_old_id()
        old_color = (0, 225, 255)
        new_color = (255, 60, 60)
        search_color = (255, 235, 0)
        context_color = (0, 255, 100)
        font = self._font(16)
        records = (
            bee_occurrences(self.documents[frame_index], frame_index)
            if self.motion_show_boxes.get()
            else []
        )
        for record in records:
            if (
                self.motion_edit_occurrence is not None
                and self.motion_edit_occurrence.frame_index == frame_index
                and self.motion_edit_occurrence.shape_index == record.shape_index
            ):
                continue
            if not intersects(record.rect, crop):
                continue
            rect = (
                record.rect[0] - crop[0],
                record.rect[1] - crop[1],
                record.rect[2] - crop[0],
                record.rect[3] - crop[1],
            )
            if record.track_id == self.motion_search_track_id:
                color, width = search_color, 5
            elif record.track_id == old_id:
                color, width = old_color, 4
            elif record.track_id == event.track_id:
                color, width = new_color, 4
            else:
                color, width = context_color, 1
            draw.rectangle(rect, outline=color, width=width)
            is_focus = record.track_id in {
                old_id,
                event.track_id,
                self.motion_search_track_id,
            }
            show_id = is_focus or self.motion_show_all_ids.get()
            if show_id or self.show_class_names.get():
                parts = []
                if self.show_class_names.get():
                    parts.append("bee")
                if show_id:
                    parts.append(str(record.track_id))
                label = " ".join(parts)
                text_box = draw.textbbox(
                    (rect[0], rect[1]), label, font=font, stroke_width=2
                )
                draw.rectangle(text_box, fill=(0, 0, 0))
                draw.text(
                    (rect[0], rect[1]),
                    label,
                    fill=color,
                    font=font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

        if self._is_trajectory_mode():
            if self.motion_show_trajectory.get():
                self._draw_trajectory_path(draw, crop, event, frame_index)
            metrics = self.trajectory_metrics.get(event.track_id)
            status = self.trajectory_reviews.get(event.signature)
            status_text = {
                "passed": "已通过",
                "issue": "有问题",
            }.get(status, "未检查")
            warning_text = (
                f"  疑似：{'；'.join(metrics.reasons)}"
                if metrics and metrics.reasons
                else ""
            )
            title = (
                f"轨迹复查  ID {event.track_id}  第 {frame_index + 1} 张  "
                f"{status_text}{warning_text}"
            )
            title_box = draw.textbbox((8, 8), title, font=font, stroke_width=2)
            draw.rectangle(title_box, fill=(0, 0, 0))
            draw.text(
                (8, 8),
                title,
                fill=(255, 255, 255),
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            return

        def path_points(track_id: int | None):
            if track_id is None or not self.motion_show_trajectory.get():
                return []
            return [
                (
                    rect_center(item.rect)[0] - crop[0],
                    rect_center(item.rect)[1] - crop[1],
                )
                for item in self.tracks.get(track_id, [])
                if (
                    self.motion_sequence[0]
                    <= item.frame_index
                    <= frame_index
                    and intersects(item.rect, crop)
                )
            ]

        old_points = path_points(old_id)
        new_points = path_points(event.track_id)
        if len(old_points) >= 2:
            draw.line(old_points, fill=old_color, width=3)
        if len(new_points) >= 2:
            draw.line(new_points, fill=new_color, width=3)

        old_last = (
            last_occurrence_before(self.tracks, old_id, event.first_frame_index)
            if old_id is not None
            else None
        )
        new_first = find_occurrence(
            self.tracks, event.track_id, event.first_frame_index
        )
        if (
            self.motion_show_trajectory.get()
            and frame_index >= event.first_frame_index
            and old_last
            and new_first
            and intersects(old_last.rect, crop)
            and intersects(new_first.rect, crop)
        ):
            old_center = rect_center(old_last.rect)
            new_center = rect_center(new_first.rect)
            draw.line(
                (
                    old_center[0] - crop[0],
                    old_center[1] - crop[1],
                    new_center[0] - crop[0],
                    new_center[1] - crop[1],
                ),
                fill=(255, 220, 0),
                width=3,
            )

        phase = "新 ID 段" if frame_index >= event.first_frame_index else "旧 ID 段"
        title = (
            f"{phase}  第 {frame_index + 1} 张  "
            f"旧 {old_id if old_id is not None else '-'} → 新 {event.track_id}"
        )
        title_box = draw.textbbox((8, 8), title, font=font, stroke_width=2)
        draw.rectangle(title_box, fill=(0, 0, 0))
        draw.text(
            (8, 8),
            title,
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    def _render_motion(self, force_left: bool = False) -> None:
        if (
            self.motion_canvas is None
            or not self.motion_canvas.winfo_exists()
            or self.raw_canvas is None
            or not self.raw_canvas.winfo_exists()
        ):
            return
        canvas = self.motion_canvas
        raw_canvas = self.raw_canvas
        event = self.current_event()
        if not event or not self.motion_sequence:
            self.motion_frame_var.set("")
            if self.motion_raw_frame_group is not None:
                self.motion_raw_frame_group.configure(
                    text="原视频同步放大画面"
                )
            canvas.delete("all")
            raw_canvas.delete("all")
            self._render_trajectory_timeline()
            self.motion_transform = None
            self.raw_motion_transform = None
            canvas.create_text(
                max(1, canvas.winfo_width()) / 2,
                max(1, canvas.winfo_height()) / 2,
                text="没有可播放的局部运动",
                fill="white",
                font=("Microsoft YaHei UI", 13),
            )
            return
        self.motion_index = min(self.motion_index, len(self.motion_sequence) - 1)
        frame_index = self.motion_sequence[self.motion_index]
        image = self._get_image(frame_index)
        should_refocus = not self.motion_view_initialized
        if (
            self.motion_follow.get()
            and self.motion_focus_frame_index != frame_index
        ):
            should_refocus = True
        if should_refocus:
            self._initialize_motion_view(event, frame_index, image.size)
        left_key = (
            frame_index,
            canvas.winfo_width(),
            canvas.winfo_height(),
            self.work_mode_var.get(),
            self._selected_old_id(),
            self.motion_follow.get(),
            self.motion_show_boxes.get(),
            self.show_class_names.get(),
            self.motion_show_all_ids.get(),
            self.motion_show_trajectory.get(),
            self.motion_search_track_id,
            round(self.motion_zoom, 4),
            round(self.motion_pan_offset[0], 2),
            round(self.motion_pan_offset[1], 2),
        )
        if force_left or left_key != self.motion_left_render_key:
            canvas.delete("motion_edit")
            full_crop = (0, 0, image.width, image.height)
            self._draw_motion_overlay(image, frame_index, full_crop, event)
            resized, scale, x, y = self._fit_motion_to_canvas(
                canvas,
                image,
                resample=(
                    Image.Resampling.BILINEAR
                    if self.motion_zoom > 1.0
                    else Image.Resampling.LANCZOS
                ),
            )
            motion_image_items = canvas.find_withtag("motion_frame_image")
            can_reuse = (
                self.motion_photo is not None
                and motion_image_items
                and self.motion_photo.width() == resized.width
                and self.motion_photo.height() == resized.height
            )
            if can_reuse:
                self.motion_photo.paste(resized)
                canvas.coords(motion_image_items[0], x, y)
            else:
                canvas.delete("all")
                self.motion_photo = ImageTk.PhotoImage(resized)
                canvas.create_image(
                    x,
                    y,
                    image=self.motion_photo,
                    anchor=tk.NW,
                    tags=("motion_pan_content", "motion_frame_image"),
                )
            self.motion_transform = (scale, x, y, full_crop, frame_index)
            self._draw_motion_edit_selection()
            self.motion_left_render_key = left_key

        raw_canvas.delete("raw_marker")
        raw_canvas.delete("raw_message")
        self.raw_motion_transform = None
        raw_image = self._get_raw_image(self.raw_frame_index)
        raw_total_text = (
            str(self.raw_video_frame_count)
            if self.raw_video_frame_count > 0
            else "未知"
        )
        if self.motion_raw_frame_group is not None:
            self.motion_raw_frame_group.configure(
                text=(
                    "原视频同步放大画面｜"
                    f"当前帧 {self.raw_frame_index}｜总帧数 {raw_total_text}"
                )
            )
        if raw_image is None:
            raw_canvas.delete("all")
            self.raw_photo = None
            expected = self._resolve_raw_video_path()
            hint = (
                str(expected)
                if expected is not None
                else f"{self.raw_video_root}\n（未找到当前区段对应的 MP4）"
            )
            raw_canvas.create_text(
                max(1, raw_canvas.winfo_width()) / 2,
                max(1, raw_canvas.winfo_height()) / 2,
                text=f"无法读取原视频：\n{hint}\n\n可点击上方“原视频目录”重新选择",
                fill="white",
                justify=tk.CENTER,
                font=("Microsoft YaHei UI", 11),
                tags=("raw_message",),
            )
        else:
            full_crop = (0, 0, raw_image.width, raw_image.height)
            draw = ImageDraw.Draw(raw_image)
            font = self._font(16)
            source_frame = self.motion_source_frames[self.motion_index]
            sampled_width, sampled_height = image.size
            raw_markers = []
            marker_specs = (
                (self._selected_old_id(), (0, 180, 255)),
                (event.track_id, (255, 30, 30)),
            )
            for marker_track_id, marker_color in marker_specs:
                if marker_track_id is None:
                    continue
                marker_occurrence = find_occurrence(
                    self.tracks,
                    marker_track_id,
                    frame_index,
                )
                if marker_occurrence is None:
                    continue
                center_x, center_y = rect_center(marker_occurrence.rect)
                marker_x = center_x * raw_image.width / sampled_width
                marker_y = center_y * raw_image.height / sampled_height
                raw_markers.append((marker_x, marker_y, marker_color))
            title = (
                f"原视频帧 {self.raw_frame_index}/{raw_total_text}  "
                f"对应抽帧 {source_frame}"
            )
            title_box = draw.textbbox((8, 8), title, font=font, stroke_width=2)
            draw.rectangle(title_box, fill=(0, 0, 0))
            draw.text(
                (8, 8),
                title,
                fill=(255, 255, 255),
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            resized, scale, x, y = self._fit_motion_to_canvas(
                raw_canvas,
                raw_image,
                resample=Image.Resampling.BILINEAR,
            )
            # 每个原帧都替换 Tk 图片对象，避免帧号已经变化但画布仍
            # 短暂保留上一帧 PhotoImage 缓存。
            raw_canvas.delete("raw_frame_image")
            self.raw_photo = ImageTk.PhotoImage(resized)
            raw_canvas.create_image(
                x,
                y,
                image=self.raw_photo,
                anchor=tk.NW,
                tags=("raw_pan_content", "raw_frame_image"),
            )
            marker_radius = self._motion_marker_radius()
            marker_outline_width = 1 if marker_radius <= 3 else 2
            for marker_x, marker_y, marker_color in raw_markers:
                canvas_x = x + marker_x * scale
                canvas_y = y + marker_y * scale
                color = "#%02x%02x%02x" % marker_color
                raw_canvas.create_oval(
                    canvas_x - marker_radius,
                    canvas_y - marker_radius,
                    canvas_x + marker_radius,
                    canvas_y + marker_radius,
                    fill=color,
                    outline="#ffffff",
                    width=marker_outline_width,
                    tags=("raw_pan_content", "raw_marker"),
                )
            self.raw_motion_transform = (
                scale,
                x,
                y,
                full_crop,
                self.raw_frame_index,
            )
        self.motion_frame_var.set(
            f"抽帧 {frame_index + 1}/{len(self.images)}  |  "
            f"原帧 {self.raw_frame_index}/{raw_total_text}"
        )
        self._render_trajectory_timeline()

    def _reference_spec(self, event: ReviewEvent):
        frame_index = min(
            max(0, self.reference_frame_index),
            max(0, len(self.images) - 1),
        )
        selected_old = self._selected_old_id()
        occurrence = (
            find_occurrence(self.tracks, selected_old, frame_index)
            if selected_old is not None
            else None
        )
        focus_rect = occurrence.rect if occurrence is not None else event.rect
        return frame_index, focus_rect, selected_old

    def _render_reference(self) -> None:
        canvas = self.reference_canvas
        canvas.delete("all")
        event = self.current_event()
        if not self.images or not event:
            return
        frame_index, focus_rect, selected_old = self._reference_spec(event)
        image = self._get_image(frame_index)
        crop = self._crop_around(image.size, focus_rect)
        cropped = image.crop(crop)
        self._draw_boxes(
            cropped,
            frame_index,
            selected_old,
            offset=(crop[0], crop[1]),
            crop_rect=crop,
            scale_hint=1.4,
        )
        resized, scale, x, y = self._fit_to_canvas(canvas, cropped)
        self.reference_photo = ImageTk.PhotoImage(resized)
        canvas.create_image(x, y, image=self.reference_photo, anchor=tk.NW)
        self.reference_transform = (scale, x, y, crop)
        canvas.create_text(
            10,
            10,
            text=f"参考第 {frame_index + 1} 张",
            fill="white",
            anchor=tk.NW,
            font=("Microsoft YaHei UI", 10, "bold"),
        )

    def _refresh(self) -> None:
        event = self.current_event()
        if event and self.images:
            self.frame_var.set(
                f"当前第 {self.display_frame_index + 1}/{len(self.images)} 张"
                f"    审核新 ID：{event.track_id}"
            )
        else:
            self.frame_var.set("")
        self._render_global()
        self._render_current_local()
        self._render_reference()
        self._render_motion()

    def _schedule_redraw(self, _event=None) -> None:
        if self.redraw_job is not None:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(80, self._run_redraw)

    def _run_redraw(self) -> None:
        self.redraw_job = None
        self._refresh()

    def _canvas_to_image(self, event, transform):
        if transform is None:
            return None
        scale, x, y = transform[:3]
        if scale <= 0:
            return None
        return (event.x - x) / scale, (event.y - y) / scale

    def _on_global_click(self, event) -> None:
        point = self._canvas_to_image(event, self.global_transform)
        current = self.current_event()
        if point is None or current is None:
            return
        records = [
            record
            for record in bee_occurrences(
                self.documents[self.display_frame_index], self.display_frame_index
            )
            if point_in_rect(point, record.rect)
        ]
        if not records:
            return
        selected = min(records, key=lambda record: rect_area(record.rect))
        if selected.track_id != current.track_id:
            self._stop_global_video()
            self.old_id_var.set(str(selected.track_id))
            self._reset_motion()
            self._refresh()

    def _on_motion_click(self, event) -> None:
        current = self.current_event()
        selected = self._motion_record_at_event(event)
        if selected is None or current is None:
            return
        if selected.track_id == current.track_id:
            self.motion_status_var.set(
                f"ID {selected.track_id} 是当前新 ID，不能选为旧 ID"
            )
            return
        self.old_id_var.set(str(selected.track_id))
        self._reset_motion()
        self._refresh()

    def _on_reference_click(self, event) -> None:
        if self.reference_transform is None:
            return
        point = self._canvas_to_image(event, self.reference_transform)
        if point is None:
            return
        crop = self.reference_transform[3]
        original_point = (point[0] + crop[0], point[1] + crop[1])
        records = [
            record
            for record in bee_occurrences(
                self.documents[self.reference_frame_index], self.reference_frame_index
            )
            if point_in_rect(original_point, record.rect)
        ]
        if not records:
            return
        selected = min(records, key=lambda record: rect_area(record.rect))
        self.old_id_var.set(str(selected.track_id))
        self._reset_motion()
        self._refresh()

    def open_help(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Track ID 修正器帮助")
        window.geometry("720x620")
        text = tk.Text(
            window,
            wrap=tk.WORD,
            font=("Microsoft YaHei UI", 10),
            padx=14,
            pady=12,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(
            tk.END,
            """使用流程

1. 打开标注员任务目录或单个区段。
2. 左侧默认显示当前新 ID 第一次出现的局部画面。
3. 右侧参考画面可从当前区段中选择任意一帧。
4. 可用参考帧下拉框、左右按钮或 [ / ] 快捷键切换参考帧。
5. 点击右侧参考画面中的旧框，或在“对应旧 ID”中输入编号。
6. 按 P 打开独立的“局部运动对比”窗口并播放。
7. 确认是同一只蜜蜂后，按 C 合并。
8. 如果确实是新蜜蜂，按 V 保留。
9. 按 G 可单独打开当前帧的全局鸟瞰图。

室外轨迹复查

• 在顶部“模式”中选择“室外轨迹复查”，软件只列出名称以 A- 开头的室外区段。
• 每个 Track ID 都会进入复查队列，包括从第一帧就已出现的 ID。
• 软件默认只显示“未检查”轨迹；重新打开任务时会自动进入当前区段第一个未检查 ID。
• 每次按 Space 标记通过或按 M 标记有问题后都会立即保存审核进度。
• 局部运动窗口中的“选择检查 ID”可直接选择任意 ID；下拉项会显示未检查、通过或有问题状态。
• 按 P 打开局部运动窗口；左侧显示完整轨迹，青色为已走过部分、橙色为后续部分、红色为疑似异常段。
• 下方时间轴可点击任意帧；绿色表示该 ID 存在，灰色表示缺失，红边表示规则检测到疑似异常。
• 可按“全部、未检查、疑似异常、有问题”筛选轨迹。
• 确认轨迹正确后按 Space 标记通过；需要后续处理时按 M 标记有问题。
• J / K 切换上一条 / 下一条轨迹，走到区段末尾后会自动进入下一个室外区段。
• 复查结果保存在各区段的 .trackid_trajectory_review.json，不修改原标注格式。
• 如果移动、缩放、添加、删除或改 ID，受影响轨迹会自动恢复为“未检查”。
• 轨迹复查模式仍保留 Ctrl 编辑框、右键改 ID/删除/拆分、R 添加框和 Ctrl+Z 撤销。
• 复查时新建或手动改出的末尾 ID 会进入延后队列；所有 A 类轨迹复查完成后，软件会自动切换到“新 ID 纠正”，并按 A 类区段继续审核这些 ID。

全局鸟瞰

• 左侧显示带检测框和 Track ID 的抽帧全局画面。
• 右侧同步显示同一时刻、不带框的原视频全局画面。
• 点击左侧任意旧 ID 检测框，可将其选为“对应旧 ID”。
• Q / E 每次切换一张抽帧；“原帧”按钮每次切换一个原视频帧。
• 播放时右侧逐原帧前进，左侧每经过 5 个原帧同步更新。

局部运动

• 青色框和轨迹代表候选旧 ID，红色代表当前新 ID。
• 在“查找 ID”中输入编号并回车或点击“定位”，可跳到该 ID 距当前最近的一次出现。
• 定位成功后会自动退出输入框；按 Esc 或点击左右画面也能释放输入框和输入法。
• 搜索结果会自动居中放大并使用亮绿色高亮；Ctrl+F 可快速聚焦搜索框。
• 黄色线连接旧 ID 最后位置和新 ID 首次位置。
• 点击左侧带框画面中的任意旧 ID 框，可直接切换候选旧 ID。
• 按住 Ctrl 用左键点击框可选中；拖动框内部可移动，拖动 8 个控制点可缩放。
• 调整框时只显示预览，松开左键后自动保存；关联 head/tail 会同比例移动缩放。
• 不按 Ctrl 时保持原有选择、画框和画面拖拽逻辑；调整操作可用 Ctrl+Z 撤销。
• 右键点击左侧任意 ID 框，可手动输入并修改当前帧 ID，也可把该轨迹从当前帧起拆出并赋予末尾新 ID。
• 按 R 或点击“画框”进入画框模式；左键拖框，松开后填写当前 ID。
• ID 输入框默认填写当前正在审核的新 ID，可以直接修改。
• ID 输入完成后直接按 G 确认，也可以按 Enter 或点击 OK。
• 每成功添加一个框后会自动退出画框模式，鼠标恢复选择/拖拽状态。
• 画框模式中按住鼠标中键拖动，可平移左右画面；左键继续用于画框。
• 右键框后选择“删除当前帧”，会立即删除框及同 ID 的 head/tail，不再二次确认。
• 拆分会同步修改检测框及关联 head/tail，可用 Ctrl+Z 撤销。
• 在左侧或右侧按住鼠标左键拖拽，可同步平移两边的观察位置。
• 左右两侧始终加载完整原图；打开时只是自动放大并定位到当前目标。
• 在左侧或右侧滚动鼠标滚轮，可同步缩放两边画面（1.0×～6.0×）。
• 缩小到 1.0× 时显示完整原图；放大后拖拽可查看整张图的任意位置。
• 拖拽时会自动暂停播放；“视野回到目标”可恢复初始局部放大和定位。
• 打开窗口时默认定位到新 ID 第一次出现；按钮可随时返回该时刻。
• 局部窗口中 A / S 控制上一/下一抽帧，D / F 控制上一/下一原帧。
• H 隐藏或恢复左侧全部检测框和 ID；轨迹线仍由 B 单独控制。
• Z 切换显示全部框的 ID；再次按 Z 只保留当前新旧 ID 标签。
• B 切换轨迹线、轨迹点和轨迹帧序号；检测框不会被隐藏。
• X 回到新 ID 第一次出现的位置。
• 默认采用固定视野，错误的位置跳变会更明显。
• “跟随目标”适合目标移动距离较大时放大观察。
• 打开时定位到新 ID 首次出现，但 A / S 和播放可覆盖当前区段的全部抽帧。
• 左侧播放带框的抽帧图，右侧同步播放同一位置、同一时段的原视频。
• 右侧画面标题和窗口右上角会显示原视频当前帧 / 总帧数。
• “点大小”可将右侧红色、蓝色中心点半径调整为 1～20 px，并自动记忆。
• “抽帧”按钮每次跳 5 个原视频帧；“原帧”按钮每次只跳 1 帧。
• 播放速度表示左侧抽帧速度，右侧会按原始帧连续播放。
• 便携版默认从程序旁的“原视频”目录查找，也可点击“原视频目录”更换。
• “显示类别名”开启后，框标题会显示为 bee + ID。

合并规则

• 新 ID 的全部框和关联 head/tail 会改成旧 ID。
• 旧 ID 可以大于新 ID；删除新 ID 后，较大的旧 ID 会随编号压缩而减 1。
• 所有大于新 ID 的编号自动减 1，保持编号连续。
• 如果新旧 ID 在同一帧同时存在，软件会先弹出确认说明。
• 确认后，仅把冲突帧中原旧 ID 的框及其 head/tail 改为当前末尾 ID，并加入待修正队列；再将当前新 ID 合并为旧 ID。
• 首次修改前会完整备份当前区段。
• 每次合并都会重新生成 MOT/gt.txt。

快捷键

A / D        上一个 / 下一个待审核新 ID
Q / E        上一帧 / 下一帧
[ / ]        上一参考帧 / 下一参考帧
C            合并并重编号
V            确认为真实新目标
Ctrl+Z       撤销上一步
Ctrl+O       打开目录
F1           帮助
P            打开独立运动窗口 / 播放 / 暂停
G            打开全局鸟瞰图
J / K        上一条 / 下一条室外复查轨迹
Space        室外轨迹标记为通过
M            室外轨迹标记为有问题

局部运动窗口专用

A / S        上一抽帧 / 下一抽帧
D / F        上一原帧 / 下一原帧
Z            显示全部 ID / 隐藏其他 ID
B            显示 / 隐藏轨迹线、轨迹点和帧序号
X            回到新 ID 第一次出现
R            开启 / 退出画框模式
Esc          退出画框模式
""",
        )
        text.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._stop_motion()
        self._stop_global_video()
        self._release_raw_video()
        self._release_global_raw_video()
        self._save_settings()
        self.root.destroy()


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass
    root = tk.Tk()
    TrackIdCorrector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
