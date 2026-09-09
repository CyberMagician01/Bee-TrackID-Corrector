from app import TrackIdCorrector


def test_trajectory_sections_include_indoor_and_outdoor(tmp_path):
    corrector = object.__new__(TrackIdCorrector)
    corrector.all_section_dirs = [
        tmp_path / "A-5-1_区段_01",
        tmp_path / "B-5-1_区段_01",
        tmp_path / "说明文件",
    ]

    assert [path.name for path in corrector._trajectory_sections()] == [
        "A-5-1_区段_01",
        "B-5-1_区段_01",
    ]
