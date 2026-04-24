"""Tests for OCR cache behavior."""

import logging
from typing import cast

import screen_ocr
from screen_ocr import _base

from gaze_ocr._gaze_ocr import BoundingBox, Controller, EyeTrackerFallback, OcrCache


def _contents(
    bounding_box: tuple[int, int, int, int], words: list[_base.OcrWord] | None = None
) -> screen_ocr.ScreenContents:
    return screen_ocr.ScreenContents(
        screen_coordinates=None,
        bounding_box=bounding_box,
        screenshot=None,
        result=_base.OcrResult(lines=[_base.OcrLine(words or [])]),
        confidence_threshold=1,
        homophones={},
        search_radius=None,
    )


class FakeReader:
    SCREEN = (0, 0, 100, 100)

    def __init__(self):
        self.read_screen_calls: list[tuple[int, int, int, int] | None] = []
        self.read_current_window_calls = 0

    def read_screen(self, bounding_box: tuple[int, int, int, int] | None = None):
        self.read_screen_calls.append(bounding_box)
        return _contents(bounding_box or self.SCREEN)

    def read_current_window(self):
        self.read_current_window_calls += 1
        return _contents((20, 30, 120, 150))


def _box(left: int, top: int, right: int, bottom: int) -> BoundingBox:
    return BoundingBox(left=left, right=right, top=top, bottom=bottom)


def _cache(reader: FakeReader) -> OcrCache:
    return OcrCache(cast(screen_ocr.Reader, reader))


def test_subset_bounding_box_reuses_cache():
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(_box(0, 0, 100, 100))
    second = cache.read(_box(10, 10, 20, 20))

    assert second.bounding_box == (10, 10, 20, 20)
    assert reader.read_screen_calls == [(0, 0, 100, 100)]


def test_non_subset_bounding_box_misses_cache():
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(_box(0, 0, 20, 20))
    cache.read(_box(30, 30, 40, 40))

    assert reader.read_screen_calls == [(0, 0, 20, 20), (30, 30, 40, 40)]


def test_active_window_fallback():
    reader = FakeReader()
    cache = OcrCache(
        cast(screen_ocr.Reader, reader),
        fallback_when_no_eye_tracker=EyeTrackerFallback.ACTIVE_WINDOW,
    )

    cache.read(None)

    assert reader.read_current_window_calls == 1
    assert reader.read_screen_calls == []


def test_empty_cache_miss_does_not_warn(caplog):
    reader = FakeReader()
    cache = _cache(reader)

    with caplog.at_level(logging.WARNING):
        cache.read(None)

    assert not caplog.records


def test_cache_hit_does_not_warn(caplog):
    reader = FakeReader()
    cache = _cache(reader)
    cache.read(_box(0, 0, 100, 100))

    with caplog.at_level(logging.WARNING):
        cache.read(_box(10, 10, 20, 20))

    assert not caplog.records


def test_populated_cache_miss_warns(caplog):
    reader = FakeReader()
    cache = _cache(reader)
    cache.read(_box(0, 0, 20, 20))

    with caplog.at_level(logging.WARNING):
        cache.read(_box(30, 30, 40, 40))

    assert len(caplog.records) == 1
    assert "OCR cache miss with populated cache" in caplog.records[0].message


def test_controller_start_reading_is_noop_and_reads_reuse_ocr_cache():
    reader = FakeReader()
    controller = Controller(
        ocr_reader=cast(screen_ocr.Reader, reader),
        eye_tracker=None,
        mouse=None,
        keyboard=None,
    )
    try:
        controller.start_reading_nearby()
        assert reader.read_screen_calls == []

        first = controller.read_nearby(gaze_bounds=_box(0, 0, 100, 100))
        second = controller.read_nearby(gaze_bounds=_box(10, 10, 20, 20))

        assert second.bounding_box == (-90, -90, 120, 120)
        assert controller.latest_screen_contents() is second
        assert first is not second
        assert reader.read_screen_calls == [(-100, -100, 200, 200)]
    finally:
        controller.shutdown()
