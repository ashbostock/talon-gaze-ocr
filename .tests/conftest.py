"""Pytest configuration and shared fixtures for scroll detection tests."""

import os
import sys
from pathlib import Path

import pytest
from synthetic_images import create_text_pattern_image

# Add parent directory to path to import scroll_detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to test data directory
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_text_image():
    """Create a sample text pattern image for testing."""
    return create_text_pattern_image(1000, 1280, "text")


@pytest.fixture
def sample_grid_image():
    """Create a sample grid pattern image for testing."""
    return create_text_pattern_image(720, 1280, "grid")


@pytest.fixture
def load_real_image_pair():
    """Factory fixture to load real image pairs from data directory."""
    import numpy as np
    from PIL import Image

    def _load(before_name: str, after_name: str):
        before_path = DATA_DIR / before_name
        after_path = DATA_DIR / after_name

        if not before_path.exists() or not after_path.exists():
            pytest.skip(f"Image files not found: {before_name}, {after_name}")

        img_before = np.array(Image.open(before_path).convert("RGB"))
        img_after = np.array(Image.open(after_path).convert("RGB"))

        return img_before, img_after

    return _load


@pytest.fixture
def load_json_test_case():
    """Factory fixture to load test cases from JSON files."""
    import json

    import numpy as np
    from PIL import Image

    def _load(json_name: str):
        json_path = DATA_DIR / json_name

        if not json_path.exists():
            pytest.skip(f"JSON file not found: {json_name}")

        with open(json_path) as f:
            data = json.load(f)

        base = str(json_path).rsplit(".json", 1)[0]
        before_path = base + "_before.png"
        after_path = base + "_after.png"

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            pytest.skip(f"Image files not found for {json_name}")

        img_before = np.array(Image.open(before_path).convert("RGB"))
        img_after = np.array(Image.open(after_path).convert("RGB"))

        cursor_pos = None
        cp = data.get("cursor_position", {})
        if "x" in cp and "y" in cp:
            cursor_pos = (cp["x"], cp["y"])

        return img_before, img_after, cursor_pos, data

    return _load
