"""Synthetic image generation helpers shared by scroll detection tests."""

import numpy as np


def create_text_pattern_image(
    height: int, width: int, pattern_type: str = "text"
) -> np.ndarray:
    """
    Create a synthetic image with text-like patterns.

    Args:
        height: Image height
        width: Image width
        pattern_type: Type of pattern ('lines', 'grid', 'text')

    Returns:
        Numpy array (H, W, 3) with RGB values
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    if pattern_type == "lines":
        # Horizontal lines simulating text
        for y in range(50, height - 50, 40):
            thickness = 20
            img[y : y + thickness, 100 : width - 100] = [0, 0, 0]

    elif pattern_type == "grid":
        # Grid pattern
        for y in range(0, height, 50):
            img[y : y + 2, :] = [100, 100, 100]
        for x in range(0, width, 50):
            img[:, x : x + 2] = [100, 100, 100]

    elif pattern_type == "text":
        # Non-repeating text-like pattern with position-dependent unique content
        y = 50
        line_num = 0
        while y < height - 50:
            thickness = 15 + ((line_num * 7) % 8)
            left_margin = 100 + ((line_num * 11) % 50)
            right_offset = (line_num * 13) % 30
            base_intensity = ((line_num * 23) % 180) + 20

            img[y : y + thickness, left_margin : width - 100 - right_offset] = [
                base_intensity,
                base_intensity,
                base_intensity,
            ]

            marker_x = 50 + ((line_num * 31) % 30)
            marker_intensity = ((line_num * 41) % 150) + 50
            img[y : y + thickness, marker_x : marker_x + 5] = [
                marker_intensity,
                marker_intensity,
                marker_intensity,
            ]

            y += 35 + ((line_num * 19) % 15)
            line_num += 1

    return img


def create_scrolled_image(
    original_img: np.ndarray, scroll_distance: int, direction: str = "down"
) -> np.ndarray:
    """
    Create a scrolled version of an image.

    Args:
        original_img: Original image array
        scroll_distance: Pixels to scroll (positive value)
        direction: "down" (content moves up) or "up" (content moves down)

    Returns:
        Scrolled image with same dimensions
    """
    height = original_img.shape[0]
    scrolled = np.ones_like(original_img) * 255  # White background

    remaining_height = height - scroll_distance
    if remaining_height > 0:
        if direction == "down":
            # Scroll down: content moves up, new content appears at bottom
            scrolled[:remaining_height] = original_img[scroll_distance:]
        else:
            # Scroll up: content moves down, new content appears at top
            scrolled[scroll_distance:] = original_img[:remaining_height]

    return scrolled
