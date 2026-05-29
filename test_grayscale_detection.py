#!/usr/bin/env python3
"""测试灰度检测器功能。"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyvips
from image_compressor.grayscale_detector import (
    detect_grayscale,
    _analyze_hue_concentration,
    get_detection_config,
)


def create_test_image(width=100, height=100, color_type="grayscale"):
    """创建测试图像。"""
    if color_type == "grayscale":
        # 纯灰度: 所有通道都是 128
        img = pyvips.Image.black(width, height, bands=3)
        img = img.draw_rect1(128, 128, 128)
        img = img.colourspace("srgb")
    elif color_type == "tinted_green":
        # 泛绿黑白漫: R=100, G=150, B=80
        img = pyvips.Image.black(width, height, bands=3)
        img = img.draw_rect1(100, 150, 80)
        img = img.colourspace("srgb")
    elif color_type == "colorful":
        # 真正彩漫 (多种颜色)
        img = pyvips.Image.black(width, height, bands=3)
        # 左半部分: 肤色 (R=255, G=200, B=150)
        left = pyvips.Image.black(width//2, height, bands=3)
        left = left.draw_rect1(255, 200, 150)
        # 右半部分: 天空蓝 (R=100, G=150, B=255)
        right = pyvips.Image.black(width//2, height, bands=3)
        right = right.draw_rect1(100, 150, 255)
        # 拼接
        img = left.composite(right, "over")
        img = img.colourspace("srgb")
    return img


def test_detection():
    """测试灰度检测。"""
    print("灰度检测器测试")
    print("=" * 50)

    config = get_detection_config()
    print(f"检测配置: {config}")

    # 测试纯灰度
    print("\n1. 测试纯灰度图像:")
    img = create_test_image(100, 100, "grayscale")
    thumb = img.thumbnail_image(256, height=256)
    hsv = thumb.colourspace("srgb").colourspace("hsv")
    sat_np = pyvips.Image.new_from_memory(
        hsv[1].write_to_memory(), hsv[1].width, hsv[1].height, 1, "uchar"
    ).write_to_memory()
    import numpy as np
    sat_np = np.frombuffer(sat_np, dtype=np.uint8).flatten()
    
    is_grayscale, meta = detect_grayscale(thumb, sat_np)
    print(f"   结果: {'灰度' if is_grayscale else '彩色'}")
    print(f"   元数据: {meta}")

    # 测试泛绿黑白漫
    print("\n2. 测试泛绿黑白漫:")
    img = create_test_image(100, 100, "tinted_green")
    thumb = img.thumbnail_image(256, height=256)
    hsv = thumb.colourspace("srgb").colourspace("hsv")
    sat_np = pyvips.Image.new_from_memory(
        hsv[1].write_to_memory(), hsv[1].width, hsv[1].height, 1, "uchar"
    ).write_to_memory()
    sat_np = np.frombuffer(sat_np, dtype=np.uint8).flatten()
    
    is_grayscale, meta = detect_grayscale(thumb, sat_np)
    print(f"   结果: {'灰度' if is_grayscale else '彩色'}")
    print(f"   元数据: {meta}")

    # 测试真正彩漫
    print("\n3. 测试真正彩漫:")
    img = create_test_image(100, 100, "colorful")
    thumb = img.thumbnail_image(256, height=256)
    hsv = thumb.colourspace("srgb").colourspace("hsv")
    sat_np = pyvips.Image.new_from_memory(
        hsv[1].write_to_memory(), hsv[1].width, hsv[1].height, 1, "uchar"
    ).write_to_memory()
    sat_np = np.frombuffer(sat_np, dtype=np.uint8).flatten()
    
    is_grayscale, meta = detect_grayscale(thumb, sat_np)
    print(f"   结果: {'灰度' if is_grayscale else '彩色'}")
    print(f"   元数据: {meta}")

    print("\n" + "=" * 50)
    print("测试完成!")


if __name__ == "__main__":
    test_detection()
