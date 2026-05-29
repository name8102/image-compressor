#!/usr/bin/env python3
"""分析误分类的彩色漫画图像。"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyvips
import numpy as np
from image_compressor.grayscale_detector import (
    _vips_band_to_numpy,
    _analyze_hue_concentration,
    HUE_HISTOGRAM_BINS,
)


def analyze_hue_distribution(image_path: Path):
    """分析图像的色相分布。"""
    try:
        img = pyvips.Image.new_from_file(str(image_path), access="random")
        thumb = img.thumbnail_image(256, height=256)

        if thumb.bands < 3:
            print("  图像不是 RGB 格式")
            return

        srgb = thumb.colourspace("srgb")
        hsv = srgb.colourspace("hsv")

        # 提取色相和饱和度
        hue_np = _vips_band_to_numpy(hsv[0])
        sat_np = _vips_band_to_numpy(hsv[1])

        # 过滤高饱和度像素
        sat_threshold = 30
        color_mask = sat_np > sat_threshold
        color_pixels = np.count_nonzero(color_mask)
        total_pixels = sat_np.size
        color_ratio = color_pixels / total_pixels

        print(f"  饱和度 > {sat_threshold} 的像素: {color_pixels}/{total_pixels} ({color_ratio:.3f})")

        if color_pixels == 0:
            print("  没有高饱和度像素")
            return

        # 获取色相分布
        valid_hues = hue_np[color_mask]

        # 创建色相直方图
        hist, bin_edges = np.histogram(valid_hues, bins=HUE_HISTOGRAM_BINS, range=(0, 360))

        # 计算集中度
        concentration, _ = _analyze_hue_concentration(valid_hues)

        print(f"  色相集中度: {concentration:.3f}")
        print(f"\n  色相分布 (12 个桶, 每桶 30°):")
        print(f"  {'桶':<10} {'范围':<15} {'像素数':<10} {'占比':<10}")
        print(f"  {'-'*50}")

        # 定义色系名称
        hue_names = [
            "红", "橙", "黄", "黄绿", "绿", "青绿",
            "青", "天蓝", "蓝", "靛", "紫", "品红"
        ]

        for i, (count, name) in enumerate(zip(hist, hue_names)):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            ratio = count / color_pixels if color_pixels > 0 else 0
            bar = "█" * int(ratio * 50)
            print(f"  {name:<10} {start:>5.0f}°-{end:>5.0f}°   {count:<10} {ratio:<10.3f} {bar}")

        # 环形处理: 检查红色跨 0°/360° 的情况
        padded_hist = np.concatenate((hist, hist[:2]))
        window_sums = padded_hist[:-1] + padded_hist[1:]
        max_window_sum = window_sums.max()
        max_window_idx = window_sums.argmax()

        # 计算最大连续 60° 范围
        start_idx = max_window_idx % HUE_HISTOGRAM_BINS
        end_idx = (max_window_idx + 1) % HUE_HISTOGRAM_BINS

        print(f"\n  最大连续 60° 范围:")
        print(f"    从 {start_idx}: {hue_names[start_idx]} ({hist[start_idx]} 像素)")
        print(f"    到 {end_idx}: {hue_names[end_idx]} ({hist[end_idx]} 像素)")
        print(f"    总计: {max_window_sum} 像素 ({max_window_sum/color_pixels:.3f})")

        # 分析主色系
        primary_hue_idx = np.argmax(hist)
        primary_hue_name = hue_names[primary_hue_idx]
        primary_ratio = hist[primary_hue_idx] / color_pixels

        print(f"\n  主色系: {primary_hue_name} ({primary_ratio:.3f})")

        # 判断是否为单色偏色
        if concentration > 0.9:
            print(f"\n  结论: 高度集中 → 可能是单色偏色 (泛{primary_hue_name})")
            print(f"  (色相集中度 {concentration:.3f} > 阈值 0.9)")
        else:
            print(f"\n  结论: 色相分散 → 真正彩漫")
            print(f"  (色相集中度 {concentration:.3f} < 阈值 0.9)")

    except Exception as e:
        import traceback
        print(f"  错误: {e}")
        print(traceback.format_exc())


def main():
    """主函数。"""
    print("=" * 70)
    print("分析误分类的彩色漫画图像")
    print("=" * 70)

    # 误分类的文件
    misclassified_files = [
        "samples/manga/color/manga_color_016.jpg",
        "samples/manga/color/manga_color_02.jpg",
        "samples/manga/color/manga_color_14.jpg",
        "samples/manga/color/manga_color_18_1.jpg",
        "samples/manga/color/manga_color_25_1.jpg",
        "samples/manga/color/manga_color_35.jpg",
    ]

    for file_path in misclassified_files:
        print(f"\n{'='*70}")
        print(f"分析: {file_path}")
        print("=" * 70)

        if not Path(file_path).exists():
            print("  文件不存在")
            continue

        analyze_hue_distribution(Path(file_path))

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
