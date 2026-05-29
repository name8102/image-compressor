#!/usr/bin/env python3
"""检查误分类图像的 RGB 值。"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyvips
import numpy as np


def check_rgb_values(image_path: Path):
    """检查图像的 RGB 值关系。"""
    try:
        img = pyvips.Image.new_from_file(str(image_path), access="random")
        thumb = img.thumbnail_image(256, height=256)

        if thumb.bands < 3:
            print("  图像不是 RGB 格式")
            return

        srgb = thumb.colourspace("srgb")

        # 提取 RGB 通道
        r_np = np.frombuffer(srgb[0].write_to_memory(), dtype=np.uint8).flatten()
        g_np = np.frombuffer(srgb[1].write_to_memory(), dtype=np.uint8).flatten()
        b_np = np.frombuffer(srgb[2].write_to_memory(), dtype=np.uint8).flatten()

        # 采样一些像素点
        sample_indices = np.random.choice(len(r_np), min(10, len(r_np)), replace=False)

        print(f"  前 10 个像素的 RGB 值:")
        print(f"  {'像素':<10} {'R':<10} {'G':<10} {'B':<10} {'R/G':<10} {'R/B':<10} {'G/B':<10}")
        print(f"  {'-'*70}")

        for i, idx in enumerate(sample_indices[:10]):
            r, g, b = r_np[idx], g_np[idx], b_np[idx]
            rg_ratio = r / g if g > 0 else float('inf')
            rb_ratio = r / b if b > 0 else float('inf')
            gb_ratio = g / b if b > 0 else float('inf')
            print(f"  {i+1:<10} {r:<10} {g:<10} {b:<10} {rg_ratio:<10.3f} {rb_ratio:<10.3f} {gb_ratio:<10.3f}")

        # 计算统计信息
        print(f"\n  统计信息:")
        print(f"    R 范围: {r_np.min()}-{r_np.max()}")
        print(f"    G 范围: {g_np.min()}-{g_np.max()}")
        print(f"    B 范围: {b_np.min()}-{b_np.max()}")

        # 检查是否 R=G=B (灰度)
        r_equals_g = np.allclose(r_np, g_np, atol=1)
        r_equals_b = np.allclose(r_np, b_np, atol=1)
        g_equals_b = np.allclose(g_np, b_np, atol=1)

        print(f"\n  R ≈ G: {r_equals_g}")
        print(f"    最大差值: {np.max(np.abs(r_np.astype(int) - g_np.astype(int)))}")
        print(f"    平均差值: {np.mean(np.abs(r_np.astype(int) - g_np.astype(int))):.2f}")

        print(f"  R ≈ B: {r_equals_b}")
        print(f"    最大差值: {np.max(np.abs(r_np.astype(int) - b_np.astype(int)))}")
        print(f"    平均差值: {np.mean(np.abs(r_np.astype(int) - b_np.astype(int))):.2f}")

        print(f"  G ≈ B: {g_equals_b}")
        print(f"    最大差值: {np.max(np.abs(g_np.astype(int) - b_np.astype(int)))}")
        print(f"    平均差值: {np.mean(np.abs(g_np.astype(int) - b_np.astype(int))):.2f}")

        # 判断
        if r_equals_g and r_equals_b and g_equals_b:
            print(f"\n  结论: 纯灰度图 (R=G=B)")
        elif r_equals_g and not r_equals_b:
            print(f"\n  结论: 红色偏色 (R=G, 但 B 不同)")
        elif r_equals_b and not r_equals_g:
            print(f"\n  结论: 绿色偏色 (R=B, 但 G 不同)")
        elif g_equals_b and not r_equals_g:
            print(f"\n  结论: 蓝色偏色 (G=B, 但 R 不同)")
        else:
            print(f"\n  结论: 有颜色差异")

    except Exception as e:
        import traceback
        print(f"  错误: {e}")
        print(traceback.format_exc())


def main():
    """主函数。"""
    print("=" * 70)
    print("检查误分类图像的 RGB 值")
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

        check_rgb_values(Path(file_path))

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
