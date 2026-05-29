#!/usr/bin/env python3
"""可视化误分类的图像，帮助理解问题。"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyvips
import numpy as np


def create_thumbnail_with_info(image_path: Path, output_path: Path):
    """创建带信息的缩略图。"""
    try:
        img = pyvips.Image.new_from_file(str(image_path), access="random")
        thumb = img.thumbnail_image(200, height=200)

        # 添加文本标注
        filename = image_path.name[:20]
        thumb = thumb.text(
            filename,
            font="sans 12",
            width=200,
            height=20,
        )
        thumb.write_to_file(str(output_path))
        print(f"  已保存: {output_path}")
    except Exception as e:
        print(f"  错误: {e}")


def analyze_color_diversity(image_path: Path):
    """分析图像的颜色多样性。"""
    try:
        img = pyvips.Image.new_from_file(str(image_path), access="random")
        thumb = img.thumbnail_image(256, height=256)

        if thumb.bands < 3:
            return

        srgb = thumb.colourspace("srgb")
        hsv = srgb.colourspace("hsv")

        # 提取 RGB 通道
        r_np = np.frombuffer(srgb[0].write_to_memory(), dtype=np.uint8).flatten()
        g_np = np.frombuffer(srgb[1].write_to_memory(), dtype=np.uint8).flatten()
        b_np = np.frombuffer(srgb[2].write_to_memory(), dtype=np.uint8).flatten()

        # 计算颜色直方图
        hist_r, _ = np.histogram(r_np, bins=16, range=(0, 256))
        hist_g, _ = np.histogram(g_np, bins=16, range=(0, 256))
        hist_b, _ = np.histogram(b_np, bins=16, range=(0, 256))

        # 计算非零 bin 的数量 (颜色多样性)
        unique_r = np.count_nonzero(hist_r)
        unique_g = np.count_nonzero(hist_g)
        unique_b = np.count_nonzero(hist_b)

        print(f"  RGB 颜色多样性: R={unique_r}, G={unique_g}, B={unique_b}")
        print(f"  总多样性: {unique_r + unique_g + unique_b}")

        # 计算 RGB 相关性
        if len(r_np) > 1000:
            corr_rg = np.corrcoef(r_np[:1000], g_np[:1000])[0, 1]
            corr_rb = np.corrcoef(r_np[:1000], b_np[:1000])[0, 1]
            corr_gb = np.corrcoef(g_np[:1000], b_np[:1000])[0, 1]
            print(f"  RGB 相关性: RG={corr_rg:.3f}, RB={corr_rb:.3f}, GB={corr_gb:.3f}")

            # 高相关性表明颜色是同步变化的 (可能是单色偏色)
            avg_corr = (corr_rg + corr_rb + corr_gb) / 3
            print(f"  平均相关性: {avg_corr:.3f}")
            if avg_corr > 0.8:
                print(f"  → 高度相关: 可能是单色偏色")
            elif avg_corr > 0.5:
                print(f"  → 中等相关: 可能是有限色板")
            else:
                print(f"  → 低相关: 真正彩漫")

    except Exception as e:
        print(f"  错误: {e}")


def main():
    """主函数。"""
    print("=" * 70)
    print("可视化分析误分类的彩色漫画")
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

    # 创建输出目录
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    for file_path in misclassified_files:
        print(f"\n{'='*70}")
        print(f"分析: {file_path}")
        print("=" * 70)

        if not Path(file_path).exists():
            print("  文件不存在")
            continue

        # 创建缩略图
        img_path = Path(file_path)
        output_path = output_dir / f"{img_path.stem}_thumb.jpg"
        create_thumbnail_with_info(img_path, output_path)

        # 分析颜色多样性
        analyze_color_diversity(img_path)

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
