#!/usr/bin/env python3
"""测试样本目录中的黑白/彩色漫画识别。"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyvips
import numpy as np
from image_compressor.grayscale_detector import (
    detect_grayscale,
    _vips_band_to_numpy,
    get_detection_config,
)
from image_compressor.scanner import IMAGE_EXTENSIONS


def analyze_image(image_path: Path) -> dict:
    """分析单张图像的灰度检测结果。"""
    try:
        # 使用 random access 模式以支持多次 write_to_memory 调用
        img = pyvips.Image.new_from_file(str(image_path), access="random")
        thumb = img.thumbnail_image(256, height=256)

        result = {
            "path": str(image_path),
            "width": img.width,
            "height": img.height,
            "bands": thumb.bands,
            "format": img.format,
            "is_grayscale": True,
            "detection_meta": {},
        }

        if thumb.bands >= 3:
            # 转换为 sRGB 再转 HSV
            srgb = thumb.colourspace("srgb")
            hsv = srgb.colourspace("hsv")
            
            # 提取色相和饱和度通道
            hue_band = hsv[0]
            hue_mem = hue_band.write_to_memory()
            hue_np = np.frombuffer(hue_mem, dtype=np.uint8).flatten()
            
            sat_band = hsv[1]
            sat_mem = sat_band.write_to_memory()
            sat_np = np.frombuffer(sat_mem, dtype=np.uint8).flatten()
            
            # 使用新的灰度检测器
            is_grayscale, meta = detect_grayscale(sat_np, hue_np)
            result["is_grayscale"] = is_grayscale
            result["detection_meta"] = meta
        else:
            result["detection_meta"] = {"method": "single_band"}

        return result
    except Exception as e:
        import traceback
        return {
            "path": str(image_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "is_grayscale": True,
            "detection_meta": {},
        }


def main():
    """主测试函数。"""
    print("=" * 70)
    print("样本漫画灰度检测测试")
    print("=" * 70)

    config = get_detection_config()
    print(f"\n检测配置:")
    print(f"  饱和度阈值: {config['sat_threshold']}")
    print(f"  彩色像素容差: {config['color_tolerance']}")
    print(f"  色相集中度阈值: {config['hue_threshold']}")
    print(f"  色相直方图分桶数: {config['hue_bins']}")
    print(f"  相邻桶跨度: {config['hue_span_bins']}")

    # 测试目录
    test_dirs = [
        ("samples/manga/black_and_white", "黑白漫画 (应该全是灰度)"),
        ("samples/manga/color", "彩色漫画 (应该全是彩色)"),
        ("samples/manga/raw", "原始样本 (混合)"),
    ]

    total_files = 0
    total_correct = 0
    misclassified = []

    for dir_path, description in test_dirs:
        print(f"\n{'='*70}")
        print(f"测试: {description}")
        print(f"目录: {dir_path}")
        print("=" * 70)

        if not Path(dir_path).exists():
            print(f"  目录不存在，跳过")
            continue

        # 获取所有图像文件
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(Path(dir_path).glob(f"*{ext}"))
            image_files.extend(Path(dir_path).glob(f"*{ext.upper()}"))

        # 去重并排序
        image_files = sorted(set(image_files))
        print(f"  找到 {len(image_files)} 个图像文件")

        if not image_files:
            continue

        # 分析每个文件
        grayscale_count = 0
        color_count = 0
        error_count = 0

        print(f"\n  {'文件名':<30} {'实际类型':<10} {'检测结果':<10} {'置信度':<10}")
        print(f"  {'-'*60}")

        for img_path in image_files[:50]:  # 限制前 50 个避免输出过多
            result = analyze_image(img_path)
            total_files += 1

            filename = img_path.name[:28]  # 截断长文件名

            if "error" in result:
                print(f"  {filename:<30} {'错误':<10} {result['error'][:30]}")
                error_count += 1
                # 只在调试模式下显示完整错误
                if '--debug' in sys.argv:
                    print(f"    完整错误: {result.get('traceback', 'N/A')}")
                continue

            meta = result["detection_meta"]
            color_ratio = meta.get("color_ratio", 0)
            hue_concentration = meta.get("hue_concentration", 0)

            # 判断实际类型 (基于目录名)
            if "black_and_white" in str(img_path):
                actual_type = "黑白"
                expected_grayscale = True
            elif "color" in str(img_path):
                actual_type = "彩色"
                expected_grayscale = False
            else:
                actual_type = "未知"
                expected_grayscale = None

            # 检测结果
            detected_type = "灰度" if result["is_grayscale"] else "彩色"
            is_correct = (
                expected_grayscale is not None
                and result["is_grayscale"] == expected_grayscale
            )

            if is_correct:
                total_correct += 1
                confidence = f"✓ {hue_concentration:.2f}"
            else:
                misclassified.append(result)
                confidence = f"✗ {hue_concentration:.2f}"

            print(
                f"  {filename:<30} {actual_type:<10} {detected_type:<10} {confidence}"
            )

            if result["is_grayscale"]:
                grayscale_count += 1
            else:
                color_count += 1

        print(f"\n  统计: 灰度={grayscale_count}, 彩色={color_count}, 错误={error_count}")

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"\n总文件数: {total_files}")
    print(f"正确分类: {total_correct}")
    print(f"准确率: {total_correct/total_files*100:.1f}%" if total_files > 0 else "N/A")

    if misclassified:
        print(f"\n误分类文件 ({len(misclassified)}):")
        for result in misclassified:
            meta = result["detection_meta"]
            print(f"  - {result['path']}")
            print(f"    颜色比例: {meta.get('color_ratio', 0):.3f}")
            print(f"    色相集中度: {meta.get('hue_concentration', 0):.3f}")
            print(f"    是否单色偏色: {meta.get('is_single_tint', False)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
