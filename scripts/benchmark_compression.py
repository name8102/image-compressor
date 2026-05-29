#!/usr/bin/env python3
"""压缩基准测试脚本。

测试不同压缩格式和参数的压缩率、速度和质量。

使用方式:
    python scripts/benchmark_compression.py
"""

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pyvips
from PIL import Image

# 配置
SAMPLE_DIR = Path("samples")
OUTPUT_DIR = Path("benchmark_results")

# 测试配置
TEST_CONFIGS = [
    # WebP 有损
    {"format": "webp", "lossy": True, "quality": 75},
    {"format": "webp", "lossy": True, "quality": 80},
    {"format": "webp", "lossy": True, "quality": 85},
    {"format": "webp", "lossy": True, "quality": 90},
    
    # WebP 无损
    {"format": "webp", "lossy": False, "quality": 100},
    
    # JPEG 质量调整
    {"format": "jpeg", "lossy": True, "quality": 75},
    {"format": "jpeg", "lossy": True, "quality": 80},
    {"format": "jpeg", "lossy": True, "quality": 85},
    
    # PNG 优化
    {"format": "png", "lossy": False, "quality": 100},
    
    # JPEG XL 有损
    {"format": "jxl", "lossy": True, "quality": 75},
    {"format": "jxl", "lossy": True, "quality": 80},
    {"format": "jxl", "lossy": True, "quality": 85},
    {"format": "jxl", "lossy": True, "quality": 90},
    
    # JPEG XL 无损
    {"format": "jxl", "lossy": False, "quality": 100},
    
    # AVIF 有损
    {"format": "avif", "lossy": True, "quality": 30},
    {"format": "avif", "lossy": True, "quality": 40},
    {"format": "avif", "lossy": True, "quality": 50},
    {"format": "avif", "lossy": True, "quality": 60},
    
    # AVIF 无损
    {"format": "avif", "lossy": False, "quality": 100},
]


def get_image_info(path: Path) -> dict[str, Any]:
    """获取图片信息"""
    try:
        img = pyvips.Image.new_from_file(str(path), access="sequential")
        return {
            "width": img.width,
            "height": img.height,
            "bands": img.bands,
            "size": path.stat().st_size,
        }
    except Exception as e:
        return {"error": str(e)}


def compress_image_jxl_cli(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """使用 cjxl 命令行工具压缩 JPEG XL"""
    start_time = time.perf_counter()
    
    try:
        quality = config.get("quality", 85)
        lossy = config.get("lossy", True)
        
        cmd = ["cjxl", str(input_path), str(output_path)]
        
        if lossy:
            cmd.extend(["--quality", str(quality)])
        else:
            cmd.extend(["--quality", "100", "--lossless"])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            return {"error": f"cjxl failed: {result.stderr}"}
        
        elapsed = time.perf_counter() - start_time
        output_size = output_path.stat().st_size
        
        return {
            "success": True,
            "output_size": output_size,
            "time": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"error": "cjxl timeout"}
    except FileNotFoundError:
        return {"error": "cjxl not found"}
    except Exception as e:
        return {"error": str(e)}


def compress_image_avif_cli(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """使用 avifenc 命令行工具压缩 AVIF"""
    start_time = time.perf_counter()
    
    try:
        quality = config.get("quality", 50)
        lossy = config.get("lossy", True)
        
        cmd = ["avifenc"]
        
        if lossy:
            cmd.extend(["--min", str(quality), "--max", str(quality), "--speed", "6"])
        else:
            cmd.extend(["--lossless", "--speed", "6"])
        
        cmd.extend([str(input_path), str(output_path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            return {"error": f"avifenc failed: {result.stderr}"}
        
        elapsed = time.perf_counter() - start_time
        output_size = output_path.stat().st_size
        
        return {
            "success": True,
            "output_size": output_size,
            "time": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"error": "avifenc timeout"}
    except FileNotFoundError:
        return {"error": "avifenc not found"}
    except Exception as e:
        return {"error": str(e)}


def compress_image_pyvips(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """使用 pyvips 压缩图片"""
    start_time = time.perf_counter()
    
    try:
        img = pyvips.Image.new_from_file(str(input_path), access="sequential")
        
        fmt = config["format"]
        quality = config.get("quality", 85)
        lossy = config.get("lossy", True)
        
        if fmt == "webp":
            img.webpsave(
                str(output_path),
                Q=quality,
                lossless=not lossy,
                strip=True,
            )
        elif fmt == "jpeg":
            # 转换为 RGB（如果需要）
            if img.bands == 4:
                img = img.flatten()
            elif img.bands == 1:
                img = img.colourspace("srgb")
            img.jpegsave(
                str(output_path),
                Q=quality,
                strip=True,
            )
        elif fmt == "png":
            img.pngsave(
                str(output_path),
                compression=9,
                strip=True,
            )
        else:
            return {"error": f"Unsupported format: {fmt}"}
        
        elapsed = time.perf_counter() - start_time
        output_size = output_path.stat().st_size
        
        return {
            "success": True,
            "output_size": output_size,
            "time": elapsed,
        }
    except Exception as e:
        return {"error": str(e)}


def compress_image_pillow(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """使用 Pillow 压缩图片（备用方案）"""
    start_time = time.perf_counter()
    
    try:
        with Image.open(input_path) as img:
            fmt = config["format"]
            quality = config.get("quality", 85)
            lossy = config.get("lossy", True)
            
            if fmt == "webp":
                img.save(
                    output_path,
                    format="WEBP",
                    quality=quality,
                    lossless=not lossy,
                    method=4,
                )
            elif fmt == "jpeg":
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(
                    output_path,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                )
            elif fmt == "png":
                img.save(
                    output_path,
                    format="PNG",
                    optimize=True,
                )
            else:
                return {"error": f"Unsupported format: {fmt}"}
        
        elapsed = time.perf_counter() - start_time
        output_size = output_path.stat().st_size
        
        return {
            "success": True,
            "output_size": output_size,
            "time": elapsed,
        }
    except Exception as e:
        return {"error": str(e)}


def test_single_image(
    image_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """测试单个图片的压缩"""
    original_size = image_path.stat().st_size
    
    # 生成输出文件名
    fmt = config["format"]
    quality = config.get("quality", 100)
    lossy_str = "lossy" if config.get("lossy", True) else "lossless"
    output_name = f"{image_path.stem}_{fmt}_{lossy_str}_q{quality}{image_path.suffix}"
    output_path = output_dir / output_name
    
    # 使用 pyvips 压缩
    result = compress_image_pyvips(image_path, output_path, config)
    
    # 如果 pyvips 失败，使用 CLI 工具
    if "error" in result:
        fmt = config["format"]
        if fmt == "jxl":
            result = compress_image_jxl_cli(image_path, output_path, config)
        elif fmt == "avif":
            result = compress_image_avif_cli(image_path, output_path, config)
        else:
            result = compress_image_pillow(image_path, output_path, config)
    
    if "error" in result:
        return {
            "image": str(image_path),
            "config": config,
            "error": result["error"],
        }
    
    compressed_size = result["output_size"]
    ratio = compressed_size / original_size * 100 if original_size > 0 else 0
    
    return {
        "image": str(image_path),
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": ratio,
        "time": result["time"],
        "config": config,
    }


def run_benchmark():
    """运行基准测试"""
    print("=" * 60)
    print("压缩基准测试")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 收集所有测试图片
    all_images = []
    
    # 漫画图片
    manga_dir = SAMPLE_DIR / "manga" / "raw"
    if manga_dir.exists():
        for f in manga_dir.rglob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                all_images.append(("manga", f))
    
    # 照片
    photo_dir = SAMPLE_DIR / "photos" / "raw"
    if photo_dir.exists():
        for f in photo_dir.rglob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                all_images.append(("photo", f))
    
    print(f"\n找到 {len(all_images)} 张测试图片")
    print(f"测试配置: {len(TEST_CONFIGS)} 种")
    print(f"总测试数: {len(all_images) * len(TEST_CONFIGS)}")
    
    # 运行测试
    results = []
    total_tests = len(all_images) * len(TEST_CONFIGS)
    completed = 0
    
    for category, image_path in all_images:
        print(f"\n处理: {image_path.name} ({category})")
        
        for config in TEST_CONFIGS:
            fmt = config["format"]
            quality = config.get("quality", 100)
            lossy_str = "有损" if config.get("lossy", True) else "无损"
            
            print(f"  {fmt} {lossy_str} q={quality}...", end=" ", flush=True)
            
            # 创建类别输出目录
            category_output_dir = OUTPUT_DIR / category / f"{fmt}_{lossy_str}_q{quality}"
            category_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 测试压缩
            result = test_single_image(image_path, config, category_output_dir)
            results.append(result)
            
            if "error" in result:
                print(f"错误: {result['error']}")
            else:
                print(f"{result['ratio']:.1f}% ({result['time']:.2f}s)")
            
            completed += 1
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存测试结果...")
    
    # 保存 CSV
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "category", "format", "lossy", "quality",
            "original_size", "compressed_size", "ratio", "time"
        ])
        
        for result in results:
            if "error" not in result:
                writer.writerow([
                    Path(result["image"]).name,
                    "manga" if "manga" in result["image"] else "photo",
                    result["config"]["format"],
                    result["config"].get("lossy", True),
                    result["config"].get("quality", 100),
                    result["original_size"],
                    result["compressed_size"],
                    f"{result['ratio']:.2f}",
                    f"{result['time']:.3f}",
                ])
    
    print(f"CSV 结果已保存到: {csv_path}")
    
    # 保存 JSON
    json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON 结果已保存到: {json_path}")
    
    # 生成统计报告
    print("\n" + "=" * 60)
    print("压缩率统计")
    print("=" * 60)
    
    # 按配置分组统计
    stats_by_config: dict[str, list[float]] = {}
    
    for result in results:
        if "error" not in result:
            config_key = f"{result['config']['format']}_{'lossy' if result['config'].get('lossy', True) else 'lossless'}_q{result['config'].get('quality', 100)}"
            if config_key not in stats_by_config:
                stats_by_config[config_key] = []
            stats_by_config[config_key].append(result["ratio"])
    
    print("\n配置 | 平均压缩率 | 最小 | 最大 | 样本数")
    print("-" * 60)
    
    for config_key, ratios in sorted(stats_by_config.items()):
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        print(f"{config_key:30} | {avg_ratio:6.1f}% | {min_ratio:5.1f}% | {max_ratio:5.1f}% | {len(ratios)}")
    
    # 按类别统计
    print("\n" + "=" * 60)
    print("按类别统计")
    print("=" * 60)
    
    stats_by_category: dict[str, dict[str, list[float]]] = {}
    
    for result in results:
        if "error" not in result:
            category = "manga" if "manga" in result["image"] else "photo"
            config_key = f"{result['config']['format']}_{'lossy' if result['config'].get('lossy', True) else 'lossless'}_q{result['config'].get('quality', 100)}"
            
            if category not in stats_by_category:
                stats_by_category[category] = {}
            if config_key not in stats_by_category[category]:
                stats_by_category[category][config_key] = []
            
            stats_by_category[category][config_key].append(result["ratio"])
    
    for category, configs in stats_by_category.items():
        print(f"\n{category}:")
        for config_key, ratios in sorted(configs.items()):
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  {config_key:30} | {avg_ratio:6.1f}%")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"\n详细结果已保存到: {OUTPUT_DIR}")
    print("\n下一步:")
    print("1. 查看 CSV/JSON 文件了解详细数据")
    print("2. 根据压缩率选择最优配置")
    print("3. 在 config.toml 中配置预设参数")


if __name__ == "__main__":
    run_benchmark()
