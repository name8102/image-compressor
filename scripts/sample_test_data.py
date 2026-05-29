#!/usr/bin/env python3
"""从 NAS 抽取代表性样本用于 TB 级归档压缩测试。

架构:
    Phase 1: scandir 快速扫描 → 收集文件元数据（stat 缓存，避免 NFS roundtrip）
    Phase 2: 并发 pyvips 特征提取 → 熵、边缘密度、色彩方差、分辨率
    Phase 3: 批量写 manifest (SQLite WAL + JSON)
    Phase 4: 特征量化分层抽样
    Phase 5: hardlink → copy 样本落盘（st_dev 预检）
    Phase 6: CBZ 代表性页面抽取

使用方式:
    .venv/bin/python3 scripts/sample_test_data.py
    或: uv run python3 scripts/sample_test_data.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyvips

# 导入灰度检测模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from image_compressor.grayscale_detector import (
    detect_grayscale,
    get_detection_config,
)

# ── 配置 ──────────────────────────────────────────────────────────────

SAMPLE_DIR = Path("samples")
MANIFEST_DB = SAMPLE_DIR / "manifest.db"
MANIFEST_JSON = SAMPLE_DIR / "manifest.json"

STORAGE_BASE = Path.home() / "storage"
MANGA_DIR = STORAGE_BASE / "Temp"
PHOTO_DIR = STORAGE_BASE / "本地资源库" / "福利姬"
CBZ_DIR = STORAGE_BASE / "Hentai"

MANGA_SAMPLES = 60
PHOTO_SAMPLES = 40
CBZ_SAMPLES = 8

# 特征提取时缩略图最大边长（平衡精度与速度）
THUMBNAIL_SIZE = 256

# 灰度检测：被判为"彩色"的最低饱和度 (0-255, >30 过滤纸张泛黄/JPEG噪点)
COLOR_PIXEL_SATURATION_THRESHOLD = 30
# 灰度检测：彩色像素占比 < 此值 → 判为黑白 (容忍小面积Logo/水印)
COLOR_RATIO_TOLERANCE = 0.03

# CBZ 每归档最多抽取页数
CBZ_MAX_PAGES = 20

# 并发提取线程数（NAS 场景不宜过高）
EXTRACT_WORKERS = 8

# NFS / 跨文件系统：hardlink 失败后自动回退 copy
# 脚本启动时通过 st_dev 预检自动修正
PREFER_HARDLINK = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".jxl", ".bmp", ".tiff", ".tif", ".gif"}
CBZ_EXTENSIONS = {".cbz"}

# pyvips format → numpy dtype 映射
_BAND_FORMAT_DTYPE: dict[str, np.dtype] = {
    "uchar": np.dtype(np.uint8),
    "char": np.dtype(np.int8),
    "ushort": np.dtype(np.uint16),
    "short": np.dtype(np.int16),
    "uint": np.dtype(np.uint32),
    "int": np.dtype(np.int32),
    "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
}


def _numpy_dtype(vips_img: pyvips.Image) -> np.dtype:
    """pyvips 图像格式 → numpy dtype。"""
    return _BAND_FORMAT_DTYPE.get(vips_img.format, np.dtype(np.float32))


# ── 数据结构 ──────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    """扫描条目：路径 + scandir 缓存的 stat 尺寸，避免 NFS 二次 stat。"""
    path: str
    size: int


@dataclass
class ImageFeatures:
    """图像复杂度特征，用于分层抽样。"""
    path: str
    width: int
    height: int
    megapixels: float

    entropy: float             # Shannon 熵 (bits)，信息密度
    edge_density: float        # 边缘像素占比，结构复杂度
    color_variance: float      # RGB 通道方差均值，色彩丰富度
    unique_colors_ratio: float # 唯一色占比 (≈ palette cardinality)

    is_grayscale: bool
    is_animated: bool = False
    frames: int = 1
    image_format: str = "unknown"
    grayscale_detection_meta: dict = field(default_factory=dict)  # 灰度检测诊断信息

    file_size: int = 0

    # 综合复杂度评分（用于分层）
    complexity_score: float = 0.0


@dataclass
class CbzInfo:
    """CBZ 归档元数据。"""
    path: str
    file_size: int
    image_count: int
    total_image_bytes: int      # 内部图片总大小（压缩前）
    avg_page_bytes: float       # 平均每页大小
    png_ratio: float            # PNG 占比 → 扫描稿概率
    webp_ratio: float           # WebP 占比 → 已优化概率
    formats: dict[str, int] = field(default_factory=dict)


# ── 文件系统扫描 ──────────────────────────────────────────────────────

def scan_directory(
    root: Path,
    extensions: set[str],
    max_files: int = 20000,
) -> list[FileEntry]:
    """os.scandir 递归扫描，缓存文件尺寸避免 NFS 二次 stat。

    对 NFS 挂载点仍然高效：scandir 批量返回 dirent（含 stat 缓存），
    DirEntry.stat() 通常零 roundtrip。
    """
    entries_out: list[FileEntry] = []
    stack = [root]

    while stack and len(entries_out) < max_files:
        d = stack.pop()
        try:
            with os.scandir(d) as dir_entries:
                for entry in dir_entries:
                    if len(entries_out) >= max_files:
                        break
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        if Path(entry.name).suffix.lower() in extensions:
                            size = entry.stat(follow_symlinks=False).st_size
                            entries_out.append(FileEntry(path=entry.path, size=size))
        except (OSError, PermissionError):
            continue

    return entries_out


# ── 图像特征提取 (pyvips) ─────────────────────────────────────────────

def _compute_entropy_from_histogram(hist: pyvips.Image) -> float:
    """从 pyvips 直方图计算 Shannon 熵。"""
    mem = hist.write_to_memory()
    arr = np.frombuffer(mem, dtype=_numpy_dtype(hist))

    if len(arr) == 0:
        return 0.0

    total = arr.sum()
    if total == 0:
        return 0.0

    probs = arr[arr > 0].astype(np.float64) / float(total)
    return float(-np.sum(probs * np.log2(probs)))


def _compute_edge_density(grey_np: np.ndarray) -> float:
    """梯度幅值边缘密度。L1 范数近似，无 sqrt，CPU 友好。

    对 THUMBNAIL_SIZE×* 的缩略图计算，毫秒级。
    """
    if grey_np.size < 4:
        return 0.0

    gy = np.abs(np.diff(grey_np.astype(np.float32), axis=0))
    gx = np.abs(np.diff(grey_np.astype(np.float32), axis=1))

    h = min(gy.shape[0], gx.shape[0])
    w = min(gy.shape[1], gx.shape[1])
    gy = gy[:h, :w]
    gx = gx[:h, :w]

    # L1 范数：|Gx| + |Gy|，速度 ≈ 2× L2，分层抽样排名无损
    mag = gy + gx
    edges = (mag > 30).sum()
    total = mag.size
    return float(edges / total) if total > 0 else 0.0


def _vips_band_to_numpy(band: pyvips.Image) -> np.ndarray:
    """pyvips 波段 → numpy 1D 数组。"""
    mem = band.write_to_memory()
    dtype = _numpy_dtype(band)
    return np.frombuffer(mem, dtype=dtype).flatten()


def _vips_to_numpy_2d(img: pyvips.Image) -> np.ndarray:
    """pyvips 单波段 → numpy 2D 数组。"""
    mem = img.write_to_memory()
    dtype = _numpy_dtype(img)
    arr = np.frombuffer(mem, dtype=dtype)
    return arr.reshape(img.height, img.width)


def _unique_colors_from_hist(hist: pyvips.Image, megapixels: float) -> float:
    """从直方图估算唯一色比例。非零 bin 数 / 理论最大值。"""
    mem = hist.write_to_memory()
    arr = np.frombuffer(mem, dtype=_numpy_dtype(hist))
    nonzero = int((arr > 0).sum())
    total_bins = arr.size
    ratio = nonzero / max(total_bins, 1)
    return float(min(ratio * np.log2(megapixels + 1) / 8, 1.0))


def extract_features(
    image_path: Path,
    cached_size: int = 0,
) -> Optional[ImageFeatures]:
    """用 pyvips 提取图像复杂度特征。

    sequential access 优先，progressive JPEG 自动回退 random access。
    cached_size: scandir 缓存的文件大小，避免 NFS 二次 stat。
    """
    for access_mode in ("sequential", "random"):
        try:
            img = pyvips.Image.new_from_file(str(image_path), access=access_mode)

            width = img.width
            height = img.height
            megapixels = (width * height) / 1_000_000

            try:
                image_format = str(img.get("vips-loader"))
            except Exception:
                image_format = "unknown"

            thumb = img.thumbnail_image(THUMBNAIL_SIZE, height=THUMBNAIL_SIZE)

            if thumb.bands >= 3:
                grey = thumb.colourspace("b-w")
            else:
                grey = thumb

            hist = grey.hist_find()
            entropy = _compute_entropy_from_histogram(hist)

            grey_np = _vips_to_numpy_2d(grey)
            edge_density = _compute_edge_density(grey_np)

            # 灰度检测诊断信息
            detection_meta = {}

            if thumb.bands >= 3:
                hsv = thumb.colourspace("srgb").colourspace("hsv")
                # 提取色相和饱和度通道
                hue_band = hsv[0]
                hue_np = _vips_band_to_numpy(hue_band)
                
                saturation_band = hsv[1]
                sat_np = _vips_band_to_numpy(saturation_band)

                rgb = thumb.colourspace("srgb")
                variances: list[float] = []
                for i in range(min(rgb.bands, 3)):
                    band_np = _vips_band_to_numpy(rgb[i])
                    variances.append(float(np.var(band_np)))
                color_variance = float(np.mean(variances)) if variances else 0.0

                rgb_hist = thumb.hist_find()
                unique_colors_ratio = _unique_colors_from_hist(rgb_hist, megapixels)

                # 使用增强版灰度检测 (色相集中度分析)
                is_grayscale, detection_meta = detect_grayscale(sat_np, hue_np)
            else:
                color_variance = 0.0
                unique_colors_ratio = 0.0
                is_grayscale = True

            is_animated = False
            frames = 1
            try:
                n_pages = img.get_n_pages()
                if n_pages > 1:
                    is_animated = True
                    frames = n_pages
            except Exception:
                pass

            file_size = cached_size if cached_size > 0 else image_path.stat().st_size
            complexity = float(entropy * (edge_density + 0.01) * np.log2(megapixels + 1))

            return ImageFeatures(
                path=str(image_path),
                width=width, height=height, megapixels=float(megapixels),
                entropy=float(entropy), edge_density=float(edge_density),
                color_variance=float(color_variance), unique_colors_ratio=float(unique_colors_ratio),
                is_grayscale=is_grayscale, is_animated=is_animated, frames=frames,
                image_format=str(image_format), file_size=file_size,
                complexity_score=float(complexity),
                grayscale_detection_meta=detection_meta if thumb.bands >= 3 else {},
            )
        except Exception as e:
            if access_mode == "sequential":
                continue
            print(f"  ⚠ 特征提取失败 {image_path.name}: {e}")
            return None

    return None


# ── CBZ 分析 ──────────────────────────────────────────────────────────

def analyze_cbz(cbz_path: Path, cached_size: int = 0) -> Optional[CbzInfo]:
    """分析 CBZ 内部结构，不解压全部内容。"""
    import zipfile
    try:
        with zipfile.ZipFile(cbz_path, "r") as zf:
            entries = zf.infolist()
            image_entries = [
                e for e in entries
                if not e.is_dir()
                and Path(e.filename).suffix.lower() in IMAGE_EXTENSIONS
            ]

            if not image_entries:
                return None

            formats: dict[str, int] = {}
            total_bytes = 0
            for e in image_entries:
                ext = Path(e.filename).suffix.lower()
                formats[ext] = formats.get(ext, 0) + 1
                total_bytes += e.file_size

            image_count = len(image_entries)
            avg_page = total_bytes / image_count if image_count > 0 else 0
            png_ratio = formats.get(".png", 0) / image_count
            webp_ratio = formats.get(".webp", 0) / image_count
            file_size = cached_size if cached_size > 0 else cbz_path.stat().st_size

            return CbzInfo(
                path=str(cbz_path),
                file_size=file_size,
                image_count=image_count,
                total_image_bytes=total_bytes,
                avg_page_bytes=avg_page,
                png_ratio=png_ratio,
                webp_ratio=webp_ratio,
                formats=formats,
            )
    except Exception as e:
        print(f"  ⚠ CBZ 分析失败 {cbz_path.name}: {e}")
        return None


def extract_cbz_pages(
    cbz_path: Path, dst_dir: Path, max_pages: int = CBZ_MAX_PAGES,
) -> list[Path]:
    """安全抽取 CBZ 代表性页面：前 N/2 页 + 随机 N/2 页。"""
    import zipfile
    extracted: list[Path] = []
    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(cbz_path, "r") as zf:
            entries = zf.infolist()
            image_entries = [
                e for e in entries
                if not e.is_dir()
                and Path(e.filename).suffix.lower() in IMAGE_EXTENSIONS
            ]

            if not image_entries:
                return extracted

            n_first = min(max_pages // 2, len(image_entries))
            first_pages = image_entries[:n_first]

            remaining = [e for e in image_entries[n_first:]]
            n_random = min(max_pages - n_first, len(remaining))
            random_pages = random.sample(remaining, n_random) if n_random > 0 else []

            selected = first_pages + random_pages

            for entry in selected:
                safe_name = Path(entry.filename).name
                if not safe_name:
                    continue
                dst = dst_dir / f"{cbz_path.stem}_{safe_name}"
                try:
                    with zf.open(entry) as src_f:
                        dst.write_bytes(src_f.read())
                    extracted.append(dst)
                except Exception as e:
                    print(f"    ⚠ 抽取页面失败 {entry.filename}: {e}")

        return extracted
    except Exception as e:
        print(f"  ⚠ CBZ 解压失败 {cbz_path.name}: {e}")
        return extracted


# ── 复制 / Hardlink ───────────────────────────────────────────────────

def link_or_copy(src: Path, dst: Path) -> Optional[Path]:
    """hardlink → copy2 回退。如果 st_dev 预检已禁用 hardlink，直接 copy。"""
    dst.parent.mkdir(parents=True, exist_ok=True)

    base = dst
    counter = 1
    while dst.exists():
        dst = base.parent / f"{base.stem}_{counter}{base.suffix}"
        counter += 1

    if PREFER_HARDLINK:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            pass

    try:
        shutil.copy2(src, dst)
        return dst
    except (OSError, PermissionError) as e:
        print(f"  ⚠ 复制失败 {src}: {e}")
        return None


# ── SQLite Manifest (批量写入) ─────────────────────────────────────────

def init_manifest_db(db_path: Path) -> sqlite3.Connection:
    """初始化 manifest SQLite 数据库（WAL 模式）。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            path TEXT PRIMARY KEY,
            width INTEGER, height INTEGER, megapixels REAL,
            entropy REAL, edge_density REAL, color_variance REAL,
            unique_colors_ratio REAL,
            is_grayscale INTEGER, is_animated INTEGER, frames INTEGER,
            image_format TEXT, file_size INTEGER, complexity_score REAL,
            extracted_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cbz_info (
            path TEXT PRIMARY KEY,
            file_size INTEGER, image_count INTEGER,
            total_image_bytes INTEGER, avg_page_bytes REAL,
            png_ratio REAL, webp_ratio REAL,
            formats TEXT, extracted_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            src_path TEXT, dst_path TEXT, category TEXT, tier TEXT,
            copied_at REAL
        )
    """)
    conn.commit()
    return conn


def insert_features_batch(
    conn: sqlite3.Connection, features: list[ImageFeatures],
) -> None:
    """批量写入特征，单事务。"""
    if not features:
        return
    now = time.time()
    rows = [
        (f.path, f.width, f.height, f.megapixels, f.entropy, f.edge_density,
         f.color_variance, f.unique_colors_ratio, int(f.is_grayscale),
         int(f.is_animated), f.frames, f.image_format, f.file_size,
         f.complexity_score, now)
        for f in features
    ]
    with conn:
        conn.executemany(
            """INSERT OR REPLACE INTO features
               (path, width, height, megapixels, entropy, edge_density,
                color_variance, unique_colors_ratio, is_grayscale, is_animated,
                frames, image_format, file_size, complexity_score, extracted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )


def insert_cbz_infos_batch(
    conn: sqlite3.Connection, infos: list[CbzInfo],
) -> None:
    """批量写入 CBZ 信息，单事务。"""
    if not infos:
        return
    now = time.time()
    rows = [
        (c.path, c.file_size, c.image_count, c.total_image_bytes,
         c.avg_page_bytes, c.png_ratio, c.webp_ratio,
         json.dumps(c.formats), now)
        for c in infos
    ]
    with conn:
        conn.executemany(
            """INSERT OR REPLACE INTO cbz_info
               (path, file_size, image_count, total_image_bytes, avg_page_bytes,
                png_ratio, webp_ratio, formats, extracted_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )


def insert_samples_batch(
    conn: sqlite3.Connection,
    records: list[tuple[str, str, str, str]],
) -> None:
    """批量写入样本记录，单事务。"""
    if not records:
        return
    now = time.time()
    rows = [(src, dst, cat, tier, now) for src, dst, cat, tier in records]
    with conn:
        conn.executemany(
            "INSERT INTO samples (src_path, dst_path, category, tier, copied_at) "
            "VALUES (?,?,?,?,?)",
            rows,
        )


# ── 分层抽样 ──────────────────────────────────────────────────────────

def stratified_sample_by_complexity(
    features: list[ImageFeatures],
    n: int,
    n_tiers: int = 5,
) -> dict[str, list[ImageFeatures]]:
    """按复杂度分位点分层抽样。"""
    animated = [f for f in features if f.is_animated]
    static = [f for f in features if not f.is_animated]

    if not static:
        return {"animated": animated[:n]}

    scores = np.array([f.complexity_score for f in static])
    percentiles = np.linspace(0, 100, n_tiers + 1)
    bins = np.percentile(scores, percentiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.array([scores.min() - 0.1, scores.max() + 0.1])

    tier_names = ["ultra_low", "low", "medium", "high", "ultra_high"]
    actual_tiers = min(len(bins) - 1, len(tier_names))
    tier_names = tier_names[:actual_tiers]

    tier_files: dict[str, list[ImageFeatures]] = {t: [] for t in tier_names}
    for f, score in zip(static, scores):
        placed = False
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                if lo <= score <= hi:
                    tier_files[tier_names[i]].append(f)
                    placed = True
                    break
            else:
                if lo <= score < hi:
                    tier_files[tier_names[i]].append(f)
                    placed = True
                    break
        if not placed:
            tier_files[tier_names[-1]].append(f)

    per_tier = max(1, n // max(len(tier_files), 1))
    result: dict[str, list[ImageFeatures]] = {}

    for tier_name, tier_list in tier_files.items():
        count = min(per_tier, len(tier_list))
        result[tier_name] = random.sample(tier_list, count) if count > 0 else []

    if animated:
        animated_count = min(len(animated), max(2, n // 10))
        result["animated"] = random.sample(animated, animated_count)

    return result


# ── 报告 ──────────────────────────────────────────────────────────────

def format_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(Path(entry.path))
    except (OSError, PermissionError):
        pass
    return total


# ── 主流程 ────────────────────────────────────────────────────────────

def process_images(
    entries: list[FileEntry],
    conn: sqlite3.Connection,
    label: str,
) -> list[ImageFeatures]:
    """并发特征提取，批量写 DB。

    ThreadPoolExecutor 掩盖 NFS I/O 延迟：
    一个线程等待网络时，其他线程执行 numpy 计算。
    """
    features: list[ImageFeatures] = []
    total = len(entries)

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
        futures = {
            executor.submit(extract_features, Path(e.path), e.size): e
            for e in entries
        }
        for i, future in enumerate(as_completed(futures)):
            feat = future.result()
            if feat:
                features.append(feat)
            if (i + 1) % 100 == 0 or i == total - 1:
                print(f"  [{label}] {i + 1}/{total} 特征提取完成")

    # 批量写入，单事务
    insert_features_batch(conn, features)
    return features


def process_cbz_files(
    entries: list[FileEntry],
    conn: sqlite3.Connection,
    cbz_extracted_dir: Path,
) -> list[CbzInfo]:
    """CBZ 分析与安全抽取（CBZ 数量少，串行即可）。"""
    infos: list[CbzInfo] = []

    for e in entries:
        info = analyze_cbz(Path(e.path), e.size)
        if info:
            infos.append(info)

            fmt_parts = ", ".join(
                f"{ext} {cnt/info.image_count:.0%}"
                for ext, cnt in sorted(info.formats.items())
            )
            print(f"  抽取 {Path(e.path).name} ({info.image_count} 页: {fmt_parts})")
            extracted = extract_cbz_pages(Path(e.path), cbz_extracted_dir)
            print(f"    抽取了 {len(extracted)} 页")

    insert_cbz_infos_batch(conn, infos)
    return infos


def copy_samples(
    tiered: dict[str, list[ImageFeatures]],
    raw_dir: Path,
    quality_dirs: dict[str, Path],
    bw_dir: Optional[Path],
    color_dir: Optional[Path],
    conn: sqlite3.Connection,
    prefix: str,
) -> tuple[int, int, int]:
    """复制样本到目标目录结构，批量写 sample 记录。"""
    copied = 0
    bw_count = 0
    color_count = 0
    records: list[tuple[str, str, str, str]] = []

    for tier_name, samples in tiered.items():
        for feat in samples:
            src = Path(feat.path)
            dst = link_or_copy(src, raw_dir / f"{prefix}_{src.name}")
            if not dst:
                continue
            copied += 1
            records.append((feat.path, str(dst), prefix, tier_name))

            if tier_name in quality_dirs:
                link_or_copy(dst, quality_dirs[tier_name] / dst.name)

            if bw_dir and color_dir:
                if feat.is_grayscale:
                    link_or_copy(dst, bw_dir / dst.name)
                    bw_count += 1
                else:
                    link_or_copy(dst, color_dir / dst.name)
                    color_count += 1

    insert_samples_batch(conn, records)
    return copied, bw_count, color_count


def export_json_manifest(
    features: list[ImageFeatures],
    cbz_infos: list[CbzInfo],
) -> None:
    """导出 JSON manifest 供 benchmark 脚本驱动。"""
    data = {
        "images": [
            {
                "path": f.path,
                "width": f.width,
                "height": f.height,
                "megapixels": round(f.megapixels, 2),
                "entropy": round(f.entropy, 4),
                "edge_density": round(f.edge_density, 4),
                "color_variance": round(f.color_variance, 4),
                "unique_colors_ratio": round(f.unique_colors_ratio, 4),
                "is_grayscale": f.is_grayscale,
                "is_animated": f.is_animated,
                "frames": f.frames,
                "format": f.image_format,
                "file_size": f.file_size,
                "complexity_score": round(f.complexity_score, 4),
            }
            for f in features
        ],
        "cbz": [
            {
                "path": c.path,
                "file_size": c.file_size,
                "image_count": c.image_count,
                "avg_page_bytes": round(c.avg_page_bytes, 1),
                "png_ratio": round(c.png_ratio, 4),
                "webp_ratio": round(c.webp_ratio, 4),
                "formats": c.formats,
            }
            for c in cbz_infos
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    global PREFER_HARDLINK

    print("=" * 60)
    print("图片压缩测试样本抽样工具 v2.1")
    print("策略: 图像复杂度特征 → 分层抽样 → manifest 驱动")
    print("=" * 60)

    # ── 初始化 + st_dev 预检 ──
    if SAMPLE_DIR.exists():
        print(f"\n清理旧样本目录: {SAMPLE_DIR}")
        shutil.rmtree(SAMPLE_DIR)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    _check_hardlink_viability()

    conn = init_manifest_db(MANIFEST_DB)

    # 目录结构
    manga_raw = SAMPLE_DIR / "manga" / "raw"
    manga_bw = SAMPLE_DIR / "manga" / "black_and_white"
    manga_color = SAMPLE_DIR / "manga" / "color"
    manga_quality = {
        tier: SAMPLE_DIR / "manga" / "quality" / tier
        for tier in ["ultra_low", "low", "medium", "high", "ultra_high", "animated"]
    }
    photo_raw = SAMPLE_DIR / "photos" / "raw"
    photo_quality = {
        tier: SAMPLE_DIR / "photos" / "quality" / tier
        for tier in ["ultra_low", "low", "medium", "high", "ultra_high", "animated"]
    }
    cbz_raw = SAMPLE_DIR / "cbz" / "raw"
    cbz_extracted = SAMPLE_DIR / "cbz" / "extracted"

    all_dirs = [
        manga_raw, manga_bw, manga_color, photo_raw, cbz_raw, cbz_extracted,
    ] + list(manga_quality.values()) + list(photo_quality.values())
    for d in all_dirs:
        d.mkdir(parents=True, exist_ok=True)

    all_features: list[ImageFeatures] = []
    all_cbz_infos: list[CbzInfo] = []

    # ── Phase 1-2: 漫画扫描 + 并发特征 ──
    print("\n[1/5] 扫描与特征提取: 漫画")
    manga_entries = scan_directory(MANGA_DIR, IMAGE_EXTENSIONS, max_files=3000)
    print(f"  找到 {len(manga_entries)} 个漫画文件")
    manga_features = process_images(manga_entries, conn, "漫画")
    all_features.extend(manga_features)

    # ── Phase 1-2: 照片扫描 + 并发特征 ──
    print("\n[2/5] 扫描与特征提取: 照片")
    photo_entries = scan_directory(PHOTO_DIR, IMAGE_EXTENSIONS, max_files=3000)
    print(f"  找到 {len(photo_entries)} 个照片文件")
    photo_features = process_images(photo_entries, conn, "照片")
    all_features.extend(photo_features)

    # ── Phase 3: 分层抽样 ──
    print("\n[3/5] 复杂度分层抽样")
    print(f"  漫画特征: {len(manga_features)}, 照片特征: {len(photo_features)}")

    manga_bw_feats = [f for f in manga_features if f.is_grayscale]
    manga_color_feats = [f for f in manga_features if not f.is_grayscale]
    bw_tiers = stratified_sample_by_complexity(manga_bw_feats, MANGA_SAMPLES // 2)
    color_tiers = stratified_sample_by_complexity(manga_color_feats, MANGA_SAMPLES // 2)

    print("  漫画复杂度分布（黑白）:")
    for t, fs in bw_tiers.items():
        if fs:
            scores = [round(f.complexity_score, 2) for f in fs]
            print(f"    {t}: {len(fs)} 样本, 复杂度 {min(scores):.2f}~{max(scores):.2f}")
    print("  漫画复杂度分布（彩色）:")
    for t, fs in color_tiers.items():
        if fs:
            scores = [round(f.complexity_score, 2) for f in fs]
            print(f"    {t}: {len(fs)} 样本, 复杂度 {min(scores):.2f}~{max(scores):.2f}")

    photo_tiers = stratified_sample_by_complexity(photo_features, PHOTO_SAMPLES)
    print("  照片复杂度分布:")
    for t, fs in photo_tiers.items():
        if fs:
            scores = [round(f.complexity_score, 2) for f in fs]
            print(f"    {t}: {len(fs)} 样本, 复杂度 {min(scores):.2f}~{max(scores):.2f}")

    # ── Phase 4: 复制样本 ──
    print("\n[4/5] 复制样本")

    bw_copied, _, _ = copy_samples(
        bw_tiers, manga_raw, manga_quality,
        manga_bw, manga_color, conn, "manga_bw",
    )
    color_copied, _, _ = copy_samples(
        color_tiers, manga_raw, manga_quality,
        manga_bw, manga_color, conn, "manga_color",
    )
    print(f"  漫画: 复制 {bw_copied + color_copied} 个 (黑白 {bw_copied}, 彩色 {color_copied})")

    photo_copied, _, _ = copy_samples(
        photo_tiers, photo_raw, photo_quality,
        None, None, conn, "photo",
    )
    print(f"  照片: 复制 {photo_copied} 个")

    # ── Phase 5: CBZ ──
    print("\n[5/5] CBZ 分析")
    cbz_entries = scan_directory(CBZ_DIR, CBZ_EXTENSIONS, max_files=100)
    print(f"  找到 {len(cbz_entries)} 个 CBZ 文件")

    if cbz_entries:
        cbz_sorted = sorted(cbz_entries, key=lambda e: e.size, reverse=True)
        cbz_selected = cbz_sorted[:CBZ_SAMPLES]

        cbz_infos = process_cbz_files(cbz_selected, conn, cbz_extracted)
        all_cbz_infos.extend(cbz_infos)

        for e in cbz_selected:
            link_or_copy(Path(e.path), cbz_raw / Path(e.path).name)

    # ── 导出 manifest ──
    export_json_manifest(all_features, all_cbz_infos)
    print(f"\n  Manifest → {MANIFEST_JSON}")
    print(f"  Manifest DB → {MANIFEST_DB}")

    # ── 统计 ──
    print("\n" + "=" * 60)
    print("样本统计")
    print("=" * 60)

    print("\n漫画样本:")
    print(f"  原始样本: {_count_files(manga_raw)} 文件")
    print(f"  黑白漫画: {_count_files(manga_bw)} 文件")
    print(f"  彩色漫画: {_count_files(manga_color)} 文件")
    print(f"  总大小: {format_size(get_dir_size(manga_raw))}")
    print("  画质分布:")
    for tier in ["ultra_low", "low", "medium", "high", "ultra_high", "animated"]:
        d = manga_quality[tier]
        count = _count_files(d)
        if count:
            print(f"    {tier}: {count} 文件, {format_size(get_dir_size(d))}")

    print("\n照片样本:")
    print(f"  原始样本: {_count_files(photo_raw)} 文件")
    print(f"  总大小: {format_size(get_dir_size(photo_raw))}")
    print("  画质分布:")
    for tier in ["ultra_low", "low", "medium", "high", "ultra_high", "animated"]:
        d = photo_quality[tier]
        count = _count_files(d)
        if count:
            print(f"    {tier}: {count} 文件, {format_size(get_dir_size(d))}")

    print("\nCBZ 样本:")
    print(f"  CBZ 文件: {_count_files(cbz_raw)}")
    print(f"  解压内容: {_count_files(cbz_extracted)} 文件")
    print(f"  CBZ 总大小: {format_size(get_dir_size(cbz_raw))}")

    print("\n" + "=" * 60)
    print("完成！")
    print("下一步:")
    print("  1. 查看 manifest: cat samples/manifest.json")
    print("  2. 运行压缩测试: python scripts/benchmark_compression.py")
    print("  3. 根据测试结果调整压缩参数")
    print("\n目录结构:")
    print("""
samples/
├── manifest.json         # 完整特征清单（驱动 benchmark）
├── manifest.db           # SQLite checkpoint（支持 resume）
├── manga/
│   ├── raw/              # 原始样本
│   ├── black_and_white/  # 黑白漫画
│   ├── color/            # 彩色漫画
│   └── quality/          # 复杂度分层
│       ├── ultra_low/
│       ├── low/
│       ├── medium/
│       ├── high/
│       ├── ultra_high/
│       └── animated/     # 动图
├── photos/
│   ├── raw/
│   └── quality/
│       └── (同上)
└── cbz/
    ├── raw/              # CBZ 归档
    └── extracted/        # 代表性页面（前 N/2 + 随机 N/2）
""")


def _check_hardlink_viability() -> None:
    """st_dev 预检：源和目标跨文件系统 → 全局禁用 hardlink。

    避免每次 link_or_copy 都触发 OSError（NFS/Btrfs/ZFS subvolume 场景）。
    """
    global PREFER_HARDLINK
    src_dirs = [MANGA_DIR, PHOTO_DIR, CBZ_DIR]
    try:
        dst_dev = os.stat(SAMPLE_DIR).st_dev
        for src_dir in src_dirs:
            if src_dir.exists():
                src_dev = os.stat(src_dir).st_dev
                if src_dev != dst_dev:
                    PREFER_HARDLINK = False
                    print(f"  st_dev 不同 ({src_dir} → {SAMPLE_DIR})，全局禁用 hardlink")
                    return
    except OSError:
        # 源目录不存在时不 panic，脚本后续会提示 "找到 0 个文件"
        pass


def _count_files(path: Path) -> int:
    """统计目录下文件数（不含子目录递归）。"""
    count = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                count += 1
    except (OSError, PermissionError):
        pass
    return count


if __name__ == "__main__":
    main()
