"""灰度图像检测器 - 区分真正彩漫与单色偏色黑白漫。

解决了传统方法无法识别的 "泛黄/泛绿/泛红黑白扫描件" 问题。

算法核心:
    1. 明度通道过滤 (V < 30): 剔除极暗噪点，避免暗部饱和度的计算失真
       - HSV 明度极低时 (V→0)，饱和度计算公式 S = (MAX-MIN)/MAX 中分母趋零，
         导致暗部噪点的饱和度被放大到极高值且色相随机，严重干扰色相集中度分析
    2. 饱和度比例过滤: 对有效明度区间内的像素进行高饱和度比例判断
    3. 色相集中度分析: 对高饱和度有效像素进行色相直方图分析
       - 真正彩漫: 色相分散 (皮肤橙红 + 天空蓝 + 草地绿)
       - 偏色黑白漫: 色相集中在单一窄色系区间

环形色相处理:
    HSV 色相 0° 和 360° 都是红色，直接计算会崩溃。
    通过 padded_hist 首尾拼接，完美解决红光泛红扫描件识别。

参考文献:
    - HSV 色彩空间: https://docs.opencv.org/4.x/de/d05/tutorial_py_colorspaces.html
    - 色相直方图: np.histogram 用于色相分布分析
"""

from __future__ import annotations

import numpy as np
import pyvips


# ── 配置参数 ──────────────────────────────────────────────────────────

# 明度阈值: 剔除 V < 30 的暗部像素 (暗部 HSV 饱和度计算不准确)
DARK_PIXEL_VALUE_THRESHOLD = 30

# 高饱和度阈值: 过滤纸张泛黄/JPEG 噪点 (HSV 饱和度 0-255)
COLOR_PIXEL_SATURATION_THRESHOLD = 30

# 彩色像素占比容差: < 3% → 判为黑白 (容忍小面积 Logo/水印)
COLOR_RATIO_TOLERANCE = 0.03

# 色相直方图分桶数: 12 桶 × 30° = 360° 全色相覆盖
HUE_HISTOGRAM_BINS = 12

# 色相集中度阈值: 单一色系 > 90% → 判为偏色黑白漫
HUE_CONCENTRATION_THRESHOLD = 0.90

# 相邻桶跨度: 60° (2 个 30° 桶) 覆盖同一色系光影变化
HUE_SPAN_BINS = 2


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


def _vips_band_to_numpy(band: pyvips.Image) -> np.ndarray:
    """pyvips 波段 → numpy 1D 数组。"""
    mem = band.write_to_memory()
    dtype = _numpy_dtype(band)
    return np.frombuffer(mem, dtype=dtype).flatten()


def _analyze_hue_concentration(
    hue_pixels: np.ndarray,
    n_bins: int = HUE_HISTOGRAM_BINS,
    span_bins: int = HUE_SPAN_BINS,
    threshold: float = HUE_CONCENTRATION_THRESHOLD,
) -> tuple[float, float]:
    """分析高饱和度像素的色相集中度。

    Args:
        hue_pixels: 高饱和度像素的色相值 (0-360 度)
        n_bins: 色相直方图分桶数 (默认 12)
        span_bins: 相邻桶跨度 (默认 2，即 60° 范围)
        threshold: 集中度阈值 (默认 0.90)

    Returns:
        (max_concentration, total_pixels):
            max_concentration: 最大集中度 (0-1)
            total_pixels: 总像素数 (用于判断是否有足够样本)
    """
    if hue_pixels.size == 0:
        return 0.0, 0.0

    # 构建色相直方图 (0-360 度)
    hist, _ = np.histogram(hue_pixels, bins=n_bins, range=(0, 360))

    # 环形处理: 首尾拼接解决红色跨越 0°/360° 的问题
    # 例如: span_bins=2 时，检查 hist[i] + hist[i+1] 的最大值
    padded_hist = np.concatenate((hist, hist[:span_bins]))

    # 滑动窗口求和: 计算任意连续 span_bins 个桶的像素总和
    if span_bins > 1:
        kernel = np.ones(span_bins, dtype=np.int32)
        window_sums = np.convolve(padded_hist, kernel, mode='valid')
    else:
        window_sums = padded_hist

    # 最大集中度 = 最大窗口和 / 总像素
    total_pixels = int(hist.sum())
    max_window_sum = int(window_sums.max()) if window_sums.size > 0 else 0
    concentration = max_window_sum / total_pixels if total_pixels > 0 else 0.0

    return concentration, total_pixels


def detect_grayscale(
    sat_np: np.ndarray,
    hue_np: np.ndarray,
    val_np: np.ndarray | None = None,
    sat_threshold: int = COLOR_PIXEL_SATURATION_THRESHOLD,
    val_threshold: int = DARK_PIXEL_VALUE_THRESHOLD,
    color_tolerance: float = COLOR_RATIO_TOLERANCE,
    hue_threshold: float = HUE_CONCENTRATION_THRESHOLD,
) -> tuple[bool, dict]:
    """检测图像是否为灰度 (增强版: 明度过滤 + 色相集中度分析)。

    解决传统方法无法识别的单色偏色问题:
    - 泛黄扫描件: 整图饱和度 > 30，但色相集中在黄色区间
    - 泛绿滤镜: 整图饱和度 > 30，但色相集中在绿色区间
    - 真正彩漫: 色相分散在多个色系

    Bug 修复 - 暗部噪点干扰:
        纯黑区域的暗部噪点因 HSV 饱和度公式 (MAX-MIN)/MAX 在 V→0 时
        分母趋零，饱和度被错误放大且色相随机。引入 V < 30 过滤后，
        这些无效像素被排除，色相集中度分析恢复准确。

    Args:
        sat_np: 饱和度通道的 numpy 数组
        hue_np: 色相通道的 numpy 数组 (0-360)
        val_np: 明度通道的 numpy 数组 (0-255), 用于过滤暗部噪点
        sat_threshold: 饱和度阈值 (默认 30)
        val_threshold: 明度阈值，V < 此值的像素被排除 (默认 30)
        color_tolerance: 彩色像素占比容差 (默认 0.03)
        hue_threshold: 色相集中度阈值 (默认 0.90)

    Returns:
        (is_grayscale, metadata):
            is_grayscale: True 表示灰度，False 表示彩色
            metadata: 包含 color_ratio, hue_concentration 等诊断信息
    """
    total_pixels = sat_np.size
    metadata = {
        "method": "hue_concentration",
        "color_ratio": 0.0,
        "color_pixels": 0,
        "total_pixels": total_pixels,
        "hue_concentration": 0.0,
        "is_single_tint": False,
        "dark_pixels_excluded": 0,
    }

    if total_pixels == 0:
        return True, metadata

    # Step 0: 明度通道过滤 — 剔除 V < val_threshold 的暗部噪点
    # 暗部像素的 HSV 饱和度计算不准确，必须排除
    if val_np is not None:
        valid_mask = val_np >= val_threshold
        dark_excluded = int(np.count_nonzero(~valid_mask))
        metadata["dark_pixels_excluded"] = dark_excluded
        sat_np = sat_np[valid_mask]
        hue_np = hue_np[valid_mask]

    if sat_np.size == 0:
        return True, metadata

    # Step 1: 过滤出高饱和度彩色像素
    color_mask = sat_np > sat_threshold
    color_pixels = int(np.count_nonzero(color_mask))
    color_ratio = float(color_pixels / sat_np.size)

    metadata["color_ratio"] = color_ratio
    metadata["color_pixels"] = color_pixels

    # 情况 A: 几乎没有彩色像素 → 纯黑白
    if color_ratio < color_tolerance:
        return True, metadata

    # Step 2: 分析色相集中度 (区分真正彩漫 vs 偏色黑白漫)
    valid_hues = hue_np[color_mask]

    concentration, _ = _analyze_hue_concentration(valid_hues)
    metadata["hue_concentration"] = concentration
    metadata["is_single_tint"] = concentration > hue_threshold

    # 情况 B: 高饱和度像素色相高度集中 → 偏色黑白漫
    if concentration > hue_threshold:
        return True, metadata

    # 情况 C: 色相分散 → 真正彩漫
    return False, metadata


def detect_grayscale_simple(
    sat_np: np.ndarray,
    hue_np: np.ndarray,
    val_np: np.ndarray | None = None,
) -> bool:
    """简化版灰度检测 (仅返回结果，无元数据)。

    适用于不需要诊断信息的场景。
    """
    is_grayscale, _ = detect_grayscale(sat_np, hue_np, val_np=val_np)
    return is_grayscale


def get_detection_config() -> dict:
    """返回当前检测配置 (便于调试/日志)。"""
    return {
        "val_threshold": DARK_PIXEL_VALUE_THRESHOLD,
        "sat_threshold": COLOR_PIXEL_SATURATION_THRESHOLD,
        "color_tolerance": COLOR_RATIO_TOLERANCE,
        "hue_threshold": HUE_CONCENTRATION_THRESHOLD,
        "hue_bins": HUE_HISTOGRAM_BINS,
        "hue_span_bins": HUE_SPAN_BINS,
    }
