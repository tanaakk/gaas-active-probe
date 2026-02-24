"""
ImpedanceFinalizer — 最小神経確立モジュール

外部の最高精度探針（マスター校正器）を用い、対象の輪郭を物理的に確定。
個体差を不変の幾何形状（ラスト）へ固定する。
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from gap.constants import Layer, LayerConfig, LAYER_CONFIGS


@dataclass
class FinalizedContour:
    """確定された幾何形状（ラスト）"""

    centroid: Tuple[float, float, float]
    radius: float
    impedance: float  # 最小神経確立後のインピーダンス
    layer: Layer
    is_stable: bool  # 不変の幾何形状として固定されたか


class ImpedanceFinalizer:
    """
    最小神経確立器。

    対象の輪郭を物理的に確定し、個体差を不変の幾何形状へ固定する。
    マスター校正器（最高精度探針）の概念を抽象化して実装。
    """

    def __init__(self, master_calibration_precision: float = 1e-6):
        """
        Args:
            master_calibration_precision: マスター校正器の精度
        """
        self.precision = master_calibration_precision

    def establish_contour(
        self,
        target_mass: float,
        layer: Layer,
        raw_boundary: Optional[np.ndarray] = None,
    ) -> FinalizedContour:
        """
        対象の輪郭を物理的に確定し、不変の幾何形状へ固定する。

        Args:
            target_mass: 対象質量
            layer: 階層
            raw_boundary: 生の境界データ（オプション）

        Returns:
            確定された輪郭（ラスト）
        """
        config = LAYER_CONFIGS[layer]

        if raw_boundary is not None and len(raw_boundary) > 0:
            mean_pt = np.mean(raw_boundary, axis=0)
            centroid = tuple(
                (mean_pt[0], mean_pt[1], mean_pt[2])
                if len(mean_pt) >= 3
                else (*mean_pt, 0.0)
            )
            diff = raw_boundary - np.array(centroid)[: raw_boundary.shape[1]]
            radius = np.max(np.linalg.norm(diff, axis=1))
        else:
            # 質量からスケール不変の球体パラメータを導出
            log_mass = np.log10(target_mass)
            radius = 10 ** (log_mass / 6 - 2)  # スケール不変の半径公式
            centroid = (0.0, 0.0, 0.0)

        # インピーダンス: 質量/周波数比の逆数（最小神経確立）
        impedance = target_mass / (config.base_frequency * radius)
        impedance = max(impedance, self.precision)

        # 安定性: 階層内で質量が中央に近いほど安定
        mass_ratio = (target_mass - config.mass_min) / (
            config.mass_max - config.mass_min + 1e-30
        )
        is_stable = 0.2 < mass_ratio < 0.8

        return FinalizedContour(
            centroid=centroid,
            radius=radius,
            impedance=impedance,
            layer=layer,
            is_stable=is_stable,
        )

    def calibrate_geometry(
        self,
        contour: FinalizedContour,
        reference_mass: float,
    ) -> FinalizedContour:
        """
        参照質量を用いて幾何形状を校正する。

        Args:
            contour: 確定済み輪郭
            reference_mass: 校正用参照質量

        Returns:
            校正後の輪郭
        """
        scale = np.sqrt(reference_mass / (contour.radius**3 + 1e-30))
        new_radius = contour.radius * scale
        new_impedance = contour.impedance / (scale + 1e-30)

        return FinalizedContour(
            centroid=contour.centroid,
            radius=new_radius,
            impedance=new_impedance,
            layer=contour.layer,
            is_stable=contour.is_stable,
        )
