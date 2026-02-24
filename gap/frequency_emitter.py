"""
FrequencyEmitter — 周波数照射モジュール

対象のスケールに応じたパルスを生成。小規模から順に周波数を確定させ、
同調（Resonance）を確認した後に次段階のスケールへ移行する。
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from gap.constants import Layer, LayerConfig, LAYER_CONFIGS


@dataclass
class ResonanceResult:
    """同調判定の結果"""

    achieved: bool
    resonance_factor: float
    target_frequency: float
    layer: Layer
    ready_for_next: bool  # 次段階へ移行可能か


class FrequencyEmitter:
    """
    周波数照射器。

    対象の質量スケールに応じたパルスを生成し、同調を確認する。
    大数操作の線形性に従い、スケールに依存しない同一アルゴリズムを使用する。
    """

    def __init__(self, precision: float = 1e-6):
        """
        Args:
            precision: 周波数同調の精度閾値
        """
        self.precision = precision

    def emit_pulse(
        self,
        target_mass: float,
        layer: Layer,
        phase: float = 0.0,
    ) -> np.ndarray:
        """
        対象のスケールに応じたパルスを生成する。

        Args:
            target_mass: 対象の質量（スケール）
            layer: 対象の階層
            phase: 位相オフセット

        Returns:
            パルス系列（時間軸上の振幅）
        """
        config = LAYER_CONFIGS[layer]
        freq = config.base_frequency * (1 - np.log10(target_mass) / 18)

        # パルス: スケール不変の正弦波列
        t = np.linspace(0, 1, 1000)
        pulse = np.sin(2 * np.pi * freq * t + phase) * np.exp(-t * 2)

        return pulse

    def check_resonance(
        self,
        target_mass: float,
        layer: Layer,
        emitted_pulse: np.ndarray,
        observed_response: Optional[np.ndarray] = None,
    ) -> ResonanceResult:
        """
        同調（Resonance）を確認する。

        観測応答が与えられない場合は、理論的に同調可能な状態を推定する。

        Args:
            target_mass: 対象質量
            layer: 階層
            emitted_pulse: 照射したパルス
            observed_response: 観測された応答（オプション）

        Returns:
            同調判定結果
        """
        config = LAYER_CONFIGS[layer]
        target_freq = config.base_frequency * (1 - np.log10(target_mass) / 18)

        if observed_response is not None:
            # 応答の相関から同調度を算出
            min_len = min(len(emitted_pulse), len(observed_response))
            corr = np.corrcoef(
                emitted_pulse[:min_len],
                observed_response[:min_len],
            )[0, 1]
            resonance_factor = float(np.clip(corr, 0, 1))
        else:
            # 理論的同調度（質量がスケール内に収まれば高く評価）
            mass_ratio = (target_mass - config.mass_min) / (
                config.mass_max - config.mass_min
            )
            resonance_factor = 1.0 - 0.1 * abs(mass_ratio - 0.5)

        achieved = resonance_factor >= config.resonance_threshold
        next_layer = Layer(layer.value + 1) if layer.value < 4 else layer
        ready_for_next = achieved and next_layer != layer

        return ResonanceResult(
            achieved=achieved,
            resonance_factor=resonance_factor,
            target_frequency=target_freq,
            layer=layer,
            ready_for_next=ready_for_next,
        )

    def calibrate_sequence(
        self,
        target_mass: float,
        start_layer: Layer = Layer.MICRO,
    ) -> list[Layer]:
        """
        小規模から順に周波数を確定させ、同調した階層を返す。

        大数操作の線形性: 同調ロジックはスケール不変。

        Args:
            target_mass: 対象質量
            start_layer: 開始階層

        Returns:
            同調が確認された階層のリスト
        """
        layers = list(Layer)
        start_idx = start_layer.value - 1
        resolved: list[Layer] = []

        for i in range(start_idx, len(layers)):
            layer = layers[i]
            config = LAYER_CONFIGS[layer]
            if not config.contains(target_mass):
                continue

            pulse = self.emit_pulse(target_mass, layer)
            result = self.check_resonance(target_mass, layer, pulse)

            if result.achieved:
                resolved.append(layer)
                if not result.ready_for_next:
                    break

        return resolved
