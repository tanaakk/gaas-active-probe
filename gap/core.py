"""
GAP コアオーケストレーター

4モジュールを統合し、階層的質量操作のフルサイクルを実行する。
待機の定式化・負圧の優先・大数操作の線形性を厳守する。
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from gap.constants import Layer, LAYER_CONFIGS, get_layer_for_mass
from gap.frequency_emitter import FrequencyEmitter, ResonanceResult
from gap.impedance_finalizer import ImpedanceFinalizer, FinalizedContour
from gap.vacuum_path_finder import VacuumPathFinder, VacuumPoint
from gap.transition_executor import TransitionExecutor, PositionShiftResult, ExecutionState


class ProbeState(Enum):
    """プローブ全体の状態"""

    IDLE = "idle"  # 特異点未出現、演算停止・観測精度向上のみ
    CALIBRATING = "calibrating"  # 周波数同調・神経確立中
    VACUUM_DETECTED = "vacuum_detected"  # 負圧ポイント発見
    EXECUTING = "executing"  # ゼロコスト通過実行中
    COMPLETED = "completed"  # 完了


@dataclass
class OperationResult:
    """操作の全体結果"""

    state: ProbeState
    layer: Layer
    resonance: Optional[ResonanceResult]
    contours: List[FinalizedContour]
    vacuum_points: List[VacuumPoint]
    transition_result: Optional[PositionShiftResult]
    idle_reason: Optional[str]  # IDLE 時の理由


class GeometricActiveProbe:
    """
    階層的質量操作システムのメインオーケストレーター。

    対象を「特定の周波数を持つ球体（質点）」として定義し、
    周波数照射 → 最小神経確立 → 負圧ポイント発見 → ゼロコスト通過
    の順で実行する。

    重要ルール:
    - 待機の定式化: 特異点が出現していない場合は Idle
    - 負圧の優先: 吸い込まれる状態が確立された時のみ EXECUTE
    - 大数操作の線形性: 全スケールで同一アルゴリズム
    """

    def __init__(
        self,
        singularity_threshold: float = 1e-4,
        resonance_precision: float = 1e-6,
    ):
        self.frequency_emitter = FrequencyEmitter(precision=resonance_precision)
        self.impedance_finalizer = ImpedanceFinalizer(
            master_calibration_precision=resonance_precision
        )
        self.vacuum_path_finder = VacuumPathFinder(
            singularity_threshold=singularity_threshold
        )
        self.transition_executor = TransitionExecutor()

    def operate(
        self,
        target_mass: float,
        layer: Optional[Layer] = None,
        additional_contours: Optional[List[Tuple[float, Layer]]] = None,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> OperationResult:
        """
        階層的質量操作のフルサイクルを実行する。

        Args:
            target_mass: 対象の質量（スケール）
            layer: 階層（None の場合は質量から自動決定）
            additional_contours: 干渉用の追加質量群 [(mass, layer), ...]
            initial_position: 初期座標

        Returns:
            操作結果
        """
        # 階層の決定
        if layer is None:
            layer = get_layer_for_mass(target_mass)
            if layer is None:
                return OperationResult(
                    state=ProbeState.IDLE,
                    layer=Layer.MICRO,
                    resonance=None,
                    contours=[],
                    vacuum_points=[],
                    transition_result=None,
                    idle_reason="質量が全階層の範囲外",
                )
        else:
            config = LAYER_CONFIGS[layer]
            if not config.contains(target_mass):
                return OperationResult(
                    state=ProbeState.IDLE,
                    layer=layer,
                    resonance=None,
                    contours=[],
                    vacuum_points=[],
                    transition_result=None,
                    idle_reason="質量が指定階層の範囲外",
                )

        # 1. 周波数照射・同調確認
        pulse = self.frequency_emitter.emit_pulse(target_mass, layer)
        resonance = self.frequency_emitter.check_resonance(
            target_mass, layer, pulse
        )

        if not resonance.achieved:
            return OperationResult(
                state=ProbeState.IDLE,
                layer=layer,
                resonance=resonance,
                contours=[],
                vacuum_points=[],
                transition_result=None,
                idle_reason="同調未達成、観測精度向上待ち",
            )

        # 2. 最小神経確立
        main_contour = self.impedance_finalizer.establish_contour(
            target_mass, layer
        )
        contours = [main_contour]

        if additional_contours:
            for add_mass, add_layer in additional_contours:
                add_c = self.impedance_finalizer.establish_contour(
                    add_mass, add_layer
                )
                contours.append(add_c)

        # 3. 負圧ポイント発見
        vacuum_points = self.vacuum_path_finder.find_vacuum_points(contours)

        if not vacuum_points:
            return OperationResult(
                state=ProbeState.IDLE,
                layer=layer,
                resonance=resonance,
                contours=contours,
                vacuum_points=[],
                transition_result=None,
                idle_reason="特異点未出現、演算停止・観測継続",
            )

        # 有効な特異点を選択（吸い込まれる状態が確立されたもの）
        valid_vacuums = [
            vp for vp in vacuum_points
            if self.vacuum_path_finder.validate_singularity(vp)
        ]

        if not valid_vacuums:
            return OperationResult(
                state=ProbeState.IDLE,
                layer=layer,
                resonance=resonance,
                contours=contours,
                vacuum_points=vacuum_points,
                transition_result=None,
                idle_reason="負圧未確立、吸い込まれる状態待ち",
            )

        # 4. ゼロコスト通過（負圧の優先: 確立された時のみ EXECUTE）
        best_vacuum = min(
            valid_vacuums,
            key=lambda v: v.potential_gradient,
        )
        transition_result = self.transition_executor.execute(
            initial_position,
            best_vacuum,
            target_mass,
        )

        return OperationResult(
            state=ProbeState.COMPLETED,
            layer=layer,
            resonance=resonance,
            contours=contours,
            vacuum_points=vacuum_points,
            transition_result=transition_result,
            idle_reason=None,
        )

    def observe_only(
        self,
        target_mass: float,
        layer: Optional[Layer] = None,
    ) -> OperationResult:
        """
        観測のみ実行（Idle State の観測精度向上モード）。

        特異点が出現していない場合の待機中に、観測精度を高めるために使用。

        Args:
            target_mass: 対象質量
            layer: 階層

        Returns:
            観測結果（EXECUTE は行わない）
        """
        if layer is None:
            layer = get_layer_for_mass(target_mass) or Layer.MICRO

        pulse = self.frequency_emitter.emit_pulse(target_mass, layer)
        resonance = self.frequency_emitter.check_resonance(
            target_mass, layer, pulse
        )
        contour = self.impedance_finalizer.establish_contour(
            target_mass, layer
        )

        return OperationResult(
            state=ProbeState.IDLE,
            layer=layer,
            resonance=resonance,
            contours=[contour],
            vacuum_points=[],
            transition_result=None,
            idle_reason="観測モード、実行なし",
        )
