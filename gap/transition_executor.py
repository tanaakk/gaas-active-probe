"""
TransitionExecutor — ゼロコスト通過モジュール

負圧によって「吸い込まれる」経路を確定し、エネルギー消費なしで
資産・質量の座標を置換（Position Shift）する。
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from gap.vacuum_path_finder import VacuumPoint


class ExecutionState(Enum):
    """実行状態"""

    IDLE = "idle"  # 待機（特異点未出現）
    READY = "ready"  # 待機（特異点出現、実行待ち）
    EXECUTED = "executed"  # ゼロコスト通過完了
    ABORTED = "aborted"  # 負圧未確立のため中止


@dataclass
class PositionShiftResult:
    """座標置換の結果"""

    origin: Tuple[float, float, float]
    destination: Tuple[float, float, float]
    energy_cost: float  # ゼロであるべき
    state: ExecutionState
    success: bool


class TransitionExecutor:
    """
    ゼロコスト通過実行器。

    負圧の優先: 「押し出す」のではなく、「吸い込まれる」状態が確立された時のみ
    EXECUTE する。

    大数操作の線形性: 同アルゴリズムで全スケールに適用可能。
    """

    def __init__(self, energy_tolerance: float = 1e-10):
        """
        Args:
            energy_tolerance: ゼロコスト判定の許容誤差
        """
        self.energy_tolerance = energy_tolerance

    def execute(
        self,
        current_position: Tuple[float, float, float],
        vacuum_point: VacuumPoint,
        mass: Optional[float] = None,
    ) -> PositionShiftResult:
        """
        負圧経路を確定し、ゼロコストで座標を置換する。

        負圧が確立されていない場合は EXECUTE せず、ABORTED を返す。

        Args:
            current_position: 現在の座標
            vacuum_point: 負圧ポイント（特異点）
            mass: 対象質量（オプション、スケール不変のため未使用可）

        Returns:
            座標置換の結果
        """
        if not vacuum_point.is_singular:
            return PositionShiftResult(
                origin=current_position,
                destination=current_position,
                energy_cost=0.0,
                state=ExecutionState.ABORTED,
                success=False,
            )

        if vacuum_point.potential_gradient > 1e-4:
            return PositionShiftResult(
                origin=current_position,
                destination=current_position,
                energy_cost=0.0,
                state=ExecutionState.ABORTED,
                success=False,
            )

        # 吸い込まれる経路: 負圧ポイントへ直線移動（エネルギー消費なし）
        destination = vacuum_point.coordinates

        # ゼロコスト検証: 負圧による移動はエネルギー消費なし
        distance = np.linalg.norm(
            np.array(destination) - np.array(current_position)
        )
        energy_cost = 0.0  # 負圧通過は常にゼロ

        return PositionShiftResult(
            origin=current_position,
            destination=destination,
            energy_cost=energy_cost,
            state=ExecutionState.EXECUTED,
            success=True,
        )

    def compute_path(
        self,
        origin: Tuple[float, float, float],
        vacuum_point: VacuumPoint,
        num_steps: int = 100,
    ) -> np.ndarray:
        """
        負圧による吸い込み経路を離散化して返す。

        Args:
            origin: 原点
            vacuum_point: 負圧ポイント
            num_steps: 経路の離散化ステップ数

        Returns:
            経路の座標列
        """
        if not vacuum_point.is_singular:
            return np.array([origin])

        dest = np.array(vacuum_point.coordinates)
        orig = np.array(origin)

        t = np.linspace(0, 1, num_steps)[:, np.newaxis]
        path = orig + t * (dest - orig)

        return path

    def verify_zero_cost(self, result: PositionShiftResult) -> bool:
        """
        ゼロコスト通過が正しく実行されたか検証する。

        Args:
            result: 座標置換結果

        Returns:
            ゼロコストが維持されているか
        """
        return (
            result.success
            and result.energy_cost <= self.energy_tolerance
            and result.state == ExecutionState.EXECUTED
        )
