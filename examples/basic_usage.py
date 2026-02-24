#!/usr/bin/env python3
"""
GAP 基本使用例

階層的質量操作システムの基本的な呼び出し方法を示す。
"""

from gap import GeometricActiveProbe
from gap.constants import Layer


def main():
    probe = GeometricActiveProbe()

    # Layer 1 (Micro): 10^6 - 10^8 の質量
    result = probe.operate(target_mass=1e7, layer=Layer.MICRO)
    print(f"Layer MICRO (1e7): state={result.state.value}")

    # Layer 2 (Medium): 10^9 - 10^10
    result = probe.operate(target_mass=5e9, layer=Layer.MEDIUM)
    print(f"Layer MEDIUM (5e9): state={result.state.value}")

    # 複数質量の干渉による負圧ポイント探索
    result = probe.operate(
        target_mass=1e8,
        layer=Layer.MICRO,
        additional_contours=[
            (2e8, Layer.MICRO),
            (1.5e8, Layer.MICRO),
        ],
        initial_position=(1.0, 1.0, 1.0),
    )
    print(f"Multi-contour: state={result.state.value}")
    if result.transition_result and result.transition_result.success:
        print(f"  Position shift: {result.transition_result.origin} -> {result.transition_result.destination}")
        print(f"  Energy cost: {result.transition_result.energy_cost}")

    # 観測のみ（Idle モード）
    obs = probe.observe_only(target_mass=1e6)
    print(f"Observe only: state={obs.state.value}, reason={obs.idle_reason}")


if __name__ == "__main__":
    main()
