"""
VacuumPathFinder — 負圧ポイント発見モジュール

複数の質量が干渉し合い、ポテンシャルが打ち消し合う「抵抗ゼロ座標」を抽出する。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from gap.impedance_finalizer import FinalizedContour


@dataclass
class VacuumPoint:
    """負圧ポイント（特異点）"""

    coordinates: Tuple[float, float, float]
    potential_gradient: float  # ゼロに近いほど負圧が強い
    is_singular: bool  # 特異点として確定されたか


class VacuumPathFinder:
    """
    負圧ポイント発見器。

    複数の質量が干渉し合う領域において、ポテンシャルが打ち消し合う
    抵抗ゼロ座標（特異点）を抽出する。

    重要ルール: 特異点が計算上出現していない場合、一切の演算と行動を停止する。
    """

    def __init__(self, singularity_threshold: float = 1e-4):
        """
        Args:
            singularity_threshold: 特異点判定の閾値（勾配がこれ以下で特異点）
        """
        self.singularity_threshold = singularity_threshold

    def find_vacuum_points(
        self,
        contours: List[FinalizedContour],
        search_space: Optional[np.ndarray] = None,
        grid_resolution: int = 50,
    ) -> List[VacuumPoint]:
        """
        複数の質量輪郭から負圧ポイントを抽出する。

        ポテンシャルが打ち消し合う座標 = 抵抗ゼロ座標。

        Args:
            contours: 確定済みの質量輪郭群
            search_space: 探索空間（オプション）
            grid_resolution: グリッド解像度

        Returns:
            発見された負圧ポイントのリスト（空の場合は Idle 推奨）
        """
        if len(contours) < 2:
            return []

        # 探索空間の決定
        if search_space is None:
            all_centroids = np.array([c.centroid for c in contours])
            min_coords = np.min(all_centroids, axis=0) - 2
            max_coords = np.max(all_centroids, axis=0) + 2
            search_space = np.vstack([min_coords, max_coords])

        # グリッド上でポテンシャル勾配を計算
        x = np.linspace(
            search_space[0, 0], search_space[1, 0], grid_resolution
        )
        y = np.linspace(
            search_space[0, 1], search_space[1, 1], grid_resolution
        )
        z = np.linspace(
            search_space[0, 2], search_space[1, 2], grid_resolution
        )

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # 各点でのポテンシャル勾配（合計）
        total_gradient = np.zeros(len(points))

        for c in contours:
            centroid = np.array(c.centroid)
            dist = np.linalg.norm(points - centroid, axis=1) + 1e-10
            # 重力ポテンシャル的な勾配（質量/距離^2）
            grad = c.impedance * c.radius / (dist**2)
            total_gradient += grad

        # 勾配の差分（干渉による打ち消しを検出）
        grad_sorted = np.sort(total_gradient)
        min_grad = np.min(total_gradient)
        min_idx = np.argmin(total_gradient)

        if min_grad > self.singularity_threshold:
            # 特異点が出現していない → Idle State 推奨
            return []

        vacuum_point = VacuumPoint(
            coordinates=tuple(points[min_idx]),
            potential_gradient=float(min_grad),
            is_singular=min_grad <= self.singularity_threshold,
        )

        points_found = [vacuum_point]

        # 複数の特異点を探索（局所最小）
        for i in range(len(points)):
            if total_gradient[i] <= self.singularity_threshold:
                if not any(
                    np.allclose(p.coordinates, points[i])
                    for p in points_found
                ):
                    points_found.append(
                        VacuumPoint(
                            coordinates=tuple(points[i]),
                            potential_gradient=float(total_gradient[i]),
                            is_singular=True,
                        )
                    )

        return points_found[:10]  # 最大10点まで

    def validate_singularity(self, point: VacuumPoint) -> bool:
        """
        特異点が「吸い込まれる」状態として有効か検証する。

        負圧の優先: 押し出すのではなく、吸い込まれる状態が確立されているか。

        Args:
            point: 負圧ポイント

        Returns:
            有効な特異点か

        """
        return (
            point.is_singular
            and point.potential_gradient <= self.singularity_threshold
        )
