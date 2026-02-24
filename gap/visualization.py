"""
GAP 可視化モジュール

ポテンシャル場・質量球体・負圧ポイント・Vacuum Path を点群グラフで可視化する。
三角関数によるエネルギー流の測定結果を3D空間で表現する。
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Dict

from gap.constants import Layer
from gap.impedance_finalizer import ImpedanceFinalizer, FinalizedContour
from gap.vacuum_path_finder import VacuumPathFinder, VacuumPoint, PotentialFieldData
from gap.transition_executor import TransitionExecutor

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def create_three_sphere_contours(
    masses: Tuple[float, float, float],
    positions: Tuple[Tuple[float, float, float], ...],
    layer: Layer = Layer.MEDIUM,
) -> List[FinalizedContour]:
    """
    三つ巴モデル用の質量輪郭を生成する。

    互いに質量が拮抗した三球を指定座標に配置する。

    Args:
        masses: 三球の質量 (m1, m2, m3)
        positions: 三球の中心座標 ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
        layer: 階層

    Returns:
        確定された輪郭のリスト
    """
    finalizer = ImpedanceFinalizer()
    return [
        finalizer.create_contour_at(m, layer, pos)
        for m, pos in zip(masses, positions)
    ]


def build_visualization_data(
    contours: List[FinalizedContour],
    vacuum_points: Optional[List[VacuumPoint]] = None,
    vacuum_path: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    probe_position: Optional[Tuple[float, float, float]] = None,
    grid_resolution: int = 25,
) -> Dict[str, Any]:
    """
    可視化用のデータを構築する。

    Args:
        contours: 質量輪郭
        vacuum_points: 発見された負圧ポイント
        vacuum_path: ゼロコスト通過経路 (origin, destination)
        probe_position: プローブの現在位置
        grid_resolution: ポテンシャル場のグリッド解像度

    Returns:
        可視化データの辞書
    """
    finder = VacuumPathFinder()
    field = finder.compute_potential_field(
        contours,
        grid_resolution=grid_resolution,
    )

    # 負圧が強い（potential が低い）領域を抽出
    low_pressure_mask = field.potential.ravel() < np.percentile(
        field.potential.ravel(), 15
    )
    vacuum_cloud = field.points[low_pressure_mask]
    vacuum_values = field.potential.ravel()[low_pressure_mask]

    return {
        "contours": contours,
        "field": field,
        "vacuum_points": vacuum_points or [],
        "vacuum_path": vacuum_path,
        "probe_position": probe_position,
        "vacuum_cloud": vacuum_cloud,
        "vacuum_values": vacuum_values,
        "all_points": field.points,
        "all_potential": field.potential.ravel(),
    }


def create_sphere_mesh(
    center: Tuple[float, float, float],
    radius: float,
    n_theta: int = 20,
    n_phi: int = 20,
):
    """球体のメッシュを生成（質量球の可視化用）"""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    T, P = np.meshgrid(theta, phi)
    x = center[0] + radius * np.sin(P) * np.cos(T)
    y = center[1] + radius * np.sin(P) * np.sin(T)
    z = center[2] + radius * np.cos(P)
    return x, y, z


def create_vacuum_figure(
    masses: Tuple[float, float, float],
    positions: Tuple[Tuple[float, float, float], ...],
    probe_position: Tuple[float, float, float],
    layer: Layer = Layer.MEDIUM,
    grid_resolution: int = 25,
    relative_percentile: float = 3.0,
):
    """
    三つ巴モデルの Plotly Figure を生成する。

    Args:
        masses: 三球の質量
        positions: 三球の中心座標
        probe_position: プローブの現在位置
        layer: 階層
        grid_resolution: ポテンシャル場の解像度
        relative_percentile: 特異点検出の相対百分位

    Returns:
        plotly.graph_objects.Figure または None（Plotly 未導入時）
    """
    if not HAS_PLOTLY:
        return None

    contours = create_three_sphere_contours(
        masses=masses,
        positions=positions,
        layer=layer,
    )

    finder = VacuumPathFinder()
    vacuum_points = finder.find_vacuum_points(
        contours,
        grid_resolution=40,
        use_relative_threshold=True,
        relative_percentile=relative_percentile,
    )

    vacuum_path = None
    if vacuum_points:
        executor = TransitionExecutor()
        best = min(vacuum_points, key=lambda v: v.potential_gradient)
        result = executor.execute(probe_position, best)
        if result.success:
            vacuum_path = (result.origin, result.destination)

    data = build_visualization_data(
        contours=contours,
        vacuum_points=vacuum_points,
        vacuum_path=vacuum_path,
        probe_position=probe_position,
        grid_resolution=grid_resolution,
    )

    fig = go.Figure()
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, c in enumerate(contours):
        x, y, z = create_sphere_mesh(
            c.centroid, c.radius * 0.5, n_theta=15, n_phi=15
        )
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, colors[i % 3]], [1, colors[i % 3]]],
            opacity=0.6,
            showscale=False,
            name=f"Layer {c.layer.name}",
        ))

    vc = data["vacuum_cloud"]
    if len(vc) > 0:
        fig.add_trace(go.Scatter3d(
            x=vc[:, 0], y=vc[:, 1], z=vc[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=data["vacuum_values"],
                colorscale="Blues_r",
                opacity=0.5,
                showscale=True,
                colorbar=dict(title="負圧度"),
            ),
            name="負圧領域",
        ))

    if vacuum_points:
        vx = [v.coordinates[0] for v in vacuum_points]
        vy = [v.coordinates[1] for v in vacuum_points]
        vz = [v.coordinates[2] for v in vacuum_points]
        fig.add_trace(go.Scatter3d(
            x=vx, y=vy, z=vz,
            mode="markers",
            marker=dict(
                size=12, color="gold", symbol="diamond", line=dict(width=2)
            ),
            name="特異点（量子トンネル）",
        ))

    if vacuum_path:
        ox, oy, oz = vacuum_path[0]
        dx, dy, dz = vacuum_path[1]
        fig.add_trace(go.Scatter3d(
            x=[ox, dx], y=[oy, dy], z=[oz, dz],
            mode="lines+markers",
            line=dict(color="cyan", width=8, dash="dot"),
            marker=dict(size=8),
            name="Vacuum Path（空いたら通る）",
        ))

    fig.add_trace(go.Scatter3d(
        x=[probe_position[0]],
        y=[probe_position[1]],
        z=[probe_position[2]],
        mode="markers",
        marker=dict(
            size=10, color="white", symbol="x", line=dict(width=2)
        ),
        name="プローブ",
    ))

    fig.update_layout(
        title="GAP — 三つ巴モデルと負圧ポイントの可視化",
        scene=dict(
            xaxis_title="X（抽象化空間）",
            yaxis_title="Y（抽象化空間）",
            zaxis_title="Z（抽象化空間）",
            aspectmode="data",
            bgcolor="rgba(20,20,30,1)",
        ),
        paper_bgcolor="rgba(20,20,30,1)",
        font=dict(color="white"),
        showlegend=True,
    )

    return fig


def _brachistochrone_path(
    t: np.ndarray,
    start_pos: Tuple[float, float, float],
    end_pos: Tuple[float, float, float],
) -> np.ndarray:
    """
    Brachistochrone Curve（最速降下曲線）に沿った経路を計算する。

    重力に引かれて吸い込まれるように穴へ向かう曲線。
    サイクロイドのパラメータ表示を用い、最速で落下する経路を表現。

    Args:
        t: 0〜1 の進行度
        start_pos: 開始座標 (x, y, z) — やや外側の待機位置
        end_pos: 終了座標（穴の中心、通過後）

    Returns:
        (N, 3) の座標配列
    """
    theta = np.pi * np.clip(t, 0, 1)  # 0 → π
    x0, y0, z0 = start_pos
    x1, y1, z1 = end_pos

    # サイクロイド形状: x = R(θ - sin θ), z = R(1 - cos θ)
    # 水平方向: (θ - sin θ) / π で 0→1、start から end へ
    s_x = (theta - np.sin(theta)) / np.pi
    # 垂直方向: (1 - cos θ) / 2 で 0→1、上から下へ（中央で加速）
    s_z = (1 - np.cos(theta)) / 2

    x = x0 + (x1 - x0) * s_x
    y = y0 + (y1 - y0) * s_x
    z = z0 + (z1 - z0) * s_z

    return np.column_stack([x, y, z])


def _simulate_orbital_with_collisions(
    n_frames: int,
    sphere_radius: float,
    hole_radius: float,
    basin_radius: float,
    dt: float,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    ぐるぐる回転しながらぶつかり合い、黄色の穴には入れない三つ巴。
    隙の一瞬に白い球が潜り抜けるための穴開きタイミングを計算。

    軌道回転 + 衝突 + 穴への進入禁止。
    """
    if seed is not None:
        np.random.seed(seed)

    # 軌道パラメータ: 角速度を少しずつ変えて追い抜き・衝突を発生させる
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    radii = np.array([basin_radius * 0.85, basin_radius * 0.9, basin_radius * 0.88])
    omegas = np.array([0.8, 0.75, 0.82])  # 微妙に違う速度で追い抜き

    min_dist = 2 * sphere_radius
    hole_repel_radius = hole_radius + sphere_radius * 2.0

    def get_pos():
        return np.array([
            [radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i]), 0.0]
            for i in range(3)
        ])

    positions_per_frame = [np.array([get_pos()[i]]) for i in range(3)]
    angles_history = [angles.copy()]

    for _ in range(n_frames - 1):
        # 1. 軌道回転（ぐるぐる）
        angles += omegas * dt
        pos = get_pos()

        # 2. 球同士の衝突（ぶつかって跳ね返る）
        for i in range(3):
            for j in range(i + 1, 3):
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff) + 1e-10
                if dist < min_dist:
                    normal = diff / dist
                    # 角速度を反転して跳ね返る
                    omegas[i], omegas[j] = -omegas[j] * 0.8, -omegas[i] * 0.8
                    # 角度を少しずらして重なりを解消
                    angles[i] -= 0.1
                    angles[j] += 0.1
                    pos = get_pos()

        # 3. 黄色の穴には入れない（中心に近づいたら押し返す）
        for i in range(3):
            r = radii[i]
            if r < hole_repel_radius:
                radii[i] = hole_repel_radius + 0.5
                omegas[i] *= -0.7  # 跳ね返り
            radii[i] = np.clip(radii[i], hole_repel_radius + 0.3, basin_radius - 0.3)

        pos = get_pos()
        angles_history.append(angles.copy())

        # 4. 記録
        for i in range(3):
            positions_per_frame[i] = np.vstack([
                positions_per_frame[i],
                pos[i].reshape(1, 3),
            ])

    # 隙の一瞬: 三球が一か所に集まった時、反対側に隙ができる
    angles_arr = np.array(angles_history)
    angles_norm = np.mod(angles_arr, 2 * np.pi)
    angular_spread = np.max(angles_norm, axis=1) - np.min(angles_norm, axis=1)
    # 2πをまたぐ場合の補正
    angular_spread = np.minimum(angular_spread, 2 * np.pi - angular_spread)
    # 三球が集まっている（spread が小さい）= 隙が開いている
    hole_open = angular_spread < 1.8

    return positions_per_frame, hole_open


def compute_three_body_animation_frames(
    n_frames: int = 120,
    orbit_radius: float = 6.0,
    orbit_tilt: float = 0.3,
    hole_radius: float = 1.5,
    probe_drop_frame: Optional[int] = None,
    probe_drop_duration: int = 20,
    probe_start_offset: float = 2.0,
    sphere_radius: float = 0.8,
    use_collision_mode: bool = True,
) -> tuple:
    """
    三体問題のアニメーションフレームを計算する。

    三球がぶつかり合い、穴にハマらず互いに押し合う。
    スケールが一回り小さいプローブが Brachistochrone Curve に沿って
    吸い込まれるように穴をすり抜ける。

    Returns:
        (sphere_positions_per_frame, probe_positions_per_frame, hole_open_frames)
    """
    if use_collision_mode:
        sphere_positions, hole_open = _simulate_orbital_with_collisions(
            n_frames=n_frames,
            sphere_radius=sphere_radius,
            hole_radius=hole_radius,
            basin_radius=orbit_radius,
            dt=0.15,
            seed=42,
        )
    else:
        # 従来の回転モード（フォールバック）
        t = np.linspace(0, 4 * np.pi, n_frames)
        phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        snag = -0.5 * np.sin(t * 1.8) * np.sin(t * 0.5 + 0.3)
        sphere_positions = []
        for i, phase in enumerate(phases):
            angle = t + phase
            r = orbit_radius + snag * (1.1 if i == 1 else 1.0)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = orbit_tilt * (r - orbit_radius) * np.sin(t * 1.5)
            sphere_positions.append(np.column_stack([x, y, z]))
        dist_from_center = np.array([
            np.linalg.norm(sphere_positions[i], axis=1)
            for i in range(3)
        ])
        hole_open = np.min(dist_from_center, axis=0) > (orbit_radius - hole_radius)

    # プローブが落下を始めるフレーム: 穴が初めて開くタイミング
    if probe_drop_frame is None:
        open_frames = np.where(hole_open)[0]
        probe_drop_frame = int(n_frames * 0.4) if len(open_frames) == 0 else open_frames[len(open_frames) // 3]

    # プローブ: 待機 → Brachistochrone Curve に沿って吸い込まれるように穴をすり抜け
    start_pos = (probe_start_offset, 0.0, 5.0)   # やや外側から
    end_pos = (0.0, 0.0, -5.0)                   # 穴を抜けて下へ

    probe_path = _brachistochrone_path(
        np.linspace(0, 1, probe_drop_duration + 1),
        start_pos,
        end_pos,
    )

    probe_positions = []
    for frame in range(n_frames):
        if frame < probe_drop_frame:
            # 上で待機（スケールが一回り小さいので上に配置）
            probe_positions.append(list(start_pos))
        elif frame < probe_drop_frame + probe_drop_duration:
            # Brachistochrone Curve に沿って吸い込まれる
            idx = frame - probe_drop_frame
            probe_positions.append(list(probe_path[idx]))
        else:
            # 通過完了（下に到達）
            probe_positions.append(list(end_pos))

    return sphere_positions, probe_positions, hole_open


def create_vacuum_animation_figure(
    n_frames: int = 120,
    orbit_radius: float = 6.0,
    sphere_radius: float = 0.8,
    probe_radius: float = 0.3,
    probe_start_offset: float = 2.0,
) -> "go.Figure | None":
    """
    三体問題＋プローブすり抜けのアニメーション Figure を生成する。

    赤・青・緑の球がぐるぐる回り、中心の穴に引っかかり、
    小さいプローブが Brachistochrone Curve に沿って吸い込まれるようにすり抜ける。
    """
    if not HAS_PLOTLY:
        return None

    (sphere_positions, probe_positions, hole_open) = compute_three_body_animation_frames(
        n_frames=n_frames,
        orbit_radius=orbit_radius,
        probe_start_offset=probe_start_offset,
        sphere_radius=sphere_radius,
        use_collision_mode=True,
    )

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # 各フレームのデータを構築
    frames = []
    for frame in range(n_frames):
        traces_data = []

        # 三球（Surface はアニメーションで扱いにくいので Scatter3d の球で代用）
        for i in range(3):
            pos = sphere_positions[i][frame]
            # 球を点で表現（サイズで球体感を出す）
            traces_data.append(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers",
                    marker=dict(
                        size=sphere_radius * 15,
                        color=colors[i],
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    name=["赤", "青", "緑"][i],
                )
            )

        # 黄色の穴（特異点）- 隙が開いている時は強調
        hole_opacity = 0.95 if hole_open[frame] else 0.35
        traces_data.append(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers",
                marker=dict(
                    size=14,
                    color="gold",
                    symbol="diamond",
                    opacity=hole_opacity,
                    line=dict(width=2, color="yellow"),
                ),
                name="黄色の穴（隙の一瞬）",
            )
        )

        # プローブ（小さい球）
        probe_pos = probe_positions[frame]
        traces_data.append(
            go.Scatter3d(
                x=[probe_pos[0]],
                y=[probe_pos[1]],
                z=[probe_pos[2]],
                mode="markers",
                marker=dict(
                    size=probe_radius * 15,
                    color="white",
                    opacity=0.9,
                    symbol="circle",
                    line=dict(width=2, color="cyan"),
                ),
                name="プローブ（Brachistochrone）",
            )
        )

        frames.append(
            go.Frame(
                data=traces_data,
                name=str(frame),
            )
        )

    # 初期フレームのデータ
    initial_data = []
    for i in range(3):
        pos = sphere_positions[i][0]
        initial_data.append(
            go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers",
                marker=dict(
                    size=sphere_radius * 15,
                    color=colors[i],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                name=["赤", "青", "緑"][i],
            )
        )
    initial_data.append(
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers",
            marker=dict(size=14, color="gold", symbol="diamond", opacity=0.35),
            name="黄色の穴",
        )
    )
    probe_pos_0 = probe_positions[0]
    initial_data.append(
        go.Scatter3d(
            x=[probe_pos_0[0]], y=[probe_pos_0[1]], z=[probe_pos_0[2]],
            mode="markers",
            marker=dict(
                size=probe_radius * 15,
                color="white",
                opacity=0.9,
                symbol="circle",
                line=dict(width=2, color="cyan"),
            ),
            name="プローブ（Brachistochrone）",
        )
    )

    fig = go.Figure(
        data=initial_data,
        frames=frames,
    )

    # スライダーと再生ボタン
    fig.update_layout(
        title="GAP — ぐるぐる回転・ぶつかり合い・黄色の穴には入れない三つ巴",
        scene=dict(
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-8, 8]),
            aspectmode="cube",
            bgcolor="rgba(20,20,30,1)",
        ),
        paper_bgcolor="rgba(20,20,30,1)",
        font=dict(color="white"),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ 再生",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=80, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ 停止",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                x=0.1,
                xanchor="left",
                y=0,
                yanchor="top",
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [f.name],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                            ),
                        ],
                        label=str(i),
                        method="animate",
                    )
                    for i, f in enumerate(frames)
                ],
                transition=dict(duration=0),
                x=0.1,
                len=0.9,
                xanchor="left",
                y=0,
                yanchor="top",
                currentvalue=dict(
                    visible=True,
                    prefix="フレーム: ",
                    xanchor="center",
                ),
            ),
        ],
    )

    return fig
