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
        title="",
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


def create_base_establishment_figure(
    positions: Tuple[Tuple[float, float, float], ...] = ((6, 0, 0), (-6, 6, 0), (-6, -6, 0)),
    probe_position: Tuple[float, float, float] = (0, 0, 5),
    sphere_radius: float = 0.8,
    n_orbit_frames: int = 48,
    orbit_speed: float = 0.15,
) -> "go.Figure | None":
    """
    すり抜け前のフェーズを可視化。

    三つ巴の球が回転・動く姿に高周波をあて、形状を学習する姿を強調。
    接続なしから始まり、順次 HITSCAN → HITPLAN → HITSERIES が現れる。
    学習が済むと等高線が現れる。
    """
    if not HAS_PLOTLY:
        return None

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    orbit_radius = 6.0
    base_phases = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])

    def get_orbital_positions(frame: int) -> List[Tuple[float, float, float]]:
        angles = base_phases + frame * orbit_speed
        return [
            (orbit_radius * np.cos(angles[i]), orbit_radius * np.sin(angles[i]), 0.0)
            for i in range(3)
        ]

    no_connection_end = int(n_orbit_frames * 0.12)
    hitscan_start = no_connection_end
    hitplan_start = int(n_orbit_frames * 0.35)
    hitseries_start = int(n_orbit_frames * 0.55)
    learning_complete_frame = int(n_orbit_frames * 0.55)

    n_grid = 25
    xg = np.linspace(-10, 10, n_grid)
    yg = np.linspace(-10, 10, n_grid)
    X, Y = np.meshgrid(xg, yg)
    R2 = X ** 2 + Y ** 2
    Z_contour = np.clip(-6 + 0.04 * R2 - 2 * np.exp(-R2 / 40), -7, 2)

    def add_contour_trace(traces_list: list, frame: int) -> None:
        visible = frame >= learning_complete_frame
        progress = (frame - learning_complete_frame) / max(1, n_orbit_frames - learning_complete_frame) if visible else 0
        opacity = min(0.25, 0.08 + 0.17 * progress) if visible else 0
        traces_list.append(
            go.Surface(
                x=X, y=Y, z=Z_contour,
                colorscale=[[0, "rgba(40,80,120,0.4)"], [0.5, "rgba(30,90,140,0.3)"], [1, "rgba(20,100,160,0.25)"]],
                opacity=opacity,
                showscale=False,
                contours=dict(
                    z=dict(show=True, usecolormap=False, project=dict(z=True), color="rgba(120,180,220,0.4)"),
                ),
                name="等高線（学習完了）",
                visible=visible,
            )
        )

    hitscan_red_frame = 7
    hitscan_green_frame = 10
    hitscan_blue_frame = 14
    connection_order = [0, 2, 1]  # 赤, 緑, 青の順
    connection_frames = [hitscan_red_frame, hitscan_green_frame, hitscan_blue_frame]

    # GAAS zero cost breakthrough のフレーム0と同じサイズ（probe_radius * 15 = 4.5）
    probe_size = 0.3 * 15

    def _make_probe_trace():
        return go.Scatter3d(
            x=[probe_position[0]], y=[probe_position[1]], z=[probe_position[2]],
            mode="markers",
            marker=dict(size=probe_size, color="white", symbol="circle", line=dict(width=2, color="white"), opacity=1),
            name="白い球（プローブ）",
        )

    def _placeholder_scatter3d(visible: bool = False):
        return go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=1), visible=visible, showlegend=False)

    def _placeholder_surface(visible: bool = False):
        return go.Surface(x=[[0, 0], [0, 0]], y=[[0, 0], [0, 0]], z=[[0, 0], [0, 0]], visible=visible, showscale=False)

    frames = []
    for frame in range(n_orbit_frames):
        pos_list = get_orbital_positions(frame)
        traces = []
        add_contour_trace(traces, frame)

        for i, pos in enumerate(pos_list):
            traces.append(
                go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode="markers",
                    marker=dict(size=sphere_radius * 15, color=colors[i], opacity=0.8, line=dict(width=2, color="white")),
                    name=["赤", "青", "緑"][i],
                )
            )

        traces.append(_make_probe_trace())

        # HITSCAN — 常に3本分追加し visible で制御（トレース数固定）
        n_connected = sum(1 for f in connection_frames if frame >= f)
        for idx in range(3):
            if idx < n_connected:
                i = connection_order[idx]
                pos = pos_list[i]
                t = np.linspace(0, 1, 50)
                wave = 0.5 * np.sin(t * 25) * (1 - t)
                x = probe_position[0] + t * (pos[0] - probe_position[0]) + wave * (pos[1] - probe_position[1]) * 0.2
                y = probe_position[1] + t * (pos[1] - probe_position[1]) - wave * (pos[0] - probe_position[0]) * 0.2
                z = probe_position[2] + t * (pos[2] - probe_position[2])
                traces.append(
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="lines",
                        line=dict(color=colors[i], width=8),
                        opacity=1.0,
                        visible=True,
                        name="HITSCAN 高周波照射" if frame == hitscan_red_frame and idx == 0 else None,
                        showlegend=(frame == hitscan_red_frame and idx == 0),
                    )
                )
            else:
                traces.append(_placeholder_scatter3d(visible=False))

        # HITPLAN — 常に4本分追加（軌道1 + 接続線3）
        if frame >= hitplan_start:
            theta_map = np.linspace(0, 2 * np.pi, 60)
            x_map = orbit_radius * np.cos(theta_map)
            y_map = orbit_radius * np.sin(theta_map)
            z_map = np.zeros_like(theta_map)
            dash_len, gap_len = 3, 2
            n_pts = len(theta_map)
            x_dash, y_dash, z_dash = [], [], []
            k = 0
            while k < n_pts:
                for _ in range(dash_len):
                    if k < n_pts:
                        x_dash.append(float(x_map[k]))
                        y_dash.append(float(y_map[k]))
                        z_dash.append(float(z_map[k]))
                        k += 1
                if k < n_pts:
                    x_dash.append(np.nan)
                    y_dash.append(np.nan)
                    z_dash.append(np.nan)
                    k += gap_len
            map_opacity = 0.2 + 0.25 * min(1.0, (frame - hitplan_start) / max(1, hitseries_start - hitplan_start))
            traces.append(
                go.Scatter3d(
                    x=x_dash, y=y_dash, z=z_dash,
                    mode="lines",
                    line=dict(color="rgba(150,200,255,0.8)", width=2),
                    opacity=map_opacity,
                    name="マッピング（回転軌道）" if frame == hitplan_start else None,
                    showlegend=(frame == hitplan_start),
                )
            )
            for i, pos in enumerate(pos_list):
                traces.append(
                    go.Scatter3d(
                        x=[probe_position[0], pos[0]],
                        y=[probe_position[1], pos[1]],
                        z=[probe_position[2], pos[2]],
                        mode="lines",
                        line=dict(color=colors[i], width=4),
                        opacity=0.85,
                        name="HITPLAN 神経接続" if frame == hitplan_start and i == 0 else None,
                        showlegend=(frame == hitplan_start and i == 0),
                    )
                )
        else:
            for _ in range(4):
                traces.append(_placeholder_scatter3d(visible=False))

        # HITSERIES CICD — 常に4本分追加（Surface3 + hole1）
        if frame >= hitseries_start:
            for i, pos in enumerate(pos_list):
                theta = np.linspace(0, 2 * np.pi, 16)
                phi = np.linspace(0, np.pi, 10)
                T, P = np.meshgrid(theta, phi)
                r = sphere_radius * 1.2
                x = pos[0] + r * np.sin(P) * np.cos(T)
                y = pos[1] + r * np.sin(P) * np.sin(T)
                z = pos[2] + r * np.cos(P)
                traces.append(
                    go.Surface(
                        x=x, y=y, z=z,
                        colorscale=[[0, colors[i]], [1, colors[i]]],
                        opacity=0.5,
                        showscale=False,
                        name="HITSERIES 形状学習" if frame == hitseries_start and i == 0 else None,
                    )
                )
            # 空間の下部に穴があることが見えてくる
            hole_opacity = 0.3 + 0.4 * min(1.0, (frame - hitseries_start) / max(1, n_orbit_frames - hitseries_start))
            traces.append(
                go.Scatter3d(
                    x=[0], y=[0], z=[-6],
                    mode="markers",
                    marker=dict(size=8, color="gold", symbol="diamond", opacity=hole_opacity, line=dict(width=1, color="yellow")),
                    name="下部の穴（見えてくる）" if frame == hitseries_start else None,
                    showlegend=(frame == hitseries_start),
                )
            )
        else:
            for _ in range(3):
                traces.append(_placeholder_surface(visible=False))
            traces.append(_placeholder_scatter3d(visible=False))

        frames.append(go.Frame(data=traces, name=str(frame), layout=go.Layout(title=dict(text=""))))

    # initial_data は frames[0] と同じ構造にする（Plotly はトレース数が一致している必要あり）
    initial_data = frames[0].data if frames else []

    fig = go.Figure(data=initial_data, frames=frames)
    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-5, 10]),
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
                    dict(label="▶ 再生", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ 停止", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
                x=0.1, y=0,
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(args=[[str(f)], dict(frame=dict(duration=0), mode="immediate")], label=str(f), method="animate")
                    for f in range(n_orbit_frames)
                ],
                x=0.1, len=0.9, xanchor="left", y=0,
                currentvalue=dict(visible=True, prefix="フレーム: ", xanchor="center"),
            ),
        ],
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
    漏斗型の領域は凸凹しているため、三つの玉は近づいたり離れたりしながら、
    穴には入れない状態が続く。スロープトイのように並行ではなく不規則な形状。

    軌道回転 + 衝突 + 穴への進入禁止。
    """
    if seed is not None:
        np.random.seed(seed)

    # 軌道パラメータ: 角速度を少しずつ変えて追い抜き・衝突を発生させる
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    base_radii = np.array([basin_radius * 0.85, basin_radius * 0.9, basin_radius * 0.88])
    omegas = np.array([0.8, 0.75, 0.82])  # 微妙に違う速度で追い抜き

    min_dist = 2 * sphere_radius
    hole_repel_radius = hole_radius + sphere_radius * 2.0

    # 漏斗の凸凹: 各球の有効半径が時間で変動（近づいたり離れたり）
    bump_amp = basin_radius * 0.12
    bump_phases = np.array([0.0, 1.2, 2.5])

    def get_radii(t: float) -> np.ndarray:
        # 複数周波数の重ね合わせで不規則な凸凹を表現（漏斗型の凸凹）
        bumps = (
            np.sin(t * 0.4 + bump_phases) * 0.6
            + np.sin(t * 0.9 + bump_phases * 1.3) * 0.3
            + np.sin(t * 1.7 + bump_phases * 0.7) * 0.2
        )
        radii = base_radii + bump_amp * bumps
        return np.clip(radii, hole_repel_radius + 0.3, basin_radius - 0.3)

    def get_pos(t: float):
        radii = get_radii(t)
        return np.array([
            [radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i]), 0.0]
            for i in range(3)
        ])

    positions_per_frame = [np.array([get_pos(0.0)[i]]) for i in range(3)]
    angles_history = [angles.copy()]
    t = 0.0

    for _ in range(n_frames - 1):
        # 1. 軌道回転（ぐるぐる）
        angles += omegas * dt
        t += dt
        pos = get_pos(t)

        # 2. 球同士の衝突（ぶつかって跳ね返る）
        for i in range(3):
            for j in range(i + 1, 3):
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff) + 1e-10
                if dist < min_dist:
                    # 角速度を反転して跳ね返る
                    omegas[i], omegas[j] = -omegas[j] * 0.8, -omegas[i] * 0.8
                    angles[i] -= 0.1
                    angles[j] += 0.1
                    pos = get_pos(t)

        # 3. 黄色の穴には入れない（中心に近づいたら押し返す）- 凸凹があっても穴には入れない
        radii = get_radii(t)
        for i in range(3):
            if radii[i] < hole_repel_radius + 0.5:
                omegas[i] *= -0.7  # 跳ね返り
        pos = get_pos(t)

        angles_history.append(angles.copy())

        # 4. 記録
        for i in range(3):
            positions_per_frame[i] = np.vstack([
                positions_per_frame[i],
                pos[i].reshape(1, 3),
            ])

    # 隙の一瞬: 大きな玉が離れた時、一瞬だけルートがあき、球にぶつからずに黄色い穴に入れる
    angles_arr = np.array(angles_history)
    angles_norm = np.mod(angles_arr, 2 * np.pi)
    angular_spread = np.max(angles_norm, axis=1) - np.min(angles_norm, axis=1)
    # 2πをまたぐ場合の補正
    angular_spread = np.minimum(angular_spread, 2 * np.pi - angular_spread)
    # 三球が離れている（spread が大きい ≈ 120°）= 一瞬だけ中心にルートが開く
    hole_open = angular_spread > 1.85

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
    probe_radius: float = 0.3,
    use_collision_mode: bool = True,
    n_cycles: int = 1,
    cycle_scale_factor: float = 1.8,
    transition_frames: int = 15,
) -> tuple:
    """
    三体問題のアニメーションフレームを計算する。

    三球がぶつかり合い、穴にハマらず互いに押し合う。
    スケールが一回り小さいプローブが Brachistochrone Curve に沿って
    吸い込まれるように穴をすり抜ける。

    n_cycles > 1 の場合: 大きくなった球がさらに大きな三つ巴を発見し、
    同様にすり抜けていくことを繰り返す。

    Returns:
        (sphere_positions_per_frame, probe_positions_per_frame, hole_open_frames, probe_sizes, sphere_sizes, hole_sizes)
    """
    base_probe_size = probe_radius * 15
    hole_marker_size = 14
    n_growth_frames = 25

    all_sphere_positions = [[], [], []]
    all_probe_positions = []
    all_hole_open = []
    all_probe_sizes = []
    all_sphere_sizes = []
    all_hole_sizes = []

    current_probe_size = base_probe_size
    prev_end_pos = None

    for cycle in range(n_cycles):
        scale = cycle_scale_factor ** cycle
        r_orbit = orbit_radius * scale
        r_sphere = sphere_radius * scale
        r_hole = hole_radius * scale
        n_f = n_frames if cycle == 0 else n_frames  # 各サイクル同じフレーム数

        if use_collision_mode:
            sphere_positions, hole_open = _simulate_orbital_with_collisions(
                n_frames=n_f,
                sphere_radius=r_sphere,
                hole_radius=r_hole,
                basin_radius=r_orbit,
                dt=0.15,
                seed=42 + cycle,
            )
        else:
            t = np.linspace(0, 4 * np.pi, n_f)
            phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
            snag = -0.5 * np.sin(t * 1.8) * np.sin(t * 0.5 + 0.3)
            sphere_positions = []
            for i, phase in enumerate(phases):
                angle = t + phase
                r = r_orbit + snag * (1.1 if i == 1 else 1.0)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = orbit_tilt * (r - r_orbit) * np.sin(t * 1.5)
                sphere_positions.append(np.column_stack([x, y, z]))
            dist_from_center = np.array([
                np.linalg.norm(sphere_positions[i], axis=1)
                for i in range(3)
            ])
            hole_open = np.min(dist_from_center, axis=0) > (r_orbit - r_hole)

        if probe_drop_frame is None:
            open_frames = np.where(hole_open)[0]
            pdrop = int(n_f * 0.4) if len(open_frames) == 0 else open_frames[len(open_frames) // 3]
        else:
            pdrop = probe_drop_frame

        start_pos = (
            probe_start_offset * scale,
            0.0,
            5.0 * scale,
        )
        end_pos = (0.0, 0.0, -5.0 * scale)
        hole_size_scaled = hole_marker_size * (1.0 + 0.3 * cycle)
        # 1回目: 白球の膨張を控えめに
        # 2度目以降: 黄色の枠を少し広げる（小さめに）、通過後に2倍に膨張
        if cycle >= 1:
            hole_size_scaled = hole_size_scaled * 1.2
        if cycle == 0:
            growth_target = hole_size_scaled * 1.7
        else:
            growth_target = current_probe_size * 2.0  # 通り終わったら2倍に膨張

        # 三つ巴の球サイズ: 白球より大きく描く（大きな三つ巴を発見）
        base_sphere_size = sphere_radius * 15
        if cycle == 0:
            sphere_size = base_sphere_size
        else:
            # 白球（current_probe_size）より 1.4 倍大きく
            sphere_size = max(base_sphere_size * scale, current_probe_size * 1.4)

        probe_path = _brachistochrone_path(
            np.linspace(0, 1, probe_drop_duration + 1),
            start_pos,
            end_pos,
        )

        cycle_probe_positions = []
        cycle_probe_sizes = []

        # サイクル間の遷移: 前サイクルの終了位置から開始位置へ（大きな三つ巴を発見）
        if cycle > 0 and prev_end_pos is not None:
            for i in range(transition_frames):
                s = (i + 1) / (transition_frames + 1)
                s = s ** 0.5
                x = prev_end_pos[0] + (start_pos[0] - prev_end_pos[0]) * s
                y = prev_end_pos[1] + (start_pos[1] - prev_end_pos[1]) * s
                z = prev_end_pos[2] + (start_pos[2] - prev_end_pos[2]) * s
                cycle_probe_positions.append([x, y, z])
                cycle_probe_sizes.append(current_probe_size)
            for i in range(transition_frames):
                for j in range(3):
                    all_sphere_positions[j].append(sphere_positions[j][i])
                all_hole_open.append(hole_open[i])
                all_sphere_sizes.append(sphere_size)
                all_hole_sizes.append(hole_size_scaled)
            all_probe_positions.extend(cycle_probe_positions)
            all_probe_sizes.extend(cycle_probe_sizes)

        frame_start = transition_frames if cycle > 0 else 0
        for frame in range(frame_start, n_f):
            if frame < pdrop:
                pos = list(start_pos)
                size = current_probe_size
            elif frame < pdrop + probe_drop_duration:
                idx = frame - pdrop
                pos = list(probe_path[idx])
                size = current_probe_size
            else:
                pos = list(end_pos)
                frames_since_pass = frame - (pdrop + probe_drop_duration)
                if frames_since_pass < n_growth_frames:
                    s = frames_since_pass / n_growth_frames
                    s = s ** 0.4
                    size = current_probe_size + (growth_target - current_probe_size) * s
                else:
                    size = growth_target
                current_probe_size = size

            for j in range(3):
                all_sphere_positions[j].append(sphere_positions[j][frame])
            all_hole_open.append(hole_open[frame])
            all_sphere_sizes.append(sphere_size)
            all_hole_sizes.append(hole_size_scaled)
            all_probe_positions.append(pos)
            all_probe_sizes.append(size)

        prev_end_pos = end_pos

    sphere_positions = [np.array(all_sphere_positions[i]) for i in range(3)]
    n_total = len(all_probe_positions)

    return sphere_positions, all_probe_positions, np.array(all_hole_open), all_probe_sizes, all_sphere_sizes, all_hole_sizes


def create_vacuum_animation_figure(
    n_frames: int = 120,
    orbit_radius: float = 6.0,
    sphere_radius: float = 0.8,
    probe_radius: float = 0.3,
    probe_start_offset: float = 2.0,
    n_cycles: int = 2,
    cycle_scale_factor: float = 1.8,
) -> "go.Figure | None":
    """
    三体問題＋プローブすり抜けのアニメーション Figure を生成する。

    赤・青・緑の球がぐるぐる回り、中心の穴に引っかかり、
    小さいプローブが Brachistochrone Curve に沿って吸い込まれるようにすり抜ける。

    n_cycles > 1 の場合: 大きくなった球がさらに大きな三つ巴を発見し、
    同様にすり抜けていくことを繰り返す。
    """
    if not HAS_PLOTLY:
        return None

    (sphere_positions, probe_positions, hole_open, probe_sizes, sphere_sizes, hole_sizes) = compute_three_body_animation_frames(
        n_frames=n_frames,
        orbit_radius=orbit_radius,
        probe_start_offset=probe_start_offset,
        sphere_radius=sphere_radius,
        probe_radius=probe_radius,
        use_collision_mode=True,
        n_cycles=n_cycles,
        cycle_scale_factor=cycle_scale_factor,
    )

    n_total = len(probe_positions)

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # 軸範囲: 複数サイクル時は最大スケールに合わせて拡大
    max_extent = max(10, orbit_radius * (cycle_scale_factor ** max(0, n_cycles - 1)) * 1.3)
    axis_range = [-max_extent, max_extent]

    # 等高線付きランドスケープ（大きな白球が下にあることを視覚化）
    def _add_landscape_contour(traces_list: list, max_ext: float) -> None:
        n_grid = 25
        xg = np.linspace(-max_ext, max_ext, n_grid)
        yg = np.linspace(-max_ext, max_ext, n_grid)
        X, Y = np.meshgrid(xg, yg)
        # 盆地形状: 中心が低く（z=-8）、外側ほど高い
        R2 = X ** 2 + Y ** 2
        Z = -8 + 0.04 * R2
        Z = np.clip(Z, -8, max_ext * 0.5)
        traces_list.append(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, "rgba(30,60,90,0.4)"], [0.5, "rgba(20,80,120,0.3)"], [1, "rgba(10,100,150,0.2)"]],
                opacity=0.35,
                showscale=False,
                contours=dict(
                    z=dict(show=True, usecolormap=False, project=dict(z=True), color="rgba(120,180,220,0.7)"),
                ),
                name="ランドスケープ（等高線）",
            )
        )

    # 各フレームのデータを構築
    frames = []
    for frame in range(n_total):
        traces_data = []

        # 等高線ランドスケープ（白球が下にあることを示す）
        _add_landscape_contour(traces_data, max_extent)

        # 三球（白球より大きく描く — 大きな三つ巴を発見）
        sphere_size = sphere_sizes[frame]
        for i in range(3):
            pos = sphere_positions[i][frame]
            traces_data.append(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers",
                    marker=dict(
                        size=sphere_size,
                        color=colors[i],
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    name=["赤", "青", "緑"][i],
                )
            )

        # 黄色の穴（特異点）- 隙が開いている時は強調。2度目以降は枠を広げて白球と同程度に
        hole_opacity = 0.95 if hole_open[frame] else 0.35
        hole_size = hole_sizes[frame]
        traces_data.append(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers",
                marker=dict(
                    size=hole_size,
                    color="gold",
                    symbol="diamond",
                    opacity=hole_opacity,
                    line=dict(width=2, color="yellow"),
                ),
                name="黄色の穴（隙の一瞬）",
            )
        )

        # プローブ（通過後に黄色の穴より巨大に膨らむ）
        probe_pos = probe_positions[frame]
        traces_data.append(
            go.Scatter3d(
                x=[probe_pos[0]],
                y=[probe_pos[1]],
                z=[probe_pos[2]],
                mode="markers",
                marker=dict(
                    size=probe_sizes[frame],
                    color="white",
                    opacity=0.85,
                    symbol="circle",
                    line=dict(width=2, color="cyan"),
                ),
                name="プローブ（通過後膨張）",
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
    _add_landscape_contour(initial_data, max_extent)
    for i in range(3):
        pos = sphere_positions[i][0]
        initial_data.append(
            go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers",
                marker=dict(
                    size=sphere_sizes[0],
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
            marker=dict(size=hole_sizes[0], color="gold", symbol="diamond", opacity=0.35),
            name="黄色の穴",
        )
    )
    probe_pos_0 = probe_positions[0]
    initial_data.append(
        go.Scatter3d(
            x=[probe_pos_0[0]], y=[probe_pos_0[1]], z=[probe_pos_0[2]],
            mode="markers",
            marker=dict(
                size=probe_sizes[0],
                color="white",
                opacity=0.85,
                symbol="circle",
                line=dict(width=2, color="cyan"),
            ),
            name="プローブ（通過後膨張）",
        )
    )

    fig = go.Figure(
        data=initial_data,
        frames=frames,
    )

    # スライダーと再生ボタン
    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(range=axis_range),
            yaxis=dict(range=axis_range),
            zaxis=dict(range=axis_range),
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


def compute_failure_animation_frames(
    n_frames: int = 120,
    orbit_radius: float = 6.0,
    sphere_radius: float = 0.8,
    probe_radius: float = 0.3,
    probe_start_offset: float = 2.0,
    collision_path_ratio: float = 0.45,
    blowaway_duration: int = 35,
) -> tuple:
    """
    失敗パターン: 白球が三球に衝突し、カーリングのように吹き飛ばされる。

    ルートが開いていない時（三球が離れていない時）に下降を試み、
    衝突して怪我をし、吹き飛ばされる。
    """
    sphere_positions, hole_open = _simulate_orbital_with_collisions(
        n_frames=n_frames,
        sphere_radius=sphere_radius,
        hole_radius=1.5,
        basin_radius=orbit_radius,
        dt=0.15,
        seed=42,
    )

    # ルートが開いていないフレームを選ぶ（失敗条件）
    closed_frames = np.where(~hole_open)[0]
    pdrop = int(n_frames * 0.35) if len(closed_frames) == 0 else closed_frames[len(closed_frames) // 2]

    start_pos = (probe_start_offset, 0.0, 5.0)
    end_pos = (0.0, 0.0, -5.0)
    probe_path = _brachistochrone_path(
        np.linspace(0, 1, 21),
        start_pos,
        end_pos,
    )

    # 衝突点: パスの途中
    n_path = len(probe_path)
    collision_idx = int(n_path * collision_path_ratio)
    collision_pos = probe_path[collision_idx]

    # 吹き飛び軌道: カーリングのように弧を描いて飛ぶ
    t_blow = np.linspace(0, 1, blowaway_duration)
    blow_x = collision_pos[0] + 6 * t_blow + 1.5 * np.sin(t_blow * np.pi)
    blow_y = collision_pos[1] + 4 * t_blow - 2 * (1 - t_blow) ** 2
    blow_z = collision_pos[2] - 3 * t_blow - 2 * t_blow ** 2
    blow_path = np.column_stack([blow_x, blow_y, blow_z])

    n_before = collision_idx + 1
    n_after = blowaway_duration

    probe_positions = []
    probe_sizes = []
    base_size = probe_radius * 15

    for frame in range(n_frames):
        if frame < pdrop:
            probe_positions.append(list(start_pos))
            probe_sizes.append(base_size)
        elif frame < pdrop + n_before:
            idx = frame - pdrop
            probe_positions.append(list(probe_path[idx]))
            probe_sizes.append(base_size)
        elif frame < pdrop + n_before + n_after:
            idx = frame - pdrop - n_before
            probe_positions.append(list(blow_path[idx]))
            probe_sizes.append(base_size * 0.9)
        else:
            probe_positions.append(list(blow_path[-1]))
            probe_sizes.append(base_size * 0.9)

    sphere_sizes = [sphere_radius * 15] * n_frames
    hole_sizes = [14] * n_frames

    return sphere_positions, probe_positions, hole_open, probe_sizes, sphere_sizes, hole_sizes


def create_failure_animation_figure(
    n_frames: int = 120,
    orbit_radius: float = 6.0,
    sphere_radius: float = 0.8,
    probe_radius: float = 0.3,
    probe_start_offset: float = 2.0,
) -> "go.Figure | None":
    """
    失敗パターン: 白球が三球に衝突し、怪我をしてカーリングのように吹き飛ばされる。
    """
    if not HAS_PLOTLY:
        return None

    (sphere_positions, probe_positions, hole_open, probe_sizes, sphere_sizes, hole_sizes) = (
        compute_failure_animation_frames(
            n_frames=n_frames,
            orbit_radius=orbit_radius,
            sphere_radius=sphere_radius,
            probe_radius=probe_radius,
            probe_start_offset=probe_start_offset,
        )
    )

    n_total = len(probe_positions)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    max_extent = max(10, orbit_radius * 1.3)
    axis_range = [-max_extent, max_extent]

    frames = []
    for frame in range(n_total):
        traces_data = []

        for i in range(3):
            pos = sphere_positions[i][frame]
            traces_data.append(
                go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode="markers",
                    marker=dict(
                        size=sphere_sizes[frame],
                        color=colors[i],
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    name=["赤", "青", "緑"][i],
                )
            )

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
                name="黄色の穴",
            )
        )

        probe_pos = probe_positions[frame]
        traces_data.append(
            go.Scatter3d(
                x=[probe_pos[0]], y=[probe_pos[1]], z=[probe_pos[2]],
                mode="markers",
                marker=dict(
                    size=probe_sizes[frame],
                    color="white",
                    opacity=0.85,
                    symbol="circle",
                    line=dict(width=2, color="red"),
                ),
                name="白球（衝突で吹き飛び）",
            )
        )

        frames.append(go.Frame(data=traces_data, name=str(frame)))

    initial_data = []
    for i in range(3):
        pos = sphere_positions[i][0]
        initial_data.append(
            go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers",
                marker=dict(size=sphere_sizes[0], color=colors[i], opacity=0.7, line=dict(width=1, color="white")),
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
    initial_data.append(
        go.Scatter3d(
            x=[probe_positions[0][0]], y=[probe_positions[0][1]], z=[probe_positions[0][2]],
            mode="markers",
            marker=dict(size=probe_sizes[0], color="white", opacity=0.85, symbol="circle", line=dict(width=2, color="red")),
            name="白球",
        )
    )

    fig = go.Figure(data=initial_data, frames=frames)
    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(range=axis_range),
            yaxis=dict(range=axis_range),
            zaxis=dict(range=axis_range),
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
                    dict(label="▶ 再生", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ 停止", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
                x=0.1, y=0,
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=[dict(args=[[str(f)], dict(frame=dict(duration=0), mode="immediate")], label=str(f), method="animate") for f in range(n_total)],
                x=0.1, len=0.9, xanchor="left", y=0,
                currentvalue=dict(visible=True, prefix="フレーム: ", xanchor="center"),
            ),
        ],
    )

    return fig
