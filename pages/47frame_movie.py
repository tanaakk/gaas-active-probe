"""
47フレームムービー — 三つ巴と白い球の点線接続

フレーム5で赤、10で青、15で緑が白い球と点線で接続。
接続したまま三つ巴が回転し続ける。
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

N_FRAMES = 47
ORBIT_RADIUS = 6.0
ORBIT_SPEED = 0.15
BASE_PHASES = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
COLORS = ["#e74c3c", "#3498db", "#2ecc71"]  # 赤, 青, 緑
WHITE_BALL_POS = (0.0, 0.0, 5.0)


def get_orbital_positions(frame: int):
    """フレームに応じた三つ巴の軌道座標"""
    angles = BASE_PHASES + frame * ORBIT_SPEED
    return [
        (ORBIT_RADIUS * np.cos(angles[i]), ORBIT_RADIUS * np.sin(angles[i]), 0.0)
        for i in range(3)
    ]


def make_dashed_line(start, end, n_pts=60, dash_len=4, gap_len=2):
    """点線用の座標配列（NaNで区切り）"""
    t = np.linspace(0, 1, n_pts)
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    z = start[2] + t * (end[2] - start[2])
    x_dash, y_dash, z_dash = [], [], []
    k = 0
    while k < n_pts:
        for _ in range(dash_len):
            if k < n_pts:
                x_dash.append(float(x[k]))
                y_dash.append(float(y[k]))
                z_dash.append(float(z[k]))
                k += 1
        if k < n_pts:
            x_dash.append(np.nan)
            y_dash.append(np.nan)
            z_dash.append(np.nan)
            k += gap_len
    return x_dash, y_dash, z_dash


st.set_page_config(page_title="47フレームムービー", page_icon="🎬", layout="wide")
st.title("🎬 47フレームムービー")
st.caption("三つ巴が回転し、白い球と点線で順に接続")

# スライダー（47フレーム: 0〜46）
frame = st.slider("フレーム", 0, N_FRAMES - 1, value=0)

# 2. 現在フレームに応じて表示する接続線を決定
active_lines = set()
if frame >= 5:
    active_lines.add("red")
if frame >= 10:
    active_lines.add("blue")
if frame >= 15:
    active_lines.add("green")

# 3. 現在フレームの球の座標を取得（回転に追従）
pos_list = get_orbital_positions(frame)
line_configs = {
    "red": {"color": COLORS[0], "target_idx": 0},
    "blue": {"color": COLORS[1], "target_idx": 1},
    "green": {"color": COLORS[2], "target_idx": 2},
}

fig = go.Figure()

# --- 赤・青・緑の球（三つ巴）---
for i, pos in enumerate(pos_list):
    fig.add_trace(
        go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode="markers",
            marker=dict(
                size=12,
                color=COLORS[i],
                opacity=0.8,
                line=dict(width=2, color="white"),
            ),
            name=["赤", "青", "緑"][i],
        )
    )

# --- 白い球（プローブ）— 常に表示 ---
fig.add_trace(
    go.Scatter3d(
        x=[WHITE_BALL_POS[0]],
        y=[WHITE_BALL_POS[1]],
        z=[WHITE_BALL_POS[2]],
        mode="markers",
        marker=dict(
            size=14,
            color="white",
            symbol="circle",
            line=dict(width=2, color="white"),
            opacity=1,
        ),
        name="白い球（プローブ）",
    )
)

# --- 3. 点線を描画（回転する球の現在位置に接続）---
for color in active_lines:
    config = line_configs[color]
    target_pos = pos_list[config["target_idx"]]
    x_d, y_d, z_d = make_dashed_line(WHITE_BALL_POS, target_pos)
    fig.add_trace(
        go.Scatter3d(
            x=x_d,
            y=y_d,
            z=z_d,
            mode="lines",
            line=dict(color=config["color"], width=5),
            opacity=1.0,
            name=f"{color} 接続",
        )
    )

fig.update_layout(
    showlegend=True,
    scene=dict(
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
        zaxis=dict(range=[-5, 10]),
        aspectmode="cube",
        bgcolor="rgba(20,20,30,1)",
    ),
    paper_bgcolor="rgba(20,20,30,1)",
    font=dict(color="white"),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**フレーム進行:**
- **5** — 赤い点線で白い球と赤い球が接続
- **10** — 青い点線が追加
- **15** — 緑の点線が追加

接続したまま三つ巴が回転し続けます。
""")
