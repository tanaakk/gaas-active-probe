#!/usr/bin/env python3
"""
GAP — GAAS Active Probe / Streamlit 可視化アプリケーション

GAASActiveProbe のシミュレーション結果を Plotly 3D 点群グラフで表示する。
三つ巴モデルによる質量の干渉と特異点（負圧ポイント）の可視化。

実行: streamlit run app.py
"""

import streamlit as st
from gap.constants import Layer
from gap.visualization import create_vacuum_figure, create_vacuum_animation_figure

st.set_page_config(
    page_title="GAP — GAAS Active Probe",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 GAP — GAAS Active Probe")
st.caption("三つ巴モデルと負圧ポイントの可視化 | わらしべ長者の幾何学的拡張")

# 表示モード
view_mode = st.sidebar.radio(
    "表示モード",
    ["静的（パラメータ調整）", "アニメーション（三体問題・すり抜け）"],
    index=1,
    horizontal=True,
)

st.sidebar.header("パラメータ設定")

# 質量スケールの選択
layer_option = st.sidebar.selectbox(
    "質量階層 (Layer)",
    options=list(Layer),
    format_func=lambda x: {
        Layer.MICRO: "MICRO (10^6〜10^8)",
        Layer.MEDIUM: "MEDIUM (10^9〜10^10)",
        Layer.MACRO: "MACRO (10^11〜10^13)",
        Layer.GLOBAL: "GLOBAL (10^14〜10^17)",
    }.get(x, x.name),
    index=1,
)

# 三球の質量（スライダーで 10^6 〜 10^13 の範囲）
st.sidebar.subheader("三球の質量（拮抗させる）")
mass_scale = st.sidebar.slider(
    "質量スケール（対数）",
    min_value=6,
    max_value=13,
    value=10,
    help="10^N の N",
)

m1 = st.sidebar.number_input(
    "球1の質量係数",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    format="%.1f",
)
m2 = st.sidebar.number_input(
    "球2の質量係数",
    min_value=0.1,
    max_value=10.0,
    value=5.0,
    step=0.1,
    format="%.1f",
)
m3 = st.sidebar.number_input(
    "球3の質量係数",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    format="%.1f",
)

base = 10 ** mass_scale
masses = (m1 * base, m2 * base, m3 * base)

# 三球の位置
st.sidebar.subheader("三球の位置 (X, Y, Z)")
pos1 = (
    st.sidebar.slider("球1 X", -15, 15, 6),
    st.sidebar.slider("球1 Y", -15, 15, 0),
    st.sidebar.slider("球1 Z", -10, 10, 0),
)
pos2 = (
    st.sidebar.slider("球2 X", -15, 15, -6),
    st.sidebar.slider("球2 Y", -15, 15, 6),
    st.sidebar.slider("球2 Z", -10, 10, 0),
)
pos3 = (
    st.sidebar.slider("球3 X", -15, 15, -6),
    st.sidebar.slider("球3 Y", -15, 15, -6),
    st.sidebar.slider("球3 Z", -10, 10, 0),
)
positions = (pos1, pos2, pos3)

# プローブ位置
st.sidebar.subheader("プローブ位置")
probe_position = (
    st.sidebar.slider("プローブ X", -15, 15, 0),
    st.sidebar.slider("プローブ Y", -15, 15, 0),
    st.sidebar.slider("プローブ Z", -15, 15, 3),
)

# 解像度
grid_resolution = st.sidebar.slider(
    "ポテンシャル場の解像度",
    min_value=15,
    max_value=40,
    value=25,
)
relative_percentile = st.sidebar.slider(
    "特異点検出の相対百分位 (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
)

if view_mode == "アニメーション（三体問題・すり抜け）":
    st.sidebar.subheader("アニメーション設定")
    n_frames = st.sidebar.slider("フレーム数", 60, 180, 120)
    orbit_radius = st.sidebar.slider("軌道半径", 3.0, 10.0, 6.0, 0.5)
    probe_start_offset = st.sidebar.slider("プローブ開始位置（中心からの距離）", 0.5, 5.0, 2.0, 0.5)

    with st.spinner("アニメーションを生成中..."):
        fig = create_vacuum_animation_figure(
            n_frames=n_frames,
            orbit_radius=orbit_radius,
            probe_start_offset=probe_start_offset,
        )

    if fig is not None:
        st.markdown("""
        **三体問題アニメーション** — 赤・青・緑の球が**ぐるぐる回転**しながら**ぶつかり合う**。
        **黄色の穴**には入れず、互いに押し合う三つ巴。その**隙の一瞬**を、白い球が潜り抜けていく。
        """)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Plotly がインストールされていません。")
else:
    # 静的表示
    with st.spinner("シミュレーション実行中..."):
        fig = create_vacuum_figure(
            masses=masses,
            positions=positions,
            probe_position=probe_position,
            layer=layer_option,
            grid_resolution=grid_resolution,
            relative_percentile=relative_percentile,
        )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Plotly がインストールされていません。`pip install plotly` を実行してください。")

st.sidebar.markdown("---")
st.sidebar.markdown("### 凡例")
if view_mode == "アニメーション（三体問題・すり抜け）":
    st.sidebar.markdown("- 🔴 **赤球** — 三つ巴の1")
    st.sidebar.markdown("- 🔵 **青球** — 三つ巴の2")
    st.sidebar.markdown("- 🟢 **緑球** — 三つ巴の3")
    st.sidebar.markdown("- 💛 **黄色の穴** — 入れない（明るい=隙が開いている）")
    st.sidebar.markdown("- ⬜ **白球** — 隙の一瞬に潜り抜ける")
else:
    st.sidebar.markdown("- 🔴🟢🔵 **質量球体** — 三つ巴の球")
    st.sidebar.markdown("- 💠 **青系点群** — 負圧領域")
    st.sidebar.markdown("- 💎 **金色** — 特異点（量子トンネル）")
    st.sidebar.markdown("- 🔵 **シアン線** — Vacuum Path（空いたら通る）")
    st.sidebar.markdown("- ⬜ **白×** — プローブ位置")
