#!/usr/bin/env python3
"""
GAP — GAAS Active Probe / Streamlit 可視化アプリケーション

GAASActiveProbe のシミュレーション結果を Plotly 3D 点群グラフで表示する。
三つ巴モデルによる質量の干渉と特異点（負圧ポイント）の可視化。

実行: streamlit run app.py
"""

import streamlit as st
from gap.constants import Layer
from gap.visualization import (
    create_vacuum_figure,
    create_vacuum_animation_figure,
    create_base_establishment_figure,
    create_failure_animation_figure,
)

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
    [
        "HITSCAN/HITPLAN/HITSERIES（→ モニタリング）",
        "GAAS zero cost breakthrough（→ すり抜け）",
        "失敗パターン（衝突・吹き飛び）",
        "静的（パラメータ調整）",
    ],
    index=0,
    horizontal=False,
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

if view_mode == "HITSCAN/HITPLAN/HITSERIES（→ モニタリング）":
    st.markdown("""
    **HITSCAN/HITPLAN/HITSERIES（→ モニタリング）**

    ３つの球と白い球が最初にある。白い球と３つの球は**最初は接続なし**から始まり、順次:

    1. **HITSCAN** — 白い球が**一つずつ**、赤・青・緑の順に接続。赤→青→緑の順でパルスが増えていく。
    2. **HITPLAN** — 白い球と接続された**３つ巴の回転によるマッピング**。神経接続が安定するにつれ、軌道円の不透明度を徐々に上げ、回転によるマッピングが描かれる。
    3. **HITSERIES CICD** — 三つの超重量を三つ巴で干渉させ続けると、空間が摩耗し、最後に**穴が自然に開く**。

    **学習が済むと等高線が現れる**。▶ 再生でアニメーション表示。
    """)
    with st.spinner("可視化を生成中..."):
        fig = create_base_establishment_figure(
            positions=positions,
            probe_position=probe_position,
        )
    if fig is not None:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
    else:
        st.error("Plotly がインストールされていません。")
elif view_mode == "GAAS zero cost breakthrough（→ すり抜け）":
    st.sidebar.subheader("アニメーション設定")
    n_frames = st.sidebar.slider("フレーム数（1サイクルあたり）", 60, 180, 120)
    n_cycles = st.sidebar.slider("繰り返しサイクル数", 1, 4, 2, help="大きくなった球がさらに大きな三つ巴を発見し、同様にすり抜けていく回数")
    orbit_radius = st.sidebar.slider("軌道半径", 3.0, 10.0, 6.0, 0.5)
    cycle_scale_factor = st.sidebar.slider("サイクルごとのスケール倍率", 1.2, 2.5, 1.8, 0.1)
    probe_start_offset = st.sidebar.slider("プローブ開始位置（中心からの距離）", 0.5, 5.0, 2.0, 0.5)

    with st.spinner("アニメーションを生成中..."):
        fig = create_vacuum_animation_figure(
            n_frames=n_frames,
            orbit_radius=orbit_radius,
            probe_start_offset=probe_start_offset,
            n_cycles=n_cycles,
            cycle_scale_factor=cycle_scale_factor,
        )

    if fig is not None:
        st.markdown("""
        **GAAS zero cost breakthrough** — イノベーションとは、巨大な鉄球を三つ巴にさせて空間を摩耗させ、
        空いたところをすり抜けるゲームである。漏斗型の領域は凸凹しているため、三つの玉は**近づいたり離れたり**しながら、
        穴には入れない状態が続く（スロープトイのように並行ではない）。その隙の一瞬を、白い球が潜り抜けていく。
        すり抜ける瞬間は**負圧**を用いるため、コストが**ゼロあるいはマイナス**になる。
        白い球はすり抜けた後に**低い位置**で栄養を急激に蓄え、上にあった三つ巴の球の穴よりも**大きくなる**。
        大きくなった球はさらに**大きな三つ巴**を発見し、同様にすり抜けていくことを**繰り返す**。

        なお、白い球に**連星**になった質量をもつ球も白い球にくっついて穴をすり抜け、巨大化**することがある**。
        """)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
    else:
        st.error("Plotly がインストールされていません。")
elif view_mode == "失敗パターン（衝突・吹き飛び）":
    st.sidebar.subheader("失敗パターン設定")
    n_frames_fail = st.sidebar.slider("フレーム数", 60, 180, 120)
    orbit_radius_fail = st.sidebar.slider("軌道半径", 3.0, 10.0, 6.0, 0.5)
    probe_start_offset_fail = st.sidebar.slider("プローブ開始位置", 0.5, 5.0, 2.0, 0.5)

    with st.spinner("失敗パターンを生成中..."):
        fig = create_failure_animation_figure(
            n_frames=n_frames_fail,
            orbit_radius=orbit_radius_fail,
            probe_start_offset=probe_start_offset_fail,
        )

    if fig is not None:
        st.markdown("""
        **失敗パターン** — 白い球が下に抜ける時に三つの球に衝突すると**怪我**をしてしまい、
        **カーリングのように吹き飛ばされてしまう**。

        ルートが開いていない時（三球が離れていない時）に下降を試みると発生する。
        """)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
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
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
    else:
        st.error("Plotly がインストールされていません。`pip install plotly` を実行してください。")

st.sidebar.markdown("---")
st.sidebar.markdown("### 凡例")
if view_mode == "HITSCAN/HITPLAN/HITSERIES（→ モニタリング）":
    st.sidebar.markdown("- ⬜ **白い球** — プローブ（接続なしから始まる）")
    st.sidebar.markdown("- 📡 **赤・青・緑の点線** — HITSCAN（白い球が赤→青→緑と一つずつ接続）")
    st.sidebar.markdown("- 🔗 **実線＋軌道円** — HITPLAN（白い球と接続された３つ巴の回転によるマッピング）")
    st.sidebar.markdown("- 🌐 **半透明面** — HITSERIES（形状観測・継続学習）")
    st.sidebar.markdown("- 💛 **下部の穴** — 三つ巴干渉で空間が摩耗し、最後に自然に開く")
    st.sidebar.markdown("- 📐 **等高線** — 学習完了後に現れるランドスケープ")
elif view_mode == "GAAS zero cost breakthrough（→ すり抜け）":
    st.sidebar.markdown("- 🔴 **赤球** — 三つ巴の1")
    st.sidebar.markdown("- 🔵 **青球** — 三つ巴の2")
    st.sidebar.markdown("- 🟢 **緑球** — 三つ巴の3")
    st.sidebar.markdown("- 💛 **黄色の穴** — 入れない（明るい=隙が開いている）")
    st.sidebar.markdown("- ⬜ **白球** — 隙の一瞬に潜り抜け、低い位置で栄養を急激に蓄え、黄色の穴より大きくなる。繰り返しでさらに大きな三つ巴をすり抜ける。連星の球もくっついてすり抜け巨大化することがある")
elif view_mode == "失敗パターン（衝突・吹き飛び）":
    st.sidebar.markdown("- 🔴 **赤球** — 三つ巴の1")
    st.sidebar.markdown("- 🔵 **青球** — 三つ巴の2")
    st.sidebar.markdown("- 🟢 **緑球** — 三つ巴の3")
    st.sidebar.markdown("- 💛 **黄色の穴** — ルートが開いていない時は暗い")
    st.sidebar.markdown("- ⬜ **白球（赤枠）** — 衝突で怪我をし、カーリングのように吹き飛ばされる")
else:
    st.sidebar.markdown("- 🔴🟢🔵 **質量球体** — 三つ巴の球")
    st.sidebar.markdown("- 💠 **青系点群** — 負圧領域")
    st.sidebar.markdown("- 💎 **金色** — 特異点（量子トンネル）")
    st.sidebar.markdown("- 🔵 **シアン線** — Vacuum Path（空いたら通る）")
    st.sidebar.markdown("- ⬜ **白×** — プローブ位置")
