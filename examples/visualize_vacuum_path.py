#!/usr/bin/env python3
"""
GAP 可視化例 — 負圧ポイントと Vacuum Path の点群グラフ

三つ巴モデルによる質量の干渉と、特異点（量子トンネル）の可視化。
Plotly によるインタラクティブ3D表示。

実行: python examples/visualize_vacuum_path.py
"""

import sys
from gap.constants import Layer
from gap.visualization import create_vacuum_figure

try:
    import plotly
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def main():
    print("GAP 負圧ポイント可視化")
    print("三つ巴モデル: 拮抗した三球の干渉による特異点を可視化")
    print()

    fig = create_vacuum_figure(
        masses=(1e10, 5e10, 1e10),
        positions=((6, 0, 0), (-6, 6, 0), (-6, -6, 0)),
        probe_position=(0, 0, 3),
        layer=Layer.MEDIUM,
    )

    if fig is None:
        print("Plotly が必要です: pip install plotly")
        return

    output_file = "vacuum_path_visualization.html"
    fig.write_html(output_file)
    print(f"可視化を保存しました: {output_file}")
    print("ブラウザで開いて確認できます。")

    if "--show" in sys.argv:
        fig.show()


if __name__ == "__main__":
    main()
