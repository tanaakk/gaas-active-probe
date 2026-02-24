#!/bin/bash
# GAP Streamlit アプリのローカルプレビュー起動スクリプト
set -e
cd "$(dirname "$0")/.."

echo "🔮 GAP — ローカルプレビュー"
echo ""

# 仮想環境がなければ作成
if [ ! -d ".venv" ]; then
  echo "仮想環境を作成しています..."
  python3 -m venv .venv
fi

# 仮想環境を有効化して依存関係をインストール
echo "依存関係をインストールしています..."
.venv/bin/pip install -q -r requirements.txt

# Streamlit を起動
echo ""
echo "✅ 起動中... ブラウザで http://localhost:8501 を開いてください"
echo "   終了するには Ctrl+C"
echo ""
.venv/bin/streamlit run app.py --server.port 8501
