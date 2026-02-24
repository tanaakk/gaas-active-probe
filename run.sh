#!/bin/bash
# GAP Streamlit アプリのローカル起動スクリプト
cd "$(dirname "$0")"

if ! python3 -c "import streamlit" 2>/dev/null; then
  echo "依存関係をインストールしています..."
  pip install -r requirements.txt
fi

echo "Streamlit を起動しています..."
python3 -m streamlit run app.py --server.port 8501
