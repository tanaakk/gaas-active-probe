# GAP — 無料ホスティングで公開する方法

GAAS Active Probe の Streamlit シミュレーションを無料で公開できる主な選択肢です。

---

## 比較表

| サービス | 無料枠 | 設定の簡単さ | 備考 |
|----------|--------|-------------|------|
| **Streamlit Community Cloud** | 無制限（非商用） | ⭐⭐⭐ 最易 | Streamlit 公式、GitHub 連携 |
| **Hugging Face Spaces** | 無制限 | ⭐⭐⭐ 簡単 | ML コミュニティで人気 |
| **Render** | 750時間/月 | ⭐⭐ 普通 | スリープあり（起動に遅延） |

---

## 1. Streamlit Community Cloud（推奨）

**完全無料**。Streamlit 公式のホスティングで、GitHub と連携するだけでデプロイできます。

### 手順

1. **GitHub にリポジトリを push**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **[share.streamlit.io](https://share.streamlit.io)** にアクセス

3. **Sign in with GitHub** でログイン

4. **New app** をクリック
   - **Repository**: `tanaakk/gaas-active-probe`（あなたのリポジトリ）
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `gaas-active-probe`（任意）

5. **Deploy** をクリック

数分で `https://gaas-active-probe.streamlit.app` のような URL で公開されます。

### 注意点

- 非商用利用が前提
- 一定時間アクセスがないとスリープする場合あり
- `requirements.txt` が自動で読み込まれる

---

## 2. Hugging Face Spaces

**完全無料**。AI/ML コミュニティで広く使われています。

### 手順

1. **[huggingface.co](https://huggingface.co)** でアカウント作成

2. **New Space** を作成
   - **Space name**: `gaas-active-probe`
   - **License**: MIT
   - **SDK**: **Streamlit** を選択

3. **Space のファイルをアップロード**

   `README.md` を以下の内容で作成:
   ```yaml
   ---
   title: GAAS Active Probe
   emoji: 🔮
   sdk: streamlit
   sdk_version: "1.28.0"
   app_file: app.py
   pinned: false
   ---
   ```

   または、GitHub リポジトリを連携して同期。

4. 必要なファイルをアップロード:
   - `app.py`
   - `requirements.txt`
   - `gap/` フォルダ一式

5. Space がビルドされると自動で公開

   URL: `https://huggingface.co/spaces/<username>/gaas-active-probe`

### 注意点

- 初回ビルドに数分かかる
- ストレージ制限あり（無料枠で十分）

---

## 3. Render

**無料枠**: 750 時間/月。スリープあり（約 15 分無アクセスで停止、再起動に 30 秒〜1 分）。

### 手順

1. **[render.com](https://render.com)** でアカウント作成

2. **New → Web Service**

3. GitHub リポジトリを接続

4. 設定:
   - **Name**: `gaas-active-probe`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Instance Type**: Free

5. **Create Web Service**

### 注意点

- 無料枠はスリープするため、アクセス時に起動待ちが発生
- `PORT` 環境変数が Render から自動付与される

---

## 4. その他の選択肢

| サービス | 備考 |
|----------|------|
| **Railway** | 月 $5 分の無料枠、クレジット消費 |
| **Fly.io** | 小規模なら無料枠内 |
| **Google Cloud Run** | 月 200 万リクエストまで無料（Blaze プラン必要） |

---

## 事前準備（共通）

### requirements.txt の確認

プロジェクトの `requirements.txt` に以下が含まれていることを確認:

```
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.0.0
streamlit>=1.28.0
```

### .streamlit/config.toml（オプション）

公開用の設定を追加する場合:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true
```

---

## 推奨

**まずは Streamlit Community Cloud を試す**のが最も簡単です。GitHub に push するだけで数分で公開できます。
