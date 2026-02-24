# GAP — Firebase Hosting デプロイ手順

GAAS Active Probe の Streamlit アプリケーションを Firebase Hosting 経由で Cloud Run にデプロイする手順です。

## ⚠️ Firebase App Hosting と Firebase Hosting の違い

| サービス | 対応フレームワーク | GAP での利用 |
|----------|-------------------|--------------|
| **Firebase App Hosting** | Next.js, Angular のみ | ❌ Streamlit 非対応 |
| **Firebase Hosting + Cloud Run** | 任意（Docker） | ✅ 本プロジェクトで使用 |

[Firebase App Hosting](https://console.firebase.google.com/project/gaas-active-probe/apphosting) は Next.js/Angular 向けです。**Streamlit は Firebase Hosting（従来型）と Cloud Run の組み合わせ**でデプロイします。

---

## アーキテクチャ

```
[Firebase Hosting] → (rewrite) → [Cloud Run: Streamlit アプリ]
```

Firebase Hosting がリクエストを Cloud Run 上の Streamlit コンテナに転送します。

---

## 前提条件

- Google Cloud アカウント
- Firebase プロジェクト（GCP プロジェクトとリンク済み）
- Node.js（Firebase CLI 用）
- Docker（ローカルビルド時）

---

## 1. 初回セットアップ

### 1.1 Firebase プロジェクト `gaas-active-probe` の作成

1. [Firebase Console](https://console.firebase.google.com/) にアクセス
2. **プロジェクトを追加** → プロジェクト ID を `gaas-active-probe` に設定
3. **Blaze 料金プラン**にアップグレード（Cloud Run 連携に必要）
4. プロジェクト作成後、左メニュー **Hosting**（※ App Hosting ではない）を開く

> **注意**: [App Hosting](https://console.firebase.google.com/project/gaas-active-probe/apphosting) は Next.js/Angular 専用のため使用しません。**Hosting** で Cloud Run へのリライトを設定します。

### 1.2 GCP プロジェクトの準備

```bash
# プロジェクト ID を設定（Firebase と同一）
export PROJECT_ID=gaas-active-probe
gcloud config set project $PROJECT_ID

# 必要な API を有効化
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable firebasehosting.googleapis.com
```

### 1.3 Firebase CLI の連結

```bash
# Firebase CLI をインストール
npm install -g firebase-tools

# ログイン
firebase login

# プロジェクトを選択
firebase use gaas-active-probe
```

### 1.3 Cloud Run と Firebase Hosting の接続

Firebase Hosting から Cloud Run へのリライトを使用するには、両方が同じ GCP プロジェクト内にある必要があります。`firebase.json` の `serviceId` が Cloud Run のサービス名と一致していることを確認してください。

---

## 2. ローカルでの動作確認

```bash
# 依存関係をインストール
pip install -r requirements.txt

# Streamlit を起動
streamlit run app.py
```

ブラウザで http://localhost:8501 を開いて動作を確認します。

---

## 3. デプロイ方法

### 方法 A: 手動デプロイ（推奨：初回）

#### Step 1: Docker イメージをビルドして Cloud Run にデプロイ

```bash
# Artifact Registry にリポジトリを作成（初回のみ）
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format=docker \
  --location=asia-northeast1

# ビルド＆デプロイ（Cloud Build 使用）
gcloud run deploy gaas-active-probe \
  --source . \
  --region asia-northeast1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8501
```

`--source .` により、Dockerfile が自動でビルドされ、Cloud Run にデプロイされます。

#### Step 2: Firebase Hosting をデプロイ

```bash
firebase deploy --only hosting
```

これで `https://<project-id>.web.app` からアプリにアクセスできます。

---

### 方法 B: GitHub Actions による自動デプロイ

`main` ブランチへの push で自動デプロイされます。

#### 必要な GitHub Secrets

| Secret 名 | 説明 |
|-----------|------|
| `GCP_PROJECT_ID` | GCP プロジェクト ID |
| `GCP_SA_KEY` | サービスアカウントの JSON キー（Cloud Run デプロイ権限） |
| `FIREBASE_TOKEN` | `firebase login:ci` で取得したトークン |

#### Secrets の設定手順

**1. GCP サービスアカウントの作成**

```bash
# サービスアカウント作成
gcloud iam service-accounts create gaas-deployer \
  --display-name="GAP Deployer"

# 必要なロールを付与
export SA_EMAIL=gaas-deployer@$PROJECT_ID.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.admin"

# JSON キーを生成
gcloud iam service-accounts keys create key.json \
  --iam-account=$SA_EMAIL
```

**2. Firebase CI トークンの取得**

```bash
firebase login:ci
# 表示されたトークンをコピー
```

**3. GitHub に Secrets を登録**

- リポジトリ → Settings → Secrets and variables → Actions
- `GCP_PROJECT_ID`: プロジェクト ID
- `GCP_SA_KEY`: `key.json` の内容（全文）
- `FIREBASE_TOKEN`: `firebase login:ci` の出力

---

### 方法 C: Docker のみで Cloud Run にデプロイ（Firebase なし）

Firebase Hosting を使わず、Cloud Run の URL のみで公開する場合:

```bash
gcloud run deploy gaas-active-probe \
  --source . \
  --region asia-northeast1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8501
```

デプロイ後、`https://gaas-active-probe-xxxxx-an.a.run.app` のような URL が発行されます。

---

## 4. カスタムドメインの設定（オプション）

Firebase Hosting でカスタムドメインを設定する場合:

```bash
firebase hosting:sites:list
firebase target:apply hosting default <site-id>
```

Firebase Console の Hosting 設定から、カスタムドメインを追加できます。

---

## 5. トラブルシューティング

### Cloud Run のサービスが起動しない

- ログを確認: `gcloud run services logs read gaas-active-probe --region asia-northeast1`
- ポート 8501 が正しく公開されているか確認
- メモリ不足の場合は `--memory 1Gi` を追加

### Firebase Hosting から 502 Bad Gateway

- Cloud Run サービスが正常に起動しているか確認
- `firebase.json` の `serviceId` が Cloud Run のサービス名と一致しているか確認
- リージョンが `asia-northeast1` で一致しているか確認

### firebase-admin のインポートエラー

`app.py` では firebase-admin を使用していません。Firebase 機能を追加する場合のみ必要です。不要な場合は `requirements.txt` から削除してビルドを軽量化できます。

---

## 6. ファイル構成

```
gaas-active-probe/
├── app.py                 # Streamlit アプリ
├── Dockerfile             # コンテナ定義
├── firebase.json          # Firebase Hosting 設定
├── requirements.txt       # Python 依存
├── public/                # Firebase Hosting 用（rewrite 時は未使用）
│   └── index.html
├── .github/workflows/
│   └── deploy.yml         # GitHub Actions
└── DEPLOY.md             # 本ドキュメント
```
