# Workflow-CI

Repository publik terpisah untuk Kriteria 3 Dicoding **Membangun Sistem Machine Learning**.

## Kemampuan workflow

1. Menjalankan unit test dan `mlflow run` pada setiap push/PR/manual trigger.
2. Menyimpan model, metrics, run summary, dan local MLflow store sebagai GitHub Actions artifact.
3. Membangun serving image menggunakan `mlflow models build-docker`.
4. Login dan push tag `latest` serta commit SHA ke Docker Hub hanya dari branch `main` atau manual run.

Flag `--env-manager local` pada `mlflow models build-docker` sengaja dipakai agar image mengikuti versi Python yang tercatat pada model. Tanpa flag ini, jalur virtualenv bawaan MLflow 2.19 memakai bootstrap Python 3.9 lama yang sudah tidak kompatibel dengan endpoint `get-pip.py` terkini.

## Sebelum menjalankan di GitHub

Tambahkan repository secrets berikut:

- `DOCKERHUB_USERNAME` — username Docker Hub.
- `DOCKERHUB_TOKEN` — access token Docker Hub, bukan password akun.

Workflow memakai repository image `bank-marketing-mlflow`. URL image yang sudah dipublikasikan dicatat pada `MLProject/DockerHub_link.txt`.

## Uji lokal

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\MLProject\requirements.txt
$env:MLFLOW_TRACKING_URI = "file:$((Resolve-Path .).Path)/mlruns"
mlflow run .\MLProject --env-manager local --experiment-name bank-marketing-ci -P data_dir=.\MLProject\bank_preprocessing
```

Token, `.env`, dan isi lokal `mlruns/` dikecualikan dari repository.
