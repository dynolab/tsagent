# tsagent

Code & notes on the Time Series Agent project

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n myenv python=3.12
conda activate myenv
```
2. Install necessary packages
```bash
pip install -r requirements.txt
```
3. Set up DVC (ask the owner for `YOUR_ACCESS_KEY_ID` and `YOUR_SECRET_ACCESS_KEY`)
```bash
dvc remote add -d storage s3://sequential-post-training-with-intent-ambiguity
dvc remote modify storage endpointurl https://storage.yandexcloud.net
dvc remote modify storage access_key_id YOUR_ACCESS_KEY_ID
dvc remote modify storage secret_access_key YOUR_SECRET_ACCESS_KEY
```
4. Pull data via dvc
```bash
dvc pull
```
5. Use git to commit lightweight sources and reports in `code/` and `progress/` and use dvc to commit heavy files in `data/`.

⚠️  DO NOT commit your `.dvc/config`

