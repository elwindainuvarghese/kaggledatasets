# Kaggle Machine Learning & Tabular Datasets

Authentic 1:1 Kaggle dataset explorer and collection web application, featuring clean CSV dataset statistics, interactive tables, model visualizations, and Python training scripts.

## 🚀 Features
- **Authentic Kaggle UI**: 1:1 reproduction of Kaggle's navigation, header, dataset explorer, tabs, and layout.
- **Dynamic Dataset Explorer**: Interactive CSV table previews with column summaries (mean, min, max, unique counts, nulls) and row pagination.
- **Seamless Downloads**: Download full archives or individual scripts and datasets.
- **Official Navigation Redirects**: Direct integration with official Kaggle sections (`/competitions`, `/benchmarks`, `/arenas`).

## 🛠️ Deploy to Vercel
This repository is pre-configured with `vercel.json`, `requirements.txt`, and `api/index.py` for immediate deployment on Vercel:
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New Project** and import `kaggledatasets`.
3. Click **Deploy**. Vercel will automatically build and deploy the Python app!

## 💻 Local Development
Run locally on Windows:
```bash
python app.py
```
Or double-click `run_kaggle.bat`.
Open [http://localhost:5000](http://localhost:5000) in your browser.
