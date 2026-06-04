#   CineIQ: AI-Powered Hybrid Recommendation Engine

CineIQ is a hyper-personalized hybrid movie recommendation engine that combines Collaborative Filtering (SVD), Content-Based Filtering (TF-IDF), and algorithmic transparency to deliver accurate and explainable movie suggestions.

## Features
* **Hybrid Algorithm**: Balances the "Crowd Matrix" (what similar users like) and "Content DNA" (thematic similarity based on plot and genres).
* **Algorithmic Transparency**: Interactive visualizations explaining *why* a movie was recommended.
* **Premium Dashboard**: A sleek, dark-mode web application built with Streamlit and Plotly.
* **MLflow Tracking**: Experiment tracking for tuning SVD hyperparameters.
 
## Tech Stack
* **Backend Models**: Scikit-Learn (TruncatedSVD, TfidfVectorizer)
* **Frontend**: Streamlit, Plotly, Custom CSS (Glassmorphism)
* **Data Sources**: MovieLens 25M, TMDB 45K Movies

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd CINEIQ
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Datasets
This script will automatically download the required datasets from Kaggle into the `data/raw/` directory.
```bash
python src/download_data.py
```

### 4. Train the AI Models
This script samples the datasets, trains the SVD and TF-IDF models, and saves the `.pkl` artifacts to the `models/` directory.
```bash
python -m src.train
```

### 5. Launch the Dashboard
Fire up the Streamlit interface!
```bash
python -m streamlit run app.py
```

## 📂 Project Structure
* `app.py`: Streamlit frontend dashboard.
* `src/recommender.py`: Inference engine combining content and collaborative scores.
* `src/train.py`: Pipeline for training machine learning models.
* `src/models.py`: Architecture for the TF-IDF and SVD algorithms.
* `src/data_loader.py`: Pre-processing logic for the CSV data.
* `src/download_data.py`: Kaggle API ingestion logic.
