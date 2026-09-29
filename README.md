# 🏠 Melbourne House Price Prediction

A machine learning project that predicts residential property prices in Melbourne, Australia using the [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) dataset from Kaggle. The project implements an end-to-end scikit-learn preprocessing pipeline, trains regression models, compares evaluation metrics, and extracts feature importances driving property values.

---

## 📊 Project Overview

This project builds a regression pipeline to predict property sale prices across Melbourne suburbs. It addresses real-world data science challenges including high-cardinality categorical variables, missing property attributes, spatial coordinate features, and skewed price distributions.

### Key Evaluation Findings

Evaluated on an unseen test split (20% holdout, 2,716 samples):

| Model | MAE (AUD) | RMSE (AUD) | R² Score | Hyperparameters / Setup |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **$155,313.95** | **$248,775.82** | **0.8381** | `n_estimators=100`, `criterion='squared_error'` |
| **Decision Tree Regressor** | **$203,272.66** | **$320,464.05** | **0.7314** | `max_depth=90`, `min_samples_split=100` |

- **Random Forest** achieved superior generalization with an $R^2$ score of **~0.84**, reducing average absolute prediction error to ~$155k AUD.
- **Outlier Penalty**: Across both models, RMSE is substantially higher than MAE, reflecting premium/luxury real estate outliers that exert higher squared error penalties.
- **Feature Depth**: Dimensionality expansion to 607 features enabled tree ensembles to capture neighborhood-level premium variations effectively.

---

## 🔧 Tech Stack

- **Python 3.x**
- **pandas** — Data manipulation and exploratory data analysis
- **NumPy** — Vectorized numerical operations
- **scikit-learn** — Imputation, ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder, DecisionTreeRegressor, RandomForestRegressor, and evaluation metrics
- **Matplotlib & Seaborn** — Data visualization and feature importance charting

---

## 📁 Repository Structure

```
├── House_Price_Prediction_Project.ipynb   # Complete ML workflow notebook
├── melb_data.csv                          # Melbourne housing dataset
└── README.md                              # Project documentation
```

---

## 🧪 Machine Learning Pipeline & Methodology

### 1. Data Ingestion & Target Definition
- Dataset: **13,580 records** and **21 features**.
- Target: `Price` (continuous numerical target in AUD).
- Train/Test Split: 80% training set (10,864 rows) and 20% testing set (2,716 rows) using `random_state=3`.

### 2. Feature Selection & Engineering
- **Dropped Features (6)**: `['Regionname', 'Propertycount', 'Price', 'Postcode', 'Address', 'Date']` to eliminate redundancy and leakage.
- **Retained Inputs (15)**: `Suburb`, `Rooms`, `Type`, `Method`, `SellerG`, `Distance`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `CouncilArea`, `Lattitude`, `Longtitude`.
- **Property Age Transformation**: Converted construction year into property age:
  $$\text{Property Age} = 2026 - \text{YearBuilt}$$

### 3. Preprocessing & Encoding Pipeline
A unified `ColumnTransformer` handles missing values and categorical encoding simultaneously:
- **Numerical Imputation**: Missing values in `YearBuilt` and `BuildingArea` are filled with `0` via `SimpleImputer(strategy='constant', fill_value=0)`.
- **Frequent Imputation**: Missing `Car` parking counts are imputed using `SimpleImputer(strategy='most_frequent')`.
- **High-Cardinality One-Hot Encoding**: `Suburb`, `Method`, `Type`, and `SellerG` are encoded with `OneHotEncoder(handle_unknown='ignore')`.
- **Nested Pipeline for Council Area**: `CouncilArea` passes through sequential most-frequent imputation followed by one-hot encoding.
- **Passthrough Features**: `Rooms`, `Distance`, `Bedroom2`, `Bathroom`, `Landsize`, `Lattitude`, and `Longtitude` pass through without loss of spatial fidelity.
- **Sparse Feature Matrix**: The resulting dataset spans **607 features**.
- **Standardization**: Scaled via `StandardScaler(with_mean=False)` to normalize feature scales while preserving matrix sparsity.

### 4. Model Training & Comparison
Two tree-based regression architectures were trained and compared on standardized features:
1. **Decision Tree Regressor**: Regularized with `max_depth=90` and `min_samples_split=100` to prevent deep branch memorization on high-dimensional dummy indicators.
2. **Random Forest Regressor**: 100-tree bagging ensemble delivering ensemble variance reduction and higher predictive accuracy.

### 5. Feature Importance Analysis
- Extracted Gini importance scores (`feature_importances_`) from the fitted tree models.
- Identified primary price drivers: building dimensions, geographical suburb locations, number of rooms/bathrooms, and distance from Melbourne's Central Business District (CBD).

---

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/Nawfatnahin/Melbourne-House-Price-Prediction.git
   cd Melbourne-House-Price-Prediction
   ```

2. Launch Jupyter Notebook or VS Code:
   ```bash
   jupyter notebook House_Price_Prediction_Project.ipynb
   ```

3. Execute the notebook cells sequentially to reproduce the data processing, model training, and performance metrics.

---

## 📈 Visualizations Included

- **Distribution Plots**: Target `Price` and property attribute distributions
- **Correlation Heatmap**: Cross-feature relationships and multicollinearity assessment
- **Feature Importance Chart**: Ranked horizontal bar plot of top predictive features
- **Performance Evaluation Table**: Side-by-side MAE, RMSE, and $R^2$ benchmarks

---

## 🤝 Acknowledgments

- Dataset provided by [Kaggle — Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) (scraped by Tony Pino).
- Coursework inspiration and guidance from [CampusX](https://www.youtube.com/@campusx-official).

---

## ✍️ Author's Note

This project is my first project as a new machine learning learner. I may update or change some things in the future. Thanks for checking out my project, and thanks to [CampusX](https://www.youtube.com/@campusx-official) for helping me learn machine learning easily!
