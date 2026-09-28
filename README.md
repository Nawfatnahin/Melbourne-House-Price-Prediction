# 🏠 Melbourne House Price Prediction

A machine learning project that predicts house prices in Melbourne, Australia using the [Melbourne Housing Dataset](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot). The project compares multiple regression models and identifies key features driving property values.

## 📊 Project Overview

This project follows a structured machine learning pipeline to predict residential property prices across Melbourne suburbs. It covers end-to-end data science workflow — from data cleaning to model evaluation and feature importance analysis.

### Key Findings

| Model | MAE (AUD) | RMSE (AUD) | R² Score |
|-------|-----------|------------|----------|
| **Random Forest Regressor** | **155,161** | **249,661** | **0.837** |
| Decision Tree Regressor | 204,504 | 322,354 | 0.728 |

- Random Forest outperforms Decision Tree with lower prediction errors and better fit
- RMSE > MAE for both models, indicating the presence of outlier predictions
- Top influential features include building area, number of rooms, and suburb location

## 🔧 Tech Stack

- **Python 3.x**
- **pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib & Seaborn** — Data visualization
- **scikit-learn** — Machine learning models and preprocessing

## 📁 Project Structure

```
├── House_Price_Prediction_Project.ipynb   # Main Jupyter notebook
├── melb_data.csv                          # Melbourne housing dataset
└── README.md                              # Project documentation
```

## 🧪 Methodology

### 1. Data Loading & Exploration
- Loaded Melbourne housing dataset (~13,000+ records, 21 features)
- Explored data types, distributions, and summary statistics

### 2. Data Cleaning
- Removed irrelevant columns (`Address`, `SellerG`, `Date`, `Method`, `Postcode`, `Lattitude`, `Longtitude`, `Propertycount`)
- Handled missing values using `SimpleImputer` (strategy: constant fill with 0)

### 3. Exploratory Data Analysis (EDA)
- Visualized distributions of key features like `Price`, `Rooms`, `Distance`
- Analyzed correlations between features using heatmaps
- Examined categorical variable distributions (`Type`, `Suburb`, `CouncilArea`, `Regionname`)

### 4. Feature Engineering
- Separated features into numerical and categorical
- Applied `ColumnTransformer` with:
  - **StandardScaler** for numerical features (`Rooms`, `Distance`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`)
  - **OneHotEncoder** for categorical features (`Suburb`, `CouncilArea`, `Regionname`, `Type`)

### 5. Train-Test Split
- 80/20 split using `train_test_split`

### 6. Model Training
Three models were trained and evaluated:
- **Random Forest Regressor** (with `GridSearchCV` hyperparameter tuning)
- **Decision Tree Regressor** (`max_depth=90`, `min_samples_split=100`)
- **Linear Regression** (baseline model — struggled with high-cardinality one-hot features, R² = -0.34)

### 7. Model Comparison
- Compared models using MAE, RMSE, and R² metrics
- Random Forest selected as the best performer

### 8. Feature Importance Analysis
- Extracted `feature_importances_` from the Random Forest model
- Visualized top 10 most influential features with horizontal bar chart

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/Nawfatnahin/Melbourne-House-Price-Prediction.git
   cd Melbourne-House-Price-Prediction
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook House_Price_Prediction_Project.ipynb
   ```
3. Run all cells sequentially

## 📈 Sample Visualizations

The notebook includes:
- Price distribution histograms
- Correlation heatmaps
- Feature importance bar charts
- Model comparison tables

## 📝 Lessons Learned

- **Linear Regression** can produce negative R² scores when dealing with high-dimensional sparse features (600+ one-hot encoded columns) without regularization
- **Random Forest** handles non-linear relationships and high-cardinality categorical features more robustly than linear models
- Proper feature scaling is critical — `StandardScaler` helps normalize features with different ranges
- **GridSearchCV** helps find optimal hyperparameters but doesn't always improve over defaults

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Acknowledgments

- Dataset sourced from [Kaggle — Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)
- Built as part of a Machine Learning coursework project

## This project is my first project as a new machine learning learner. I maybe update or change somethings in the future. Thanks if you saw my project and thanks to CampusX to help me learn machine learning easily.   
