# 🏦 Municipal Bond Credit Rating Predictor

## 📌 Overview

This project aims to predict the **credit rating** of municipal bonds (e.g., AAA, AA, A, BBB) using features like **coupon rate, yield, price, duration, and sector**. It combines end-to-end data science workflows including data preprocessing, EDA, ML modeling, SHAP explainability, and **Streamlit-based app deployment**.

---

## 📁 Dataset

**File:** `muni_bonds_with_ratings.csv`  
**Records:** 2,000  
**Columns:**
- `BondID`: Unique bond identifier
- `State`: State of issuance
- `CreditRating`: Target label (AAA, AA, A, BBB)
- `Sector`: Sector of the bond (Education, Utilities, etc.)
- `Coupon`: Coupon rate (%)
- `IssueDate` and `MaturityDate`
- `Price`: Market price
- `Yield`: Yield rate (%)
- `Duration`: Bond duration (years)

---

## 🔍 Exploratory Data Analysis (EDA)

- **Univariate analysis** of credit ratings, sectors, and states
- **Bivariate analysis** of bond features against credit ratings
- **Correlation heatmap** to assess relationships between numeric fields
- **Distribution plots** for yield, coupon, and duration

---

## 🧹 Data Preprocessing

- Removed rows with missing `CreditRating`
- Converted `CreditRating` into numeric labels using `LabelEncoder`
- Feature matrix `X` and target variable `y` defined
- Train-test split: 80% training / 20% testing

---

## 🔮 Model Building

- **Model:** `RandomForestClassifier` (100 trees, `random_state=42`)
- **Evaluation:**
  - Classification Report
  - Confusion Matrix
  - Accuracy & Precision
- **Feature Importance:** Plotted to show most influential factors

---

## 📈 Explainability with SHAP

- Used `shap.TreeExplainer` for model interpretation
- **SHAP Summary Plot**: Shows which features push predictions towards higher/lower credit ratings
- Helps regulators and investors understand **why** a bond received a certain prediction

---

## 🌐 Streamlit App

A user-friendly **Streamlit web app** was created to:
- Input bond features interactively
- Get predicted bond credit rating
- View **SHAP force plot** for individual prediction explainability

### To run:
```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- SHAP
- Streamlit
- Matplotlib / Seaborn for EDA

---

## ✅ Why Random Forest?

- Handles non-linearities and feature interactions well
- Provides built-in feature importance
- Easy to explain with SHAP
- Outperforms simpler models like Logistic Regression on this data

---

## 🔄 Future Improvements

- Try more advanced models like XGBoost or CatBoost
- Hyperparameter tuning using GridSearchCV
- Incorporate external market or economic data
- Deploy on cloud (Streamlit Cloud / Hugging Face Spaces)

---

## 📊 Sample Predictions

| Coupon | Price | Yield | Duration | Sector       | Predicted Rating |
|--------|-------|-------|----------|--------------|------------------|
| 4.25   | 101.5 | 3.2   | 7.1      | Education    | AA               |
| 3.0    | 98.0  | 3.6   | 5.0      | Healthcare   | A                |
| 5.0    | 110.2 | 2.9   | 8.5      | Transportation| AAA             |

---
📈 Explainability with SHAP

SHAP (SHapley Additive exPlanations) is a powerful framework based on game theory that helps interpret the output of machine learning models.

🔍 Why SHAP in This Project?

Credit rating predictions impact financial decisions, so model transparency is crucial.

SHAP helps explain how much each feature contributed to a bond's predicted rating.

It provides global insights (which features matter most overall) and local explanations (why a specific bond got its predicted rating).

We used TreeExplainer, optimized for Random Forest, to generate:

SHAP Summary Plot: Shows feature importance and direction of impact

SHAP Force Plot (via Streamlit): Visualizes individual predictions with feature-level contributions

This improves trust and interpretability, especially in financial ML systems.

✅ Why Random Forest?
We selected the Random Forest Classifier for several reasons:

✅ Strengths:
Handles non-linear relationships and interactions between features

Works well on tabular data like ours

Provides built-in feature importance

Plays nicely with SHAP for model explainability

Robust against overfitting (with enough trees)

🔁 Alternatives Considered:

Algorithm	Notes
1. Logistic Regression	Too simple, assumes linearity — misses feature interactions
2. Decision Trees	Easy to interpret but prone to overfitting
3. XGBoost / LightGBM	More powerful, but adds complexity and less interpretability


We chose Random Forest as the best balance between accuracy, speed, and interpretability.

---

## 🙋‍♂️ Author

**Vaibhav Sharma**  
[GitHub](https://github.com/vaisharma16) • [Kaggle](https://www.kaggle.com/vaisharma) • [Medium](https://medium.com/@vai_sharma)

---
