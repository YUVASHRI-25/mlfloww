# 🌍 Country Clustering Project

## 📌 Overview

This project performs **unsupervised learning (K-Means clustering)** on a dataset of countries to group them based on socio-economic and health indicators.
The goal is to identify patterns that classify countries into:

* Developed nations
* Least developed nations
* Transitional economies
* Trade-heavy economies
* Economically unstable nations

---

## 📂 Dataset

The dataset (`Country-data.csv`) contains socio-economic indicators for multiple countries.

### Columns Explained

* **`child_mort`** → Child mortality rate (deaths under age 5 per 1,000 live births).
* **`exports`** → Exports of goods & services (% of GDP).
* **`health`** → Health expenditure (% of GDP).
* **`imports`** → Imports of goods & services (% of GDP).
* **`income`** → Per capita net income (average income per person).
* **`inflation`** → Annual inflation rate (%).
* **`life_expec`** → Life expectancy at birth (years).
* **`total_fer`** → Fertility rate (average number of children per woman).
* **`gdpp`** → GDP per capita (economic output per person).

---

## 🛠️ Project Workflow

1. **Data Loading & Preprocessing**

   * Handle missing values, duplicates.
   * Standardize numerical features.

2. **Exploratory Data Analysis (EDA)**

   * Distribution plots, correlations.
   * Top/Bottom countries by GDP, income, life expectancy, mortality.

3. **Dimensionality Reduction**

   * Apply **PCA** to reduce features to 2D/3D for visualization.

4. **Clustering**

   * Apply **K-Means** clustering.
   * Evaluate using **Silhouette Score** & **ARI (Adjusted Rand Index)**.

5. **Cluster Interpretation**

   * **Cluster 1 → Rich, high life expectancy, low fertility (Developed nations)**
   * **Cluster 2 → Poor, high mortality, high fertility (Least developed)**
   * **Cluster 0 → Middle-income transitional economies**
   * **Cluster 3 → Trade-heavy nations**
   * **Cluster 4 → Economically unstable (inflation-driven)**

6. **Visualization**

   * PCA scatterplots with clusters.
   * Bar charts for top/bottom countries.

---

## 📊 Technologies Used

* **Python**
* **Pandas, NumPy** (data preprocessing)
* **Matplotlib, Seaborn** (EDA & visualization)
* **Scikit-learn** (PCA, KMeans, clustering evaluation)
* **MLflow** (experiment tracking)
* **Gradio** (interactive app)

---

## 📌 Results

* Countries are grouped into **5 meaningful clusters**.
* Helps policymakers, NGOs, and researchers identify similarities across nations.

---

