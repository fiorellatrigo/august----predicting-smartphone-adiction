## Smartphone Addiction Prediction Model

### 1. The Problem

Digital wellness platforms and health tech applications struggle to proactively identify users experiencing problematic smartphone usage. Without early detection of high-risk usage patterns, platforms cannot deliver timely digital wellbeing interventions, leading to decreased user well-being and lower long-term engagement with wellness tools.

### 2. Objective

The primary objective of this project is to develop a predictive classification model to identify users with a high probability of smartphone addiction (`addicted_label`). The model must maximize the ROC AUC score to accurately rank users by risk level, enabling targeted, threshold-based behavioral nudges.

### 3. Methodology

**Data & Target**

* **Source:** Proprietary user behavior dataset containing 691,369 training records.


* **Features:** 13 numerical and categorical variables, including daily screen time, gaming hours, and stress levels. Missing values were imputed using median and most-frequent strategies.


* **Target Metric:** `addicted_label` (Binary target evaluated via ROC AUC). The dataset presented a moderate imbalance with 70.94% belonging to the positive class.



**Experimental & Validation Strategy**

* **Exploratory Data Analysis (EDA):** Analyzed variable distributions and target correlations, discovering strong monotonic, non-linear relationships between screen time metrics and addiction probability.


* **Baseline Benchmarking:** Established an initial benchmark using a Logistic Regression pipeline.


* **Ablation & Tree-Based Modeling:** Tested Decision Trees and Random Forests on a restricted Top-3 feature set (`daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours`) to isolate non-linear predictive power.


* **Gradient Boosting Integration:** Evaluated XGBoost and CatBoost algorithms against the Random Forest baseline. Conducted a final experiment re-introducing all features into the XGBoost architecture to capture complex, multi-variable interactions.



### 4. Findings

**Experimental Performance Summary**

* **Feature Dominance:** The Top-3 screen usage features accounted for almost all the predictive signal in simpler models, yielding a Random Forest ROC AUC of 0.93343. Engineered ratio features failed to improve upon this score.


* **Interaction Effects:** While dropping minor features benefited simpler models, re-integrating all features into XGBoost unlocked critical variable interactions, drastically improving the validation ROC AUC to 0.96190.


* **Generalization vs. Regularization:** The final all-feature XGBoost model exhibited a small train-validation gap of 0.00745. Applying stricter regularization parameters (`min_child_weight`, `reg_lambda`) reduced this gap to 0.00535 but slightly degraded the validation ROC AUC, leading to the selection of the unregularized configuration.



### 5. Recommendations

* **Targeted Interventions:** Integrate the XGBoost probability scores into the application backend to trigger automated digital wellbeing nudges when users cross critical probability thresholds.
* **Threshold Monitoring:** Establish real-time tracking specifically for `daily_screen_time_hours` and `weekend_screen_time`, as EDA proved these are the primary leading indicators of addiction risk.


* **Model Simplification:** If computational resources become constrained in production, deploy the optimized Random Forest using only the Top-3 features, as it retains a highly competitive 0.93343 ROC AUC with a fraction of the data pipeline complexity.



### 6. Technologies

* **Language:** Python 3.14


* **Data Manipulation & EDA:** Pandas, NumPy, Matplotlib, Seaborn


* **Machine Learning:** Scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier), XGBoost, CatBoost


* **Model Serialization:** Joblib



**Project Structure**

```text
├── data/                  # Local directory for train and test CSV files
├── notebook/              # Jupyter Notebook containing EDA, pipelines, and evaluation
|   └── best_random_forest.joblib # Serialized baseline Random Forest model
|   └── submission.csv         # Exported target probabilities for the test set
├── README.md              # Executive summary and project documentation
└── requirements.txt       # Project dependencies

```

**Installation & Environment Setup**
This project uses an isolated Python environment. To replicate this setup, run the following commands in your terminal:

1. Clone the repository

```bash
git clone https://github.com/fiorellatrigo/august----predicting-smartphone-adiction.git
cd august----predicting-smartphone-adiction

```

2. Create and activate the virtual environment

```bash
# Windows:
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux:
python -m venv .venv
source .venv/bin/activate

```

3. Install dependencies

```bash
pip install -r requirements.txt

```
