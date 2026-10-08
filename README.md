# LendWise: Predicting Loan Defaults

A machine-learning analysis of loan-default risk completed as a team project for **CIS 5450: Big Data Analytics** at the **University of Pennsylvania** during Spring 2025.

## Project Overview

Loan default creates significant financial risk for lenders and can also affect borrowers through higher interest rates, reduced access to credit, and unfavorable loan terms. This project explores how borrower characteristics, credit information, and loan terms can be used to estimate default risk.

The project follows an end-to-end data science workflow:

- Data-quality assessment
- Exploratory data analysis
- Feature engineering and preprocessing
- Classification-model development
- Class-imbalance mitigation
- Hyperparameter tuning
- Model evaluation and interpretation

The team compared Logistic Regression, Random Forest, and XGBoost models. Because the target variable was substantially imbalanced, the analysis emphasized precision, recall, F1-score, and ROC-AUC rather than relying only on accuracy.

## Project Context

This was a team project completed for CIS 5450: Big Data Analytics at the University of Pennsylvania.

Model development and evaluation were completed collaboratively by the project team. The repository includes the complete team notebook, while the section below identifies my individual contributions.

## My Contributions

My primary contribution was exploratory data analysis and data-quality assessment. I:

- Evaluated the dataset for missing values, duplicate records, and invalid numerical values
- Analyzed the distributions of numerical, categorical, and target variables
- Identified the substantial imbalance between default and non-default records
- Examined correlations among numerical variables and between numerical variables and loan default
- Visualized relationships between borrower characteristics, loan terms, and default outcomes
- Communicated findings that informed the team’s feature-engineering, class-balancing, and model-evaluation decisions

## Dataset

The analysis uses the [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) available through Kaggle.

The dataset contains:

- **255,347 records**
- **18 variables**
- **Target variable:** `Default`
- **Non-default records:** approximately 88.39%
- **Default records:** approximately 11.61%

The features describe borrower characteristics, credit history, employment, and loan terms.

### Selected Features

| Feature | Description |
|---|---|
| `Age` | Borrower’s age |
| `Income` | Borrower’s annual income |
| `LoanAmount` | Requested loan amount |
| `CreditScore` | Borrower’s credit score |
| `MonthsEmployed` | Length of employment |
| `NumCreditLines` | Number of active credit lines |
| `InterestRate` | Interest rate assigned to the loan |
| `LoanTerm` | Duration of the loan |
| `DTIRatio` | Debt-to-income ratio |
| `Education` | Borrower’s education level |
| `EmploymentType` | Borrower’s employment category |
| `MaritalStatus` | Borrower’s marital status |
| `LoanPurpose` | Intended purpose of the loan |
| `HasMortgage` | Whether the borrower has a mortgage |
| `HasDependents` | Whether the borrower has dependents |
| `HasCoSigner` | Whether the loan has a cosigner |
| `Default` | Whether the borrower defaulted |

The dataset is not redistributed in this repository. Download `Loan_default.csv` directly from Kaggle and place it in the local `data/` directory.

## Exploratory Data Analysis

The EDA examined dataset quality, feature distributions, class balance, and relationships among variables.

### Data Quality

The analysis found:

- No rows containing missing values
- No duplicate records
- No obvious invalid values in the primary numerical fields
- A substantial imbalance in the target variable

### Target-Class Distribution

Approximately 88.39% of the records represent non-defaults, while approximately 11.61% represent defaults.

This imbalance means that accuracy can be misleading. A model that predicts nearly every record as a non-default could achieve high accuracy while failing to identify borrowers who actually default.

This finding informed the team’s use of:

- Stratified train-test splitting
- Class weighting
- SMOTE
- Decision-threshold analysis
- Precision, recall, F1-score, and ROC-AUC

### Feature Distributions

The categorical features—including education, employment type, marital status, and loan purpose—were relatively evenly distributed across their categories.

The numerical variables also covered broad ranges without obvious data-quality issues or extreme distributional anomalies.

### Correlation Analysis

The numerical variables generally showed weak pairwise correlations, suggesting limited multicollinearity and a diverse feature space.

Individual numerical variables also had relatively weak linear relationships with the target. Among the original numerical features:

- Higher interest rates were associated with greater default risk
- Greater age was associated with lower default risk
- No single feature was sufficient to explain loan default independently

These findings supported the use of both linear and nonlinear classification models.

## Feature Engineering and Preprocessing

The team’s modeling workflow included:

- Removing `LoanID` because it did not provide predictive information
- Separating numerical and categorical features
- Encoding categorical variables
- Scaling numerical variables when appropriate
- Creating a loan-to-income feature
- Selecting informative features
- Applying stratified training and testing splits
- Experimenting with class weighting and SMOTE

The engineered loan-to-income feature helped describe the borrower’s financial burden relative to income.

## Models Evaluated

### Logistic Regression

Logistic Regression was used as an interpretable baseline for binary classification.

The team evaluated:

- Default Logistic Regression
- Class-weighted Logistic Regression
- Feature selection
- L1 and L2 regularization
- Hyperparameter tuning with `GridSearchCV`
- A reduced tuning strategy informed by statistical testing

The reduced tuning strategy decreased runtime by approximately 86.2% while maintaining similar predictive performance.

### Random Forest

Random Forest was evaluated because it can:

- Capture nonlinear relationships
- Model interactions between features
- Provide feature-importance estimates
- Handle mixed feature types
- Reduce overfitting through bagging

The team performed grid-search tuning and evaluated multiple decision thresholds to understand the tradeoff between recall and false positives.

### XGBoost

XGBoost was evaluated as a boosting-based ensemble model capable of capturing complex nonlinear relationships.

The team experimented with:

- A baseline XGBoost model
- SMOTE
- Balanced class weights
- Stratified cross-validation
- Grid-search hyperparameter tuning

The final tuned XGBoost model used class weighting and stratified validation to improve minority-class performance.

## Model Results

| Model | Accuracy | ROC-AUC | Precision for Defaults | Recall for Defaults | F1 for Defaults |
|---|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 68.96% | 0.7599 | 0.2274 | 0.6975 | — |
| Random Forest, threshold 0.4 | 57.83% | approximately 0.75 | 0.1885 | 0.8043 | — |
| Stratified XGBoost | 81.60% | 0.7602 | — | — | 0.3739 |

### Model Interpretation

The models addressed different business priorities:

- **Logistic Regression** provided an interpretable baseline with strong recall but a relatively high number of false positives.
- **Random Forest at a 0.4 threshold** achieved the highest default-class recall, making it useful when missing a potential default is especially costly.
- **XGBoost** offered the strongest overall balance among accuracy, ROC-AUC, minority-class F1, and predictive stability.

The team selected **XGBoost** as the strongest overall model.

## Important Predictors

Age, loan-to-income ratio, and interest rate were the most consistently important predictors across Logistic Regression, Random Forest, and XGBoost.

- **Age:** Older borrowers were generally less likely to default.
- **Loan-to-income ratio:** Higher loan burdens relative to income were associated with greater default risk.
- **Interest rate:** Higher interest rates were associated with greater default risk.

Although the models use different mathematical approaches, their agreement on these predictors increases confidence that the variables contain meaningful risk information within this dataset.

## Repository Structure

```text
lendwise-loan-default-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── notebooks/
    └── lendwise_loan_default_analysis.ipynb
```

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- XGBoost
- SciPy
- Statsmodels
- Google Colab
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/lendwise-loan-default-prediction.git
cd lendwise-loan-default-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/nikhil1e9/loan-default).
2. Download the dataset.
3. Extract `Loan_default.csv`.
4. Place the file in the repository’s `data/` directory:

```text
data/Loan_default.csv
```

The dataset and Kaggle credentials are excluded from version control.

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/lendwise_loan_default_analysis.ipynb
```

The notebook also checks `/content/Loan_default.csv`, allowing it to run in Google Colab after the dataset is manually uploaded.

## Key Challenges

### Class Imbalance

The most significant modeling challenge was the imbalance between default and non-default records. Baseline models achieved high overall accuracy while identifying relatively few defaults.

The team addressed this through class weighting, SMOTE, stratified validation, threshold adjustment, and evaluation metrics designed for imbalanced classification.

### Performance Versus Computation

Exhaustive hyperparameter tuning was computationally expensive because the dataset contains more than 255,000 records. The team used statistical testing and reduced parameter grids to balance model performance with training efficiency.

### Precision-Recall Tradeoff

Increasing default-class recall resulted in more false positives. This tradeoff is important because:

- Low recall may cause a lender to miss high-risk borrowers.
- Low precision may cause creditworthy borrowers to be incorrectly flagged.

The appropriate threshold would ultimately depend on business objectives, the relative cost of each error, and applicable lending requirements.

## Limitations and Responsible Use

This project is educational and should not be used as a production lending system.

Important limitations include:

- The dataset may not represent the characteristics of real lending populations.
- Model performance may change as borrower behavior and economic conditions evolve.
- Accuracy alone is not sufficient for evaluating an imbalanced risk model.
- The analysis does not constitute a complete fairness or disparate-impact assessment.
- Production use would require model calibration, explainability, privacy controls, monitoring, documentation, and regulatory review.
- Lending decisions should not be automated solely from a predictive-model output.

Future work could include:

- Conducting a formal fairness assessment
- Calibrating predicted probabilities
- Evaluating additional classification and anomaly-detection techniques
- Adding model-explainability methods such as SHAP
- Monitoring population and performance drift
- Packaging the selected model behind an API
- Creating a dashboard for controlled model interpretation

## Acknowledgments

- University of Pennsylvania, CIS 5450: Big Data Analytics
- [Loan Default Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
- Collaborators: Derek Fu, Sunita Pathak, Grace Wei
