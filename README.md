# CreditWise Loan System

An end-to-end Machine Learning classification pipeline designed to automate and streamline loan approval decisions for **SecureTrust Bank**.

## Overview

SecureTrust Bank offers personal and home loans across urban and rural regions of India. Previously, loan officers manually verified income proofs, employment details, and credit history—a process that was time-consuming, inconsistent, and prone to false rejections of reliable borrowers or approvals of high-risk applicants.

**CreditWise Loan System** analyzes historical customer records to predict whether a loan application should be **Approved (`1`)** or **Rejected (`0`)** prior to final human verification.

---

## Dataset Description

The dataset (`loan_approval_data.csv`) contains **1,000 applicant records** across **20 attributes** covering personal, financial, and credit profiles:

| Feature | Description |
| :--- | :--- |
| `Applicant_ID` | Unique identifier for each applicant (dropped prior to modeling) |
| `Applicant_Income` | Monthly income of the primary applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Employment_Status` | Employment type (`Salaried` / `Self-Employed` / `Business`) |
| `Age` | Age of the applicant |
| `Marital_Status` | Marital status (`Married` / `Single`) |
| `Dependents` | Number of financial dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of active loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Current savings balance |
| `Collateral_Value` | Value of collateral pledged |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Repayment duration in months |
| `Loan_Purpose` | Purpose of loan (`Home` / `Education` / `Personal` / `Business`) |
| `Property_Area` | Location category (`Urban` / `Semi-Urban` / `Rural`) |
| `Education_Level` | Highest qualification (`Graduate` / `Postgraduate` / `Undergraduate`) |
| `Gender` | Applicant gender (`Male` / `Female`) |
| `Employer_Category` | Employer sector (`Govt` / `Private` / `Self`) |
| `Loan_Approved` | **Target Variable:** `1` = Approved, `0` = Rejected |

---

## Machine Learning Pipeline

1. **Data Cleaning & Imputation:**
   * Handled missing values using `SimpleImputer` (`mean` strategy for numerical features and `most_frequent` for categorical features).
2. **Exploratory Data Analysis (EDA):**
   * Visualized target class balance, categorical distributions, feature skewness, and outliers via histograms and boxplots.
3. **Feature Engineering & Encoding:**
   * Encoded ordinal/target variables (`Education_Level`, `Loan_Approved`) with `LabelEncoder`.
   * Applied `OneHotEncoder(drop="first")` to nominal categorical features (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`).
   * Standardized numerical features using `StandardScaler`.
4. **Model Training & Evaluation:**
   * Trained and compared multiple classification algorithms (including Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes) evaluated on Accuracy, Precision, Recall, and F1-Score.

---

## Repository Structure

```text
CreditWise-Loan-System/
├── Credit_Wise.ipynb                        # Main Jupyter Notebook with EDA, preprocessing & modeling
├── loan_approval_data.csv                   # Historical loan application dataset
├── CreditWise Loan System Description.txt   # Problem statement and schema reference
├── README.md                                # Project documentation
└── .gitignore                               # Git ignore rules