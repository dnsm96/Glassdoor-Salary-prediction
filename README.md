# Glassdoor Salary Prediction Project

## Project Overview
In today's fast-paced tech industry, understanding salary trends is vital for job seekers, employers, and policymakers. This project leverages job postings data from Glassdoor (2017) to predict salaries for various tech roles. By analyzing features such as job title, company size, location, and job description, the project aims to uncover salary trends, compare roles across industries, and build a predictive model for salary estimation.

## Project Type
- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- Linear Regression
- Random Forest Regression
- Stacking Regressor

## Contribution
Individual

## Problem Statement
This project addresses the following key questions:
- How does salary vary by job position (e.g., Data Scientist vs. Software Engineer vs. DevOps Engineer)?
- What is the impact of company size on salary levels?
- How do salaries differ by location (e.g., San Francisco vs. Austin vs. New York)?
- Can we build a predictive model to estimate salaries based on job attributes?

## Dataset
- **Source:** Glassdoor job postings (2017)
- **Rows:** 956
- **Columns:** 15

### Key Features:
- **Job Title:** Title of the job position (e.g., Data Scientist)
- **Salary Estimate:** Estimated salary range (e.g., $53K-$91K)
- **Job Description:** Text description of the job
- **Rating:** Company rating on Glassdoor
- **Company Name:** Name of the hiring company
- **Location:** Job location (city, state)
- **Headquarters:** Company headquarters location
- **Size:** Number of employees in the company
- **Founded:** Year the company was founded
- **Type of Ownership:** Company type (e.g., Private, Public)
- **Industry & Sector:** Industry and sector of the company
- **Revenue:** Estimated company revenue range
- **Competitors:** Company competitors

## Data Characteristics
- No duplicate entries
- Minimal missing values, handled during preprocessing
- Numerical and categorical features, with textual data in job description

## Methodology

### 1. Data Preprocessing
#### Cleaning:
- Dropped irrelevant columns (e.g., Unnamed: 0)
- Replaced missing values (e.g., -1 in Competitors) with NaN and dropped columns with excessive missing data
- Extracted **Minimum Salary**, **Maximum Salary**, and computed **Average Salary** from Salary Estimate
- Derived **Company Age** from Founded and **Average Employees** from Size

#### Handling Missing Values:
- Imputed numerical columns like **Founded** and **Minimum Salary** using median/mean
- Imputed categorical columns like **Type of Ownership** with "NaN"

#### Outlier Treatment:
- Adjusted outliers in **Minimum Salary** using IQR method
- Corrected inconsistencies in **Min_emp** and **Max_emp**

#### Categorical Encoding:
- Used **Target Encoding** for **Sector** and **Type of Ownership** to convert categories into numerical values

### 2. Textual Data Preprocessing (NLP)
#### Steps:
- Expanded contractions, converted to lowercase, removed punctuation and stopwords
- Tokenized and lemmatized text, applied POS tagging
- Vectorized **Job Description** using **TF-IDF (500 features)**, followed by **PCA to reduce to 10 components**

**Reason:** To extract meaningful features from job descriptions while managing dimensionality

### 3. Feature Engineering
- Created new features like **encoded_company_name** and **encoded_job_title** by encoding based on average salary
- Applied **log transformation** to **Company Age** to reduce skewness
- Standardized numerical features using **StandardScaler**

### 4. Model Development
#### Models Used:
- **Linear Regression**
- **Random Forest Regressor**
- **Stacking Regressor** (XGBoost, Bayesian Ridge, SVR with Ridge as meta-model)

#### Evaluation Metrics:
- **R² Score:** Measures variance explained by the model
- **RMSE:** Quantifies prediction error in salary units

#### Cross-Validation & Hyperparameter Tuning:
- Used **5-fold cross-validation** for all models
- Applied **RandomizedSearchCV** for **Random Forest** and **Stacking Regressor** to optimize hyperparameters

### 5. Model Explainability
- Used **SHAP** to interpret the **Stacking Regressor**, identifying **encoded_job_title** and **encoded_company_name** as the most influential features

## Results

### Key Insights from EDA
- **Salary Distribution:** Right-skewed, with most salaries in the **$50K-$100K** range, indicating potential salary inequality
- **Company Size:** Mid-sized companies (1001-5000 employees) are most common
- **Location:** Certain hubs (e.g., major cities) dominate job openings, suggesting high demand in those areas
- **Job Titles:** "Data Scientist" and related roles are highly prevalent
- **Company Age vs. Salary:** No strong correlation, indicating other factors like role and company size are more influential

### Visualizations
#### 15 Charts Created:
- **Univariate:** Histograms (e.g., Salary Distribution), Count Plots (e.g., Company Size)
- **Bivariate:** Scatter Plots (e.g., Company Age vs. Salary), Bar Charts (e.g., Top Companies by Salary)
- **Multivariate:** Correlation Heatmap, Pair Plot

**Insights:** Highlighted salary disparities, dominant company sizes, and high-demand locations

### Best Model
- **Stacking Regressor** (after cross-validation and hyperparameter tuning) was chosen for its robustness and versatility, combining strengths of **XGBoost, Bayesian Ridge, and SVR**

### Feature Importance (SHAP Analysis)
#### Top Features:
1. **encoded_job_title:** Job roles (e.g., senior vs. junior) heavily influence salary
2. **encoded_company_name:** Certain companies are associated with higher/lower salaries
3. **Average Employees & Company Age:** Larger and older companies tend to pay more
4. **Text Features (TF-IDF PCA):** Contributed modestly, capturing nuanced patterns in job descriptions

## Conclusion
This project successfully analyzed salary trends in the tech industry, identifying key factors like job title, company size, and location that influence compensation. The **Stacking Regressor**, with an **R² of 0.97** and **RMSE of 0.17**, provides a reliable tool for salary prediction, offering actionable insights for job seekers, employers, and policymakers.

## Libraries Used
- scikit-learn
- shap
- nltk
- transformers
- pandas, numpy, matplotlib, seaborn

---

This project demonstrates how data-driven analysis can be used to make informed salary predictions, helping stakeholders navigate the tech job market effectively.

