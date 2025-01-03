# Customer Churn Prediction

## Overview
This project focuses on predicting customer churn for a subscription-based business using machine learning techniques. The objective is to identify at-risk customers and provide actionable insights to improve retention strategies.

## Dataset
The dataset used for this project is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains information about customer demographics, account details, and service usage.

### Key Features:
- **Customer Information:** Gender, Senior Citizen, Partner, Dependents.
- **Account Details:** Tenure, Monthly Charges, Total Charges.
- **Service Usage:** Internet Service, Streaming, Contract Type.
- **Target Variable:** Churn (Yes/No).

## Steps Performed
### 1. **Data Preprocessing**
- Handled missing values in the `TotalCharges` column.
- Encoded categorical variables using Label Encoding.
- Standardized numerical features for better model performance.

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed correlations between features and customer churn.
- Created visualizations like heatmaps and bar charts to understand data distributions.

### 3. **Model Building**
- Split the data into training and testing sets.
- Trained machine learning models (Random Forest, XGBoost).
- Evaluated model performance using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

### 4. **Insights**
- Identified key factors influencing customer churn (e.g., contract type, tenure).
- Suggested retention strategies, such as offering discounts for long-term contracts and improving customer support for high-risk segments.

## Results
- Achieved an AUC-ROC score of **0.92** with the Random Forest model.
- Reduced churn rate by **15%** in the simulated environment through targeted recommendations.

## Tools & Technologies
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Visualization Tools:** Matplotlib, Seaborn
- **Deployment (Optional):** Flask or Django

## File Structure

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter Notebook or scripts to reproduce results

