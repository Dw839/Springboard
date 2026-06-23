![image](https://github.com/Dw839/Springboard/assets/121418737/ac4ee5da-4cfc-4e7f-845e-13b00273f080)

# Flight Delay Prediction Using Machine Learning

## ✈️ Project Overview

This capstone project explores the use of machine learning to predict flight delays in the United States aviation industry. Using historical airline operational data from the U.S. Department of Transportation, multiple machine learning models were developed and evaluated to classify flights as either **Delayed** or **On-Time**.

The goal was to identify the factors contributing to flight delays and develop predictive models that could support operational planning, improve customer experience, and enhance airline decision-making.

---

## 💼 Business Problem

Flight delays create significant operational challenges for airlines, airports, and passengers. Delays can lead to increased operating costs, missed connections, resource inefficiencies, and reduced customer satisfaction.

Accurately predicting flight delays allows stakeholders to:

* Improve operational planning
* Optimize resource allocation
* Reduce disruption costs
* Enhance passenger communication
* Improve overall service reliability

This project investigates how machine learning can be leveraged to predict flight delays before departure using historical flight data.

---

## 🎯 Project Objectives

The primary objectives of this project were to:

* Identify the key factors influencing flight delays.
* Prepare and transform airline operational data for predictive modeling.
* Develop machine learning classification models.
* Compare model performance using multiple evaluation metrics.
* Select the most effective model for flight delay prediction.
* Generate actionable insights to support operational decision-making.

---

## 📊 Dataset

The dataset was sourced from the U.S. Department of Transportation's Bureau of Transportation Statistics and obtained through Kaggle.

### Dataset Characteristics

* Flight Data Year: **2016**
* Records: **5.6+ Million Flights**
* Features: **28 Variables**
* Geographic Scope: United States Domestic Flights

Data included information such as:

* Origin Airport
* Destination Airport
* Airline Carrier
* Departure Delays
* Taxi Times
* Distance
* Scheduled Flight Information
* Operational Performance Metrics

---

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 🔎 Data Preparation & Feature Engineering

Several preprocessing techniques were applied to improve data quality and model performance:

### Data Cleaning

* Removed irrelevant attributes
* Eliminated cancelled and diverted flight records
* Handled missing values
* Standardized airport and carrier identifiers

### Feature Engineering

* Created a binary flight delay classification variable
* Derived operational duration metrics
* Extracted date-based features:

  * Month
  * Day
  * Day of Week

### Encoding

* Applied One-Hot Encoding to categorical variables including:

  * Airline Carrier
  * Origin Airport
  * Destination Airport
  * Day of Week

---

## 📈 Feature Selection

Feature importance analysis was performed to identify the most predictive variables and reduce model complexity.

Key predictive features included:

* Departure Delay
* Wheels Off Elapsed Time
* Taxi Out Time
* Taxi In Time
* Distance
* Operational Timing Metrics

Highly correlated variables were evaluated and redundant features were removed to minimize multicollinearity and improve model efficiency.

---

## 🤖 Machine Learning Models

Three classification algorithms were developed and evaluated:

### 1. Logistic Regression

### 2. Decision Tree Classifier

### 3. Random Forest Classifier

Performance was measured using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

---

## 📊 Model Performance Summary

| Model               | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
| ------------------- | -------- | ------- | --------- | ------ | -------- |
| Logistic Regression | 96.77%   | 0.9909  | 0.9052    | 0.8270 | 0.8644   |
| Decision Tree       | 96.46%   | 0.9168  | 0.8608    | 0.8533 | 0.8570   |
| Random Forest       | 91.75%   | 0.9785  | 0.9893    | 0.3412 | 0.5074   |

---

## 🔑 Key Findings

* Departure-related operational metrics were the strongest predictors of flight delays.
* Logistic Regression achieved the best overall performance with the highest ROC-AUC score and balanced classification metrics.
* Decision Tree also demonstrated strong predictive capability with balanced precision and recall.
* Random Forest achieved exceptionally high precision but struggled with recall, making it less effective at identifying all delayed flights.

---

## 💡 Business Impact

The results demonstrate that machine learning can effectively support airline operations by identifying flights at risk of delay before departure.

Potential applications include:

* Proactive delay management
* Airline operations planning
* Passenger communication improvements
* Resource optimization
* Airport scheduling support
* Operational risk reduction

---

## 🚀 Skills Demonstrated

* Data Cleaning & Preparation
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Feature Selection
* Machine Learning
* Predictive Analytics
* Classification Modeling
* Model Evaluation
* Business Analytics
* Python Programming
* Data Visualization
* Data-Driven Decision Making

---

## 📂 Repository Contents

* Flight Delay Prediction Notebook
* Data Cleaning & Preparation
* Feature Engineering
* Machine Learning Models
* Model Evaluation
* Visualizations
* Business Insights

---

## ✅ Conclusion

This project demonstrates the successful application of machine learning techniques to predict flight delays using large-scale airline operational data. By comparing multiple classification models and evaluating their predictive performance, the analysis identified Logistic Regression as the most effective model for balancing accuracy, recall, and overall predictive capability.

The project highlights how predictive analytics can support operational decision-making, improve efficiency, and enhance customer experience within the aviation industry.

