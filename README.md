# **Volleyball Player Statistics Analysis for NCAA Division 3 (2024)**

---

### **Overview**

**PR Name**: *"Analysis of Volleyball Player Performance Metrics Using Machine Learning Techniques"*

This project analyzes player statistics from the Illinois Tech volleyball team for NCAA Division 3 (2024). Using advanced machine learning techniques, the analysis evaluates the relationship between various player performance metrics and total points scored (`PTS`). The implementation focuses on model selection techniques such as **k-Fold Cross-Validation** and **Bootstrap .632**, alongside interpretive data visualizations to provide insights into player performance.

---

## **Implementation Details**

### **1. Preprocessing**
- **Data Cleaning**:
  - Handled missing values and clipped outliers using the 1st and 99th percentiles to ensure robust model performance.
- **Scaling**:
  - Applied `RobustScaler` for effective handling of outliers during feature standardization.
- **Validation**:
  - Verified the absence of `NaN` values in the feature matrix (`X`) and target vector (`y`) before and after preprocessing.

---

### **2. Model Selection Techniques**

#### **k-Fold Cross-Validation**
- **Description**:
  - Splits the data into 5 folds, training on 4 and testing on 1 iteratively, to evaluate the model's generalization error.
  - Calculates the Mean Squared Error (MSE) across all folds.
- **Output**:
  - **k-Fold Cross-Validation MSE**: `3231.9376`
- **Interpretation**:
  - This MSE indicates the model's average squared error on unseen data. A lower value suggests better generalization to new samples.

---

#### **Bootstrap .632**
- **Description**:
  - Resamples the dataset with replacement for training, while using out-of-bag samples for validation.
  - Combines in-sample and out-of-sample errors using the `.632 adjustment` for a balanced error estimate.
- **Output**:
  - **Bootstrap .632 MSE**: `1464.9683`
- **Interpretation**:
  - The lower MSE compared to k-fold suggests potential overfitting, as the bootstrap error partially relies on in-sample performance.

---

### **3. Model Accuracy**

#### **R² Score**
- **Output**: `0.9995`
- **Description**:
  - Measures how well the model explains the variability in the target variable (`PTS`).
- **Interpretation**:
  - A value of **0.9995** indicates that the model explains **99.95% of the variance** in `PTS`, demonstrating an excellent fit.
- **Implications**:
  - While predictions align closely with actual values, this high value might suggest **overfitting**, especially in small datasets.

---

#### **Mean Absolute Error (MAE)**
- **Output**: `1.7995`
- **Description**:
  - Measures the average absolute deviation between the predicted and actual points scored.
- **Interpretation**:
  - The model's predictions are off by **1.7995 points** on average. For example, if a player scores 25 points, the model might predict **23.2 or 26.8**.
- **Implications**:
  - A low MAE indicates strong predictive accuracy.

---

## **Visualizations**

### **1. Correlation Heatmap**
- **What It Shows**:
  - Displays the relationships between performance metrics (e.g., `K`, `DIG`) and `PTS`.
- **Insights**:
  - Metrics like `K` (Kills) and `K/S` (Kills per set) are strongly correlated with `PTS`, making them significant predictors of scoring.

---

### **2. Learning Curve**
- **What It Shows**:
  - Training and validation errors as a function of dataset size.
- **Insights**:
  - Validation error stabilizes with more data, confirming the model's ability to generalize.

---

### **3. Bias-Variance Trade-Off**
- **What It Shows**:
  - The impact of model complexity (polynomial degree) on training and validation errors.
- **Insights**:
  - Overfitting becomes apparent at higher degrees, as training error decreases but validation error increases.

---

### **4. Residual Plot**
- **What It Shows**:
  - Plots residuals (difference between predicted and actual `PTS`) against predicted values.
- **Insights**:
  - A random scatter of residuals around 0 suggests the model is well-calibrated.

---

### **5. Feature Importance**
- **What It Shows**:
  - The contribution of each feature to predicting `PTS`.
- **Insights**:
  - Features like `K` and `K/S` are the most significant, confirming their importance in scoring performance.

---

### **6. Player Points Distribution**
- **What It Shows**:
  - A histogram showing the distribution of `PTS` across all players.
- **Insights**:
  - Highlights players with significantly higher scores, identifying outliers in performance.

---

### **7. ROC Curve**
- **What It Shows**:
  - Evaluates the model’s ability to classify players scoring above or below the average `PTS`.
- **Output**:
  - **AUC**: `0.98` (excellent discrimination).
- **Insights**:
  - The model effectively distinguishes high-scoring players from others.

---

### **8. Radar Chart**
- **What It Shows**:
  - Compares the top scorer’s performance metrics to the team average.
- **Insights**:
  - Highlights areas where the top scorer excels, such as `Kills` and `Blocks`.

---

## **Team Contributions**

### **Kunal Nilesh Samant (20541900)**
- Implemented data preprocessing, including handling missing values, scaling, and clipping outliers.
- Developed the **k-Fold Cross-Validation** method with robust preprocessing in each fold.
- Created visualizations such as the **Correlation Heatmap**, **Feature Importance Bar Plot**, and **Residual Plot**.
- Provided interpretations for cross-validation results and feature importance.

---

### **Dhruv Singh (A20541901)**
- Implemented the **Bootstrap .632 Estimator** with error handling.
- Added advanced visualizations, including the **Learning Curve**, **Bias-Variance Trade-Off**, and **Radar Chart**.
- Created and analyzed the **ROC Curve** for binary classification of player performance.
- Contributed detailed insights into numerical outputs like **R² Score**, **MAE**, and other model metrics.

---

### **How to Run the Code**
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`tabula-mvb_stats_2024.csv`) in the appropriate folder.
4. Run the script in a Jupyter Notebook or any Python IDE.

--- 