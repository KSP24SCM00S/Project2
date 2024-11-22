# **Volleyball Player Statistics Analysis for NCAA Division 3 (2024)**

---

### **Overview**

**PR Name**: *"Analysis of Volleyball Player Performance Metrics Using Machine Learning Techniques"*

This project analyzes player statistics from the Illinois Tech volleyball team for NCAA Division 3 (2024). Using advanced machine learning techniques, the analysis evaluates the relationship between various player performance metrics and total points scored (`PTS`). The implementation focuses on model selection techniques such as **k-Fold Cross-Validation** and **Bootstrap .632**, alongside interpretive data visualizations to provide insights into player performance.

---

### **How to Run the Code**

1. **Prepare the Required Files**:
   - Download the `.ipynb` file containing the code.
   - Download the dataset file (`tabula-mvb_stats_2024.csv`).

2. **Upload Files to Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload both the `.ipynb` file and the dataset file by clicking on the folder icon in the left sidebar and then the upload button.

3. **Set the Dataset Path**:
   - After uploading, copy the file path for the dataset from the Colab file manager (e.g., `/content/tabula-mvb_stats_2024.csv`).
   - Replace the `file_path` variable in the code with the copied path:
     ```python
     file_path = '/content/tabula-mvb_stats_2024.csv'
     ```

4. **Install Missing Libraries (If Any)**:
   - If the code encounters a missing library error, install it by running:
     ```python
     !pip install <library_name>
     ```
   - Replace `<library_name>` with the name of the required library (e.g., `seaborn`, `scikit-learn`, etc.).

5. **Execute the Notebook**:
   - Run the cells sequentially in Google Colab.
   - Ensure that the dataset path is correctly set before running the code to avoid errors.

6. **View the Results**:
   - The outputs, including model evaluation metrics and visualizations, will be displayed in the Colab notebook.

**Note**: The code has been optimized for **Google Colab**. Running it in other IDEs, such as Visual Studio Code, may result in incomplete visual outputs (e.g., heatmaps). Always use Google Colab for consistent and accurate results.

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
# **Project 2 Questions Answered**

---

## **1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?**

In simple cases like linear regression, **cross-validation (k-fold)** and **bootstrap .632** generally align with simpler model selectors like the Akaike Information Criterion (AIC). AIC balances the model's goodness of fit with its complexity, penalizing models with excessive parameters, while cross-validation estimates the generalization error directly. Bootstrap .632 combines in-sample and out-of-sample errors to provide a robust estimate.

In this dataset, the high **R² score (0.9995)** and low **MAE (1.7995)** indicate that the linear regression model fits exceptionally well, capturing almost all variance in the target (`PTS`). Both **k-fold MSE (3231.9376)** and **bootstrap .632 MSE (1464.9683)** suggest the model generalizes well, consistent with AIC’s preference for simpler models. 

However, differences may arise in more complex scenarios. For example, AIC assumes residual normality, which cross-validation and bootstrap do not. In this relatively small and clean dataset, these methods align well.

---

## **2. In what cases might the methods you've written fail or give incorrect or undesirable results?**

While cross-validation and bootstrap are robust, certain limitations can lead to failure or undesirable outcomes:

### **Small Dataset**
With only 15 players, splitting into 5 folds leaves just 12 samples for training in each fold. This small size can limit the model's ability to generalize. Similarly, bootstrap resampling may repeatedly select similar subsets, reducing variability in error estimates.

### **Overfitting**
The extremely high **R² score** suggests potential overfitting, where the model captures noise alongside actual relationships in the data. Bootstrap, which incorporates in-sample error, might underestimate the degree of overfitting.

### **Outliers**
Despite clipping extreme values, residual outliers may distort predictions. For example, a player with unusually high stats could heavily influence the model’s parameters, inflating the MSE.

### **Multicollinearity**
Features like `K` (Kills) and `K/S` (Kills per set) are highly correlated, causing multicollinearity. This can destabilize parameter estimation and inflate bootstrap variance.

### **Misaligned Metrics**
Cross-validation minimizes MSE but doesn’t directly penalize model complexity. AIC accounts for complexity, but cross-validation might favor overly complex models in small datasets.

---

## **3. What could you implement given more time to mitigate these cases or help users of your methods?**

To address these challenges, the following improvements could be implemented:

### **Regularization**
Introduce techniques like Ridge or Lasso regression to handle multicollinearity and reduce overfitting by penalizing large coefficients.

### **Robust Preprocessing**
- Add automated outlier detection mechanisms to handle anomalies more effectively.
- Use feature engineering methods like **Principal Component Analysis (PCA)** to address multicollinearity.

### **Hybrid Model Selection**
Combine AIC and cross-validation to balance goodness of fit with model complexity, leveraging AIC's simplicity criterion alongside empirical validation from cross-validation.

### **Advanced Bootstrap Techniques**
Implement alternative methods such as **balanced bootstrap**, which ensures diverse resampling, improving error variability estimates for small datasets.

### **Visualization Tools**
Expand diagnostic visualizations (e.g., residual plots, learning curves, bias-variance trade-off) to provide users with a deeper understanding of model performance and limitations.

### **Exposed Parameters**
Expose fine-tuning parameters for users, such as:
- Number of folds in cross-validation.
- Number of bootstrap iterations.
- Gradient descent hyperparameters (learning rate, iterations).
- Preprocessing thresholds for outlier clipping.

---

## **4. What parameters have you exposed to your users in order to use your model selectors?**

### **Cross-Validation**
- `k`: Number of folds (default: 5).
- `shuffle`: Whether to shuffle the data before splitting.
- `random_state`: Seed for reproducibility.

### **Bootstrap**
- `n_iterations`: Number of bootstrap samples (default: 100).
- `.632 adjustment`: Balances in-sample and out-of-bag errors.

### **Gradient Descent**
- `alpha`: Learning rate (default: 0.01).
- `iterations`: Number of iterations for convergence (default: 1000).
- `clip_value`: Threshold for gradient clipping to stabilize updates.

### **Preprocessing**
- Outlier clipping thresholds (1st and 99th percentiles).
- Scaling method (`RobustScaler`) for handling skewness and outliers.

### **Model Evaluation**
- Metrics: **MSE**, **R² score**, **MAE**, and **ROC-AUC** for comprehensive performance assessment.

---

## **Output Interpretation**

### **k-Fold Cross-Validation MSE**: `3231.9376`
- **What it Means**:
  - Reflects the average squared error on unseen data. A lower value indicates better generalization.
- **Implication**:
  - The model performs well on unseen data but shows some error variability.

### **Bootstrap .632 MSE**: `1464.9683`
- **What it Means**:
  - Provides a slightly optimistic error estimate by blending in-sample and out-of-sample errors.
- **Implication**:
  - A lower error compared to k-fold suggests some overfitting to the training data.

### **R² Score**: `0.9995`
- **What it Means**:
  - Explains **99.95% of the variability** in `PTS`. Indicates an excellent model fit.
- **Implication**:
  - Highlights strong predictive accuracy but raises concerns about overfitting.

### **MAE**: `1.7995`
- **What it Means**:
  - Average deviation of predictions from actual values is **1.8 points**.
  - For instance, if a player scores 25 points, the model might predict **23.2 or 26.8**.
- **Implication**:
  - Low MAE indicates good predictive precision, acceptable in this context.

---
