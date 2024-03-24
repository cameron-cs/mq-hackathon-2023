# MqHackathon2023

# Workers Compensation Claims Prediction

## Project Overview

This project aims to predict the "gross incurred cost" of workers' compensation claims based on psychosocial and demographic data. The primary goal is to leverage machine learning and artificial intelligence to inform strategies that manage psychosocial factors influencing insurance claims. This solution utilizes a combination of RandomForestRegressor for continuous variables and a Neural Network model for categorical outcomes.

## Data Description

The dataset comprises 71 columns of data across 552 cases, including psychosocial screening data and claims cost data. The target variables for prediction include cost-related variables (Net_total_incurred, Total_Paid, Other_Paid, Payment_medical, and Other_paid_risk) and categorical variables indicating the claimant's fitness for work at various time intervals (Fitness_week2, Fitness_week6, Fitness_week12, Fitness_week26, and Fitness_week52).

## Methodology

### Data Preprocessing

1. **Categorical Encoding:** Non-numeric columns such as 'Gender', 'State', and 'Intervention Required' were encoded using one-hot encoding to transform them into a format that can be provided to ML models for prediction.

2. **Date Handling:** The difference between 'Claim_Finalised_Date' and 'Date_of_Accident' was calculated to capture the claim duration, an important feature for predicting costs.

3. **Missing Values Handling:** Missing values in categorical data were filled with 'Unknown' or a suitable placeholder, and numerical missing values were imputed with either 0 or the mean of the column, depending on the context.

4. **Feature Reduction:** Unnecessary features and those with many missing values were dropped to simplify the model without compromising prediction accuracy.

### Machine Learning Models

#### RandomForestRegressor

- **Mathematical Background:** RandomForest is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the mean/average prediction of the individual trees for regression tasks.

- **Formula:** The prediction \( \hat{y} \) of the RandomForestRegressor for a given input vector \( X \) is calculated by averaging the predictions of \( N \) individual trees:

\[ \hat{y} = \frac{1}{N} \sum_{i=1}^{N} f_i(X) \]

where \( f_i(X) \) is the prediction of the \( i^{th} \) tree.

#### Neural Network for Categorical Outcomes

- **Architecture:** The model consists of an input layer, two hidden layers with ReLU activation, dropout layers for regularization, and a softmax output layer for multiclass classification.

- **Loss Function:** The Sparse Categorical Crossentropy loss function is used, which is suitable for multi-class classification problems when labels are integers.

- **Optimization:** Adam optimizer is used for its efficiency in both computation and memory requirement.

### Model Evaluation

- For continuous variables, Mean Squared Error (MSE) is used:

\[ MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 \]

- For categorical outcomes, accuracy is the metric of choice, representing the proportion of correctly predicted instances.

## Results

- The models' performance metrics, such as MSE for continuous outcomes and accuracy for categorical outcomes, provide insights into their effectiveness in predicting various aspects of workers' compensation claims.

## Conclusion

This project demonstrates the application of RandomForestRegressor and Neural Networks to predict the costs and recovery outcomes of workers' compensation claims based on psychosocial and demographic data. Through meticulous data preprocessing and strategic model selection, we can gain valuable insights to inform better management strategies for such claims.
