# MqHackathon2023

# Workers' Compensation Claims Prediction

This project aims to predict the "gross incurred cost" of workers' compensation claims using machine learning (ML) and artificial intelligence (AI) models based on psychosocial and demographic data. The models predict both continuous and categorical outcomes related to the costs and claimants' fitness for work over time.

## Dataset Overview

The dataset contains 71 columns of data across 552 cases, including psychosocial screening and claims cost data. The task focuses on predicting several specific variables that represent different aspects of the claim costs and the claimant's fitness for work over various periods post-injury.

## Preprocessing Steps

1. **Categorical Encoding**: Qualitative variables are encoded into numerical format using one-hot encoding, facilitating their use in ML models.
   
2. **Handling Missing Values**: Missing values are filled with appropriate placeholders or dropped, depending on the nature of the data.

3. **Date Calculations**: The time between the date of the accident and the claim finalization date is calculated to provide a duration which could be a relevant feature for prediction.

4. **Feature Selection**: Irrelevant features are dropped to simplify the model and potentially improve its performance.

## Model Building

Two types of models were built to predict the outcomes:

### Continuous Outcomes

For predicting continuous variables like "Net_total_incurred" and "Total_Paid," a **Random Forest Regressor** model was utilized. The model's performance was evaluated using the Mean Squared Error (MSE) metric, defined by the formula:

\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
\]

where:
- \(n\) is the number of observations,
- \(Y_i\) is the actual value, and
- \(\hat{Y}_i\) is the predicted value.

### Categorical Outcomes

For predicting categorical outcomes such as the claimant's fitness for work at different time intervals, a **Neural Network** model was constructed using TensorFlow and Keras. The network consists of fully connected layers with ReLU activation functions and a softmax output layer for multi-class classification. The loss function used is the sparse categorical cross-entropy, given by:

\[
\text{Loss} = -\sum_{c=1}^{M}y_{o,c}\log(p_{o,c})
\]

where:
- \(M\) is the number of classes,
- \(y\) is the binary indicator (0 or 1) of class \(c\) being the correct classification for observation \(o\),
- \(p\) is the predicted probability of observation \(o\) being of class \(c\).

The accuracy metric is used to evaluate the model's performance, defined as the ratio of correctly predicted instances to the total instances.

## Implementation

The project is implemented in Jupyter notebook, using libraries such as scikit-learn for machine learning models, pandas for data manipulation, and TensorFlow for building neural network models. The preprocessing, model building, training, and evaluation steps are encapsulated in functions to streamline the prediction process for both continuous and categorical outcomes.

## Conclusion

This project demonstrates the application of machine learning and artificial intelligence techniques to predict outcomes related to workers' compensation claims based on psychosocial and demographic data. By accurately predicting these outcomes, strategies can be informed to better manage the psychosocial factors influencing these insurance claims.

