# Predicting-Student-Math-Scores-with-a-TensorFlow-Regression-Pipeline
A TensorFlow regression pipeline to predict student math scores from the StudentsPerformance.csv dataset. Includes preprocessing, feature encoding, model training, and evaluation with MAE, RMSE, and R². Demonstrates how deep learning can be applied to educational analytics.
This project applies machine learning techniques to predict student math scores using the StudentsPerformance.csv dataset. The aim is to explore how demographic, academic, and socio-economic factors influence academic performance, and to build a regression pipeline with TensorFlow to model these relationships.

Project Overview

Dataset: Student performance data including gender, parental education level, lunch type, and test preparation course status.

Target Variable: Math score (continuous).

Approach: Regression using a feedforward neural network.

Workflow

Data Preprocessing

Cleaned the dataset and handled categorical variables with one-hot encoding.

Normalized numerical features for stable model training.

Split data into training and testing sets.

Modeling

Implemented a dense neural network using TensorFlow/Keras.

Hidden layers with ReLU activation, output layer with linear activation.

Optimized using Adam optimizer and trained with Mean Squared Error (MSE) loss.

Evaluation

Used Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

Plotted training/validation loss curves.

Visualized predicted vs. actual math scores.

Results & Insights

Students who completed test preparation courses generally performed better in math.

Parental education level showed a positive correlation with student outcomes.

Lunch type (standard vs. free/reduced) was also a strong predictor of performance.

Tech Stack

Python: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

TensorFlow / Keras: Deep learning regression model

Future Improvements

Perform hyperparameter tuning with Keras Tuner or GridSearch.

Compare with classical regression models (Linear Regression, Random Forest).

Extend pipeline to predict reading and writing scores.
