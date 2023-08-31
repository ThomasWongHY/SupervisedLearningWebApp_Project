# SupervisedLearningWebApp_Project

## Executive Summary:
The aim of this project is to develop a software product that can predict the likelihood of fatal collisions resulting in loss of life based on actual data collected by the Toronto police department over a period of five years. The predictive service will utilize various features such as weather conditions, road conditions, and location to classify incidents as either resulting in a fatality or not.

## Data Exploration:
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/d4b9b95b-09f6-40ee-9dde-121496fc5213)

![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/e4341974-7057-4971-8dff-3bd5b1271a02)


## Data Preprocessing:
1.	Drop duplicates in ACCNUM column
2.	Replace NaN value in the dataframe to ‘ ‘
3.	Retrieve the rows which do not have ‘unknown’ value
4.  Convert TIME column to a new column ‘INTERVAL’ which   shows the time period of the accident

## Data Visualization
1. Visualize the object columns

  a.	Multiple: The number of unique values >2 and < 20
  
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/96cc6c40-cb73-4777-8d51-ee3b0aec668a)

  b.	Binary: The number of unique values = 2
  
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/543431ed-008d-464c-8fff-cb3d6e4e15de)

## Data Modelling:

1. Select the related rows in the dataframe:

    a. Not too much unique values (<20)
   
    b. Include sufficient values (missing values/ all values < 0.5)
   
    c. Relate to the condition of accident and its severity (Fatal, Non-Fatal)
   
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/ca54c283-02d6-42b3-9cff-b56d2c076d3e)

2. Replace the values in ACCLASS as {'Non-Fatal':0, 'Fatal':1}
3. Split the dataframe to X (features) and y (target)
4. Implement the data transformation:

   a. SimpleImputer(strategy="constant",fill_value='missing')
   
   b. MinMaxScarler()
   
   c. get_dummies()
   
5. Feature Selection:

   a. SelectKBest(score_func = chi2, k = 10)
   
6. Manage the imbalance classes by oversampling
7. Create pipeline class to streamline the transformers

   a. cat_pipline: SimpleImputer, OneHotEncoder
   
   b. num_pipline: SimpleImputer, MinMaxScalar

8. Transform the data based on their dtypes

![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/8d6f291a-bf4f-40c6-98c4-be2234d918bd)

9. Split the data into training set and testing set with portion of 0.8 and 0.2

![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/9f377b71-7bf9-4017-8f7a-ef3d8b1f3894)

## Feature Selection
Feature selection is an important step in machine learning, as it helps to identify the most relevant features for a model. The goal of feature selection is to remove irrelevant or redundant features, which can lead to overfitting and decrease the accuracy of a model.

In this project, there are four tools and techniques were used for feature selection:

1.	SelectKBest: This is a statistical method that selects the K best features based on a given score function. In this project, chi-squared test was used as the score function to select the best features.

2.	RandomizedSearchCV: This is a technique used for hyperparameter tuning. It randomly selects combinations of hyperparameters and evaluates their performance using cross-validation.

3.	StratifiedShuffleSplit: This is a method for splitting a dataset into training and test sets while preserving the class distribution.

4.	Pipeline: This is a method for chaining multiple steps in a machine learning workflow, such as data preprocessing, feature selection, and model training.

## Model Evaluation:
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/3a99b4b4-cd8e-40a2-9d91-462e62676efe)
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/71b2c8fe-b52f-4a8f-a16d-ef6e049912cf)
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/df580fd6-fea3-4843-9c21-e69db5e0cbff)
![image](https://github.com/ThomasWongHY/SupervisedLearningWebApp_Project/assets/86035047/d6d56731-bd3f-49e4-9683-112fd62e6267)

## Conclusion
The first model (Logistic Regression) with solver 'saga', penalty '11', and C=10 achieved an accuracy of 0.7568, precision of 0.7631, recall of 0.9774, and F1-score of 0.8571. The confusion matrix shows that 95 instances were correctly classified as negative, and 2504 instances were correctly classified as positive.

The second model (The Decision Tree Classifier) with min_samples_split=10, min_samples_leaf=1, max_depth=28, and criterion='gini' achieved an accuracy of 0.7702, precision of 0.7822, recall of 0.9590, and F1-score of 0.8617. The confusion matrix shows that 188 instances were correctly classified as negative, and 2457 instances were correctly classified as positive.

The third model (Random Forest Classifier) with n_estimators=300, min_samples_split=2, min_samples_leaf=2, max_depth=None, and criterion='gini' achieved an accuracy of 0.7708, precision of 0.7815, recall of 0.9617, and F1-score of 0.8623. The confusion matrix shows that 183 instances were correctly classified as negative, and 2464 instances were correctly classified as positive.

The fourth model (MLP Classifier) with solver 'lbfgs', learning_rate='constant', hidden_layer_sizes=(20, 10), alpha=0.01, and activation='tanh' achieved an accuracy of 0.7723, precision of 0.7827, recall of 0.9617, and F1-score of 0.8630. The confusion matrix shows that 188 instances were correctly classified as negative, and 2464 instances were correctly classified as positive.
