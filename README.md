# Task3_Classification_WideBot

Task 3 - Classification

Using the same data in task 2, build a machine learning based classifier and show its
performance on the test-set. Show precision, recall, f-score, accuracy for each class and for
the whole test data. Please describe briefly the meaning of each result and metric for this
specific task. Also, please write some enhancements that you may think about to achieve
better results.

Note: add a readme file that describes the whole training process in briefed points. Note:
Use stories data only (not comments). Also, take the last 20% of each file as a test-set.


**Training Process:**

1. Import the required libraries, including GridSearchCV for hyperparameter tuning, TfidfVectorizer for feature extraction, LogisticRegression for the classifier, and various metrics for evaluation.

2. Prepare the training and testing data:
   - Extract the 'story' column from the DataFrame 'big_data_frame_80_percent_shuffled' as the training feature.
   - Extract the 'topic' column from the DataFrame 'big_data_frame_80_percent_shuffled' as the training target (labels).
   - Extract the 'story' column from the DataFrame 'big_data_frame_20_percent_shuffled' as the testing feature.
   - Extract the 'topic' column from the DataFrame 'big_data_frame_20_percent_shuffled' as the testing target (labels).

3. Create a pipeline using `make_pipeline()`:
   - The pipeline consists of TfidfVectorizer for feature extraction and LogisticRegression as the classifier.

4. Define the hyperparameter grid for the logistic regression model:
   - A range of values for the hyperparameter 'C' is specified for grid search.

5. Create and fit the model:
   - `GridSearchCV` is initialized with the pipeline and hyperparameter grid, using 5-fold cross-validation (cv=5) for tuning.
   - The model is trained using the training data (train_feature and train_target).

6. Make predictions and evaluate the model:
   - The trained model is used to predict the labels for the testing data (test_feature).
   - The accuracy score is calculated by comparing the predicted labels with the actual labels (test_target).
   - The `classification_report` function provides a detailed evaluation of precision, recall, and F1-score for each class, as well as overall metrics.

7. Print the results:
   - The accuracy score is printed to show how well the model performs on the test data.
   - The `classification_report` is printed to provide a comprehensive assessment of the model's performance for each class and overall.

The training process involves building a text classification model using a pipeline that combines TF-IDF vectorization for feature extraction and logistic regression for classification. Hyperparameter tuning is performed through grid search to optimize the regularization parameter 'C' of the logistic regression model. The model's accuracy and classification metrics are then evaluated on the test data to assess its performance.
