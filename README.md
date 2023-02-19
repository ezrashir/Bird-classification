# Bird-classification

This solution is for the 'Bird Species Classification Challenge' contest in bitgrit: https://bitgrit.net/competition/16
Submissions to this contest are evaluated on accuracy (Number of correct predictions / Total number of predictions).
The accuracy of my predictions (~88.99) can be seen in the contest website under "Leaderboard" and in the file "Bird Classification scoreboard.jpeg"

4 files were used for this solution:
* 1 general file "CLASSES.py" which contains the classes I wrote to predict missing features and classify the test set. 
* 2 script for data preprocessing (including the prediction of features with significant amount of missing values using linear regression):
"trainFeatEstimation.py" and "testFeatEstimation.py"
* 1 scripts for the main classification, including some data preprocessding on the whole test set:  "Classification.py"


Script 1: "trainFeatEstimation.py"
1) After loading the training data and sorting it in a convenient manner, I checked the number of missing features in each feature. Features with a small number of
missing features (<0.01% of the data) were imputed using a "mean" method and categorical data was imputed using the "most frequent" method. (5-24, 26-33)
2) There were two other features with a significant number of missing data (~30% of samples missing for each feature), so simple imputing would not be effective
enough. Thus, I created a linear regression model for each one of the two features, based on all other features and the target in order to predict their missing data.
The linear regression class I created (class name: LinearRegressionPredictor) can be viewd in "CLASSES.py". 
Finding the best model for each feature was done in the folowing manner:
  2.1) Sorting the relevant data for the classification (removing samples with features I need to predict). (34-36)
  2.2) Encoding the categorical data. (38-42)
  2.3) Splitting data to train and test in order to test the model before I predict the missing features. (45-48)
  2.4) Feature scaling. (51-54)
  2.5) Creating polynomial feature for ine estimator (It proved more effective). (57-60)
  2.6) Training and evaluating feature 1 (with polynimial features) using the mean absolute percentage error. (63-69)
  2.7) Training and evaluating feature 2 (with no polynimial features) using the mean absolute percentage error. (71-76)
3) Actually using all relevant samples for each feature to predict the missing values using the same trained model as in the above section. (79-97)
  3.1) Gathering the data with missing values from the two features. (79)
  3.2) Encoding categorical data and feature scaling. (83-86)
  3.3) Splitting the data relevant for the prediction of each feature. (88-90)
  3.4) Creating polynomial features for the relevant feature. (93)
  3.5) predicting the missing features using the same trained model from section 3. (96-97)
4) inserting back the predicted data to the full training dataset that will be used for classifying the bird species. (108-109)

Script 2: "testFeatEstimation.py"
Uploading the test data and following the same processes for the prediction of the features in the test set. The models for prediction here cannot be the same models
as in "trainFeatEstimation.py" since the models there used the target values in order to predict the missing features. Here, the target values can't be used. Thus,
a new model was built in order to predict each one of the two features. THe new models were basically the same only they did not use the target values. 

Script 3: "Classification.py"
1) First I generated the full data sets for the training and test (Using the above scripts with which I predicted the missing data) and sorted it in a convenient
manner. (2-9)
2) Next I encoded the categorical data and labels. (12- 20)
3) Feature scaling. (23-27)
4) Training and evaluating the prediction model. Here, I used my multiclass linear regression class (see "MulticlassLinearRegressionPredictor" in "CLASSES.py") to
predict the training set and produce a confusion matrix to test the model. (29-34)
5) Predictiong the test set (37-38) and saving it in the right format for the contest. (41-51)


