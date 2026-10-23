# Model Card


## Model Details
This model is version 1.0.0 and is a Random Forest Classifier implemented in scikit-learn.  The starter code was 
provided by Udacity and was completed by Raul Enriquez.  This model was developed in October 2026.

## Intended Use
This model was developed to fulfill the requirements of Udacity's Machine Learning DevOps Task 2 ("Deploying a Machine
Learning Model with FastAPI").  It was created for educational purposes and may be used for personal or hobbyist 
experimentation.  It's specific task is to predict whether an individual's income exceeds $50,000 per year based on 
1994 census data.  It is not intended for enterprise, production, or real-world decision making.

## Training Data
The dataset comes from the UC Irvine Machine learning Repository and is derived from the 1994 Census Database.  Features
used include both categorical variables (workclass, education, marital-status, occupation, relationship, race sex, and
native-country) and numerical variables (age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week).
During preprocessing, leading and trailing whitespace was stripped from string columns and rows containing missing
values were removed.  The data was split into an 80% training set and a 20% test set.

## Evaluation Data
Model performance was evaluated on the 20% hold-out test set that was not used during training.

## Metrics
On the hold out test set the model achieved the following metrics:
* Precision: 0.7419
* Recall: 0.6384
* F1 score: 0.6863

These are standard binary classification metrics and are appropriate for this task.

## Ethical Considerations
Historical census data can reflect past societal biases.  There is a risk of disparate impact if the model is used for
any real-world decision-making.  The dataset contains protected attributes (race, sex,and native-country).

## Caveats and Recommendations
This model should not be used for real-world decision-making.  Performance on groups with low sample counts shows high
variance and should be considered unreliable.  The model was trained on 1994 data and may not generalize well to contemporary
populations.
