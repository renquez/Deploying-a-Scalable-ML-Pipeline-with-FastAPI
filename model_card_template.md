# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* model version: 1.0.0
* model: Random Forest Classifier
* starter code provided by Udacity and completed by myself.
* developed in October 2026.


## Intended Use
* Model developed specifically for fulfilling the Udacity Machine Learning Dev
Ops task 2 submission for Deploying a Machine Learning Model with FastAPI.
* model developed for educational purposes and can be used for entertainment/ hobbyist purposes.
* model developed for the specific task of predicting whether income exceeds $50K a year based on census data.
* model is not intended for enterprise use.
* 
* Overall score: 

## Training Data
* dataset comes from UC Irvine Machine learning Repository
* dataset composed from 1994 Census Database
* Features used: Workclass, education, marital-status, occupation, relationship,race, sex, and native-country
* data cleaning: removed white space, converted and removed missing values
* split on train/test is 80/20
* 


## Evaluation Data
Evaluation data is a held out portion of the original dataset
## Metrics
Overall model metrics: Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863

## Ethical Considerations
* historical census data can reflect past societal biases.
* Risk of disparate impact if used for any real decision-making.
* Data set contains protected attributes (race, sex, native-country)

## Caveats and Recommendations
* it is recommended that this model not be used for real-world decision-making.
* 