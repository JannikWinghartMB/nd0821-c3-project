# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This Model was created by Jannik Winghart on 2022-04-28 as part of the Udacity Machine Learning DevOps Engineer nanodegree.
The model is a sklearn Support Vector Classifier with default settings.

## Intended Use
The model can be used to predict whether income exceeds $50K/yr based on census data.
To predict the income category the model needs information about age, workclass, fnlwgt
education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, 
capital-loss, hours-per-week and native-country.   
Detailed Information about this features can be found below.

## Training Data
Dataset is aquired from https://archive.ics.uci.edu/ml/datasets/census+income  
The data is preprocessed by removing all whitespaces from the file. Further the categories as one-hot-encoded and labels are binarized

80% of the data are used as training data, the rest is used as test data.

Following features and value ranges / categories are supported:
- **age**: continuous.  
- **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- **fnlwgt**: continuous.
- **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- **education-num**: continuous.  
- **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.  
- **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.  
- **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.  
- **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.  
- **sex**: Female, Male.  
- **capital-gain**: continuous.  
- **capital-loss**: continuous.  
- **hours-per-week**: continuous.  
- **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.  


## Evaluation Data
Evaluation data consists of 20% of the census dataset.
Preprocessing is the same as in the training dataset.

## Metrics
Following metrics are calculated on the complete evaluation dataset:  
precision: 0.1524234693877551  
recall: 0.9755102040816327  
fbeta: 0.26365140650854935  

Slice evaluation on the feature "education" can be found in the file /starter/starter/slice_output.txt


## Ethical Considerations
It has to be considered if this model returns biased predictions because of data bias or structural/social bias.

## Caveats and Recommendations
This model is created with default settings and has a really bad precision. 
It should be researched, if the model could be improved with another architecture or hyperparameters.
