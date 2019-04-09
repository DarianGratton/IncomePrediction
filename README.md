# Income Prediction with Machine Learning
A microservice built using django and python that takes a api call with a json object (containing user infomation), and returns a prediction of how much they make annually.  

## Quickstart Guide

## Dataset (https://archive.ics.uci.edu/ml/datasets/Adult)
```
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
```

## API (Django)
### POST Income Prediction: api/get_income_prediction
Parameters:
```
{
	age: Number,
	workclass: String,
	fnlwgt: Number,
	education: String,
	education-num: Number,
	marital-status: String,
	occupation: String,
	relationship: String,
	race: String,
	sex: String,
	capital-gain: Number,
	capital-loss: Number,
	hours-per-week: Number,
	native-country: String
} // Data is the same as the fields in the dataset
```
Response:
```
{
	prediction: String
}
```

## Unfinished
# Deployment
Was unable to deploy in time but I would probably try to deploy it on heroku, as it offers free SSL and is fairly easy to set up once you know what your doing (it's also free which is nice). However given the time I would probably look into services like aws and azure as I know that they could probably offer more in the long term. 

# Testing
In terms of integration testing I spent a large chunk of time testing and ensuring the machine learning model worked when given the right information. However I wasn't able to spend much time testing how it would react given the wrong type of infomation. In terms of unit testing I would of liked to add more, I spend a lot of time manual testing which isn't great long term.  

## Sources
https://github.com/jadianes/winerama-recommender-tutorial/tree/master/winerama
https://github.com/wkudaka/django-scikit-learn-tutorial
http://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf
