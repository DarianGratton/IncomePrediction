# Income Prediction with Machine Learning
A microservice built using django and python that takes a api call with a json object (containing user infomation), and returns a prediction of how much they make annually.  

## How to run
Step 1: Clone repo and navigate to it in the command line
Step 2: Set up a virtual environment
```
virtualenv environment_name   // Windows, different on MAC and Linux
```
Step 3: Run the environment
```
.\environment_name\Scripts\activate
```
Step 4: Download the dependencies
```
pip install -r requirements.txt
```
Step 5: Navigate to the Django project (PredictionService) and run server
```
python manage.py migrate
python manage.py runserver
```
Step 6: Open a new command line, navigate back, start environment again and then create super user (Neccessary for Token)
```
python manage.py createsuperuser   // Email is uneccessary
```
Step 7: Get a token using Httpie then copy it to clipboard
```
http post http://127.0.0.1:8000/api/token/ username=your_username password=your_password
```
Step 8: Open Postman (https://www.getpostman.com/)
Step 9: Change to a post request, input the api call, and set up authorization by clicking type -> Bearer Token and pasting the token where it says token
![postman](https://github.com/Trilobite256/IncomePrediction/blob/master/images/Postman.PNG?raw=true)
Step 10: Click on the body tab, select raw, insert json, and change type to JSON (application/json)
```
// JSON to insert
{ 
	"age": 43,
	"workclass": "Never-worked",
	"fnlwgt": 70800,
	"education": "Bachelors",
	"education-num": 13,
	"marital-status": "Never-married",
	"occupation": "?",
	"relationship": "Unmarried",
	"race": "Black",
	"sex": "Male",
	"capital-gain": 0,
	"capital-loss": 0,
	"hours-per-week": 40,
	"native-country": "United-States"
}
```
![postman2](https://github.com/Trilobite256/IncomePrediction/blob/master/images/Postman2.PNG?raw=true)
Step 11: Click send (Note if it says invalid token, it probably expired, just get a new one)

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
### Deployment
Was unable to deploy in time but I would probably try to deploy it on heroku, as it offers free SSL and is fairly easy to set up once you know what your doing (it's also free which is nice). However given the time I would probably look into services like aws and azure as I know that they could probably offer more in the long term. 

### Testing
In terms of integration testing I spent a large chunk of time testing and ensuring the machine learning model worked when given the right information. However I wasn't able to spend much time testing how it would react given the wrong type of infomation. 

## Sources
https://github.com/jadianes/winerama-recommender-tutorial/tree/master/winerama
https://github.com/wkudaka/django-scikit-learn-tutorial
http://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf
