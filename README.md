## Human Behavioral Dynamics Influenced by Social Media
This project has 2 working machine learning models which predicts:
 - Internet Addiction 
 - Impulsive Shopping Behaviour
 influenced by social media. 
   
## Data
Internet addiction: [Kaggle dataset](https://www.kaggle.com/datasets/apoorva1225/social-media-influence)
Shopping Behaviour: [Google Drive Link](https://docs.google.com/spreadsheets/d/1j18VKrb7REEodmfxH2ViELJjplRd-4Mp/edit?usp=sharing&ouid=114123281321181342880&rtpof=true&sd=true)

## Sample Features
Internet Addiction
  - Age
  - Job/School Performance
  - Sleep patterns
  - Time spent online
  - Mood offline
Shopping Behaviour
  - Age
  - Employment Status
  - Time spent on social media
  - Average amount spent on shopping per month
  - Purchase Influence
  - Frequency of online shopping

## Models
Both models were built using same algorithm - XGBoost.
Internet addiction prediction model has 3 class labels - low, moderate and high.
Shopping behaviour prediction model has 2 class labels - addicted and not addicted.
  - SMOTE is used to handle the class imbalance.
Saved the models using **joblib** and integrated it with a flask web app for interactions.

## Technologies & Tools
Python (Numpy, Pandas, Scikit-Learn, Matplotlib, Seaborn)
XGBoost (Algorithm)
Joblib (to save the models)
Flask (web application)

## Results
- Predictions made via Flask web interface
- Balanced dataset improved model performance
