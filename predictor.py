import numpy as np
from joblib import load
def predict_class(input_mail):
  model = load("./models/saved_model.sav")
  feature_extraction = load("./models/feature_extraction.sav")

  input_data_features = feature_extraction.transform(input_mail)
  prediction = model.predict(input_data_features)
  
  print(prediction)
  
  if (prediction[0]==1):
    print('Ham mail')
  else:
    print('Spam mail')

if __name__ == '__main__':
  predict_class(["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"])