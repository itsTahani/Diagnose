

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing



from flask import *
import json
import time
app = Flask(__name__)


@app.route('/Diagnose', methods=['Get'])
def Diagnose():
  user = str(request.args.get('Symptoms'))
  # user = "Diabest"
# http://127.0.0.1/Diagnose?Symptoms=0101010101010000000000000000000000000000000000000001010101010100000000000000000000000000000000000000010101010101000000000000000000000000000000
  arr = list(map(int,user))
  # Diagnose here
  training = pd.read_csv('Training_2.1.csv')
  x =training.drop('prognosis',axis=1)
  y = training['prognosis']
  # reduced_data = training.groupby(training['prognosis']).max()
  # le = preprocessing.LabelEncoder()
  # le.fit(y)
  # y = le.transform(y)
  #Split the data into training and test
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


  #SVC
  svc= SVC(probability=True,kernel='linear')
  svc.fit(x_train ,y_train)

  #Predict Diagnose
  # svc.predict([0,0,0,0,0])
  svc.predict([arr])


  data_set = {'Page': 'Home','Message': f'{svc.predict([arr])}', 'Time': time.time()}
  json_dump = json.dumps(data_set)
  return json_dump
if(__name__ == '__main__'):
    app.run(port=80)

