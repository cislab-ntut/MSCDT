import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import _tree
from sklearn import tree
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from preprocessing import transform1, transform2, transform3, transform4, transform5, transform6 
from sklearn_DTC_transform import transform
from entity2 import CloudServiceUser

import logging
CSU_Logger = logging.getLogger("CSU")

#from secure import PRIME 
import configparser

config = configparser.ConfigParser()
config.read('MSCDT.cfg')

r_config = configparser.ConfigParser()
r_config.set('DEFAULT', 'CSP0_SEVER_IP', os.environ.get('CSP0_SEVER_IP', config['DEFAULT']['CSP0_SEVER_IP']))
r_config.set('DEFAULT', 'CSP1_SEVER_IP', os.environ.get('CSP1_SEVER_IP', config['DEFAULT']['CSP1_SEVER_IP']))
r_config.set('DEFAULT', 'CSP0_SEVER_PORT', os.environ.get('CSP0_SEVER_PORT', config['DEFAULT']['CSP0_SEVER_PORT']))
r_config.set('DEFAULT', 'CSP1_SEVER_PORT', os.environ.get('CSP1_SEVER_PORT', config['DEFAULT']['CSP1_SEVER_PORT']))
r_config.set('DEFAULT', 'RETRIES', os.environ.get('RETRIES', config['DEFAULT']['RETRIES']))

data_set="weather"

def get_data():
    data = 0
    
    if data_set== "nursery":
            names = ['parents', 'has_nurs', 'form', 'children',
                    'housing', 'finance', 'social', 'health', 'label']
            data_orig = pd.read_csv('data/nursery/nursery.data', header=None, names=names, index_col=False)
            data = transform1(data_orig)
    if data_set=="weather":
            data_orig = pd.read_csv('data/weather/weatherAUS.csv', index_col=False)
            data = transform2(data_orig)
    if data_set== "cleveland":
            names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
            data_orig = pd.read_csv("data/heart-disease/processed.cleveland.data", header=None, names=names, index_col=False)
            data = transform3(data_orig)
    if data_set== "bank":
            data_orig = pd.read_csv('data/bank/bank.csv', index_col=False)
            data = transform4(data_orig)

    return data

### Model train and test
data = get_data()
margin = int(len(data)*0.8)
trainingData = data[:margin].reset_index(drop=True)
testingData = data[margin:].reset_index(drop=True)
x = data.iloc[:,:-1]
y = data.iloc[:, -1]
#print(len(y))

'''
model = DecisionTreeClassifier(min_samples_split=2)
X_train, X_test, y_train, y_test = train_test_split(x, y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("1", classification_report(y_test, pred))
#plt.figure(figsize=(18,12))
#plot_tree(model)
#plt.show()
print("Model depth:",model.tree_.max_depth)
pickle.dump(model, open(data_set, 'wb'))
'''

model = pickle.load(open(data_set, 'rb'))
#print("Depth of the model: ", model.tree_.max_depth)
modelTreeRootNode = transform(model.tree_.feature, (model.tree_.threshold)*10, model.tree_.value, model.classes_)
#print("Attribute length: ",len(model.feature_names_in_))
#print("Model is ready!!")


if __name__ == "__main__":
    csu = CloudServiceUser(r_config, version=3)
    csu.start_connection()

    ### MSCDT preprocessing (online)
    data=testingData.loc[0]
    #print("CSU data:\n",data)
    udata=np.zeros((len(model.feature_names_in_),),dtype=int)
    for i in range(len(udata)):
            udata[i]=data[i]

    CSU_Logger.info("[CSU] Transfrom query data.")
    csu.set_query_data_v3(udata)
    csu.send_query_data_to_csp() ################################## socket OK
    csu.receive_result_items_and_compute_shared_result()  ##[Socket Communication modified]

    z = csu.get_reconstruct_result()
    CSU_Logger.info("[CSU] result: {}".format(z))

    csu.close_csp_connection()
    