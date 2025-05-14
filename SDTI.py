import numpy as np
import random
import os
import sys
import time
from Crypto.Util.number import getPrime
import sympy as sy
import math
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
from entity2 import ModelOwner,CloudServiceProvider0,CloudServiceProvider1,CloudServiceUser,Protocol

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



SECUREPARAM = 20 
PRIME = 2**16

def reconstruct_secret(shares):
    secret = sum(shares) % PRIME
    
    if secret > PRIME // 2:
        secret -= PRIME
    
    return int(secret)


### Model train and test
data = get_data()
margin = int(len(data)*0.8)
trainingData = data[:margin].reset_index(drop=True)
testingData = data[margin:].reset_index(drop=True)
x = data.iloc[:,:-1]
y = data.iloc[:, -1]
#print(len(y))


model = DecisionTreeClassifier(min_samples_split=2)
X_train, X_test, y_train, y_test = train_test_split(x, y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("1", classification_report(y_test, pred))
# plt.figure(figsize=(18,12))
# plot_tree(model)
# plt.show()
print("Model depth:",model.tree_.max_depth)
pickle.dump(model, open(data_set, 'wb'))


model = pickle.load(open(data_set, 'rb'))
print("Depth of the model: ", model.tree_.max_depth)
modelTreeRootNode = transform(model.tree_.feature, (model.tree_.threshold)*10, model.tree_.value, model.classes_)
print("Attribute length: ",len(model.feature_names_in_))
#print("Model is ready!!")

v1_totaltime=[]

v2_totaltime=[]
v3_totaltime=[]
v4_totaltime=[]
SDTI_totaltime=[]

v1_me_totaltime=[]
v1_eud_totaltime=[]
v1_neval_totaltime=[]
v1_per_totaltime=[]
v1_rper_totaltime=[]
v1_fi_totaltime=[]
v1_fr_totaltime=[]
v1_di_totaltime=[]

v2_me_totaltime=[]
v2_eud_totaltime=[]
v2_neval_totaltime=[]
v2_per_totaltime=[]
v2_rper_totaltime=[]
v2_fi_totaltime=[]
v2_fr_totaltime=[]
v2_di_totaltime=[]

v3_me_totaltime=[]
v3_eud_totaltime=[]
v3_neval_totaltime=[]
v3_per_totaltime=[]
v3_rper_totaltime=[]
v3_fi_totaltime=[]
v3_fr_totaltime=[]
v3_di_totaltime=[]

v4_me_totaltime=[]
v4_eud_totaltime=[]
v4_neval_totaltime=[]
v4_per_totaltime=[]
v4_rper_totaltime=[]
v4_fi_totaltime=[]
v4_fr_totaltime=[]
v4_di_totaltime=[]


for k in range(61):
        
        print(k)
        #print("MSCDT v",v)
        mo=ModelOwner(version="sdti")
        csp0 = CloudServiceProvider0()
        csp1 = CloudServiceProvider1()
        csu = CloudServiceUser()
        p = Protocol(version="sdti")

        
        ###Prepare model(offline)
        #print("MO encrypt model and distrubte model shares")
        #start_time = time.time()
        mo.input_model_and_split_into_shares(modelTreeRootNode,model.feature_names_in_,PRIME)
        #end_time = time.time()
        #execution_time = float(end_time - start_time)*1000

        #print("MSCDT model encryption:", execution_time, "mseconds")
        mo.set_shares_to_two_parties(csp0,csp1)

        ### MSCDT preprocessing (online)
        data=testingData.loc[k]
        #print("CSU data:\n",data)
        udata=np.zeros((len(model.feature_names_in_),),dtype=int)
        for i in range(len(udata)):
                udata[i]=data[i]
                
        start_time = time.time()
        #mo.input_model_and_split_into_shares(modelTreeRootNode,model.feature_names_in_,PRIME)
        
        csu.set_query_data_sd(udata,PRIME)

        # end_time = time.time()
        # execution_time = float(end_time - start_time)*1000
        
        csu.send_query_data_to_csp(csp0,csp1)

        #start_time = time.time()
        eval_time=p.node_eval(csp0,csp1,model.feature_names_in_,PRIME)
        # end_time = time.time()
        # execution_time = float(end_time - start_time)*1000

        start_time = time.time()
        
        p.dfs_ifgen(csp0,csp1,csp0.model_share0_root,csp1.model_share1_root,[1,1])
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        print("Tree inference time: ",execution_time+eval_time)

        result= (csp0.resultshare0()+csp1.resultshare1())%PRIME
        #print(result)
        csp0.set_resultshare0(0)
        csp1.set_resultshare1(0)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        SDTI_totaltime.append(execution_time)

#print("SDTI Average time: ", sum(SDTI_totaltime)/len(SDTI_totaltime))

                