from phe import paillier
import numpy as np
import random
import os
import sys
import time
from Crypto.Util.number import getPrime
import gmpy2
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
from entityHE import ModelOwner,CloudServiceProvider,CloudServiceUser,Protocol
data_set="weather"

print(data_set)
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
'''
model = DecisionTreeClassifier(min_samples_split=2)
data = get_data()
margin = int(len(data)*0.8)
trainingData = data[:margin].reset_index(drop=True)
testingData = data[margin:].reset_index(drop=True)
x = data.iloc[:,:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("1", classification_report(y_test, pred))
#plt.figure(figsize=(18,12))
#plot_tree(model)
#plt.show()
print("Model depth:",model.tree_.max_depth)
'''
data = get_data()
margin = int(len(data)*0.8)
trainingData = data[:margin].reset_index(drop=True)
testingData = data[margin:].reset_index(drop=True)
x = data.iloc[:,:-1]
y = data.iloc[:, -1]
model = pickle.load(open(data_set, 'rb'))
modelTreeRootNode = transform(model.tree_.feature, (model.tree_.threshold)*10, model.tree_.value, model.classes_)
#print("Model is ready!!")

mo=ModelOwner()
mo.gen_keypair()
csp = CloudServiceProvider()
csu = CloudServiceUser()
csu.gen_keypair()
mo.get_upk(csu)
p = Protocol()

totaltime=[]

eud_totaltime=[]
neval_totaltime=[]
per_totaltime=[]
rper_totaltime=[]
fi_totaltime=[]
fr_totaltime=[]
di_totaltime=[]
for i in range(100):
        print(i)
        ### HEDT encrypt model
        #print("MO encrypt model and distrubte encrypted model")
        #start_time = time.time()
        mo.input_model_and_encrypt(modelTreeRootNode,model.feature_names_in_)
        #end_time = time.time()
        #execution_time = float(end_time - start_time)*1000
        #print("MSCDT model encryption:", execution_time, "mseconds")
        mo.set_model_to_csp(csp)




        ### HEDT user encrypt user query
        #print("CSU encrypt user query")
        data=testingData.loc[i]
        #print("CSU data:\n",data)
        udata=np.zeros((len(model.feature_names_in_),),dtype=int)
        for i in range(len(udata)):
                udata[i]=data[i]
        start_time = time.time()
        csu.set_query_data(udata, mo)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        eud_totaltime.append(execution_time)

        #print("Paillier enc user query:", execution_time, "mseconds")

        ### HEDT user send query data to csp
        csu.send_query_data_to_csp(csp)

        ### HEDT Node Evaluation
        start_time = time.time()
        p.node_eval(csp,model.feature_names_in_)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        neval_totaltime.append(execution_time)
        #print("HEDT node evaluation:", execution_time, "mseconds")

        ### HEDT Permute
        start_time = time.time()
        p.permute(csp)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        per_totaltime.append(execution_time)
        #print("HEDT permute nodes:", execution_time, "mseconds")

        ### HEDT send model to MO and generate fake leaf node idx vector
        start_time = time.time()
        fake_index_vector = p.find_fake_index(mo,csp,csu)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        fi_totaltime.append(execution_time)
        #print("HEDT generating fake index: ", execution_time, "mseconds")

        ### HEDT server reverse permute and reencrypt leaf_idx
        start_time = time.time()
        p.reverse_permute(csp)#permute model
        real_index_vector=p.reverse_permute_fake_index(csp,csu,fake_index_vector)#permute fake index vector and refresh it 
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        rper_totaltime.append(execution_time)
        #print("HEDT refresh index cipher and find real index: ", execution_time, "mseconds")


        result=0
        ### HEDT MO find the result
        start_time = time.time()
        result=p.hedt_find_result(mo,real_index_vector)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        fr_totaltime.append(execution_time)
        #print("HEDT MO find the encrypted result: ", execution_time, "mseconds")

        ### HEDT user decrypt the final result
        start_time = time.time()
        result=p.user_decrypt_result(csu,result)
        end_time = time.time()
        execution_time = float(end_time - start_time)*1000
        #print("Time for once inference: ",execution_time, "mseconds")
        #totaltime.append(execution_time)
        di_totaltime.append(execution_time)
        #print("HEDT user decrypt the result from MO: ", execution_time, "mseconds")
        print("Reconstruced result: ", result)

#print("Average time: ", sum(totaltime)/len(totaltime))
'''
print("Average encrypting user data time: ", sum(eud_totaltime)/len(eud_totaltime))
print("Average node evaluation time: ", sum(neval_totaltime)/len(neval_totaltime))
print("Average permute node time: ", sum(per_totaltime)/len(per_totaltime))
print("Average find fake index time: ",sum(fi_totaltime)/len(fi_totaltime))
print("Average reverse permute node time: ",sum(rper_totaltime)/len(rper_totaltime))
print("Average find result time: ",sum(fr_totaltime)/len(fr_totaltime))
print("Average decrypt result time: ",sum(di_totaltime)/len(di_totaltime))
'''
