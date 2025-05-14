from phe import paillier
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
import graphviz 
import pandas as pd
from preprocessing import transform1, transform2, transform3, transform4, transform5, transform6 
from sklearn_DTC_transform import transform
from entity2 import ModelOwner,CloudServiceProvider0,CloudServiceProvider1,CloudServiceUser,Protocol

data_set=1
thresholds=[]
def get_data():
    data = 0
    
    if data_set== 1:
            names = ['parents', 'has_nurs', 'form', 'children',
                    'housing', 'finance', 'social', 'health', 'label']
            data_orig = pd.read_csv('data/nursery/nursery.data', header=None, names=names, index_col=False)
            data = transform1(data_orig)
    if data_set== 2:
            data_orig = pd.read_csv('data/weather/weatherAUS.csv', index_col=False)
            data = transform2(data_orig)
    if data_set== 3:
            names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
            data_orig = pd.read_csv("data/heart-disease/processed.cleveland.data", header=None, names=names, index_col=False)
            data = transform3(data_orig)
    if data_set== 4:
            data_orig = pd.read_csv('data/bank/bank.csv', index_col=False)
            data = transform4(data_orig)
    if data_set== 5:
            data_orig = pd.read_csv('data/malware/malware.csv', index_col=False)
            data = transform5(data_orig)
    if data_set== 6:
            data_orig = pd.read_csv('data/學期成績/學期成績.csv', index_col=False)
            data = transform6(data_orig)
        # case 7:
        # case 8:
    return data



SECUREPARAM = 16
PRIME = getPrime(SECUREPARAM+1, os.urandom)
#PRIME=89

def reconstruct_secret(shares):
    secret = sum(shares) % PRIME
    
    if secret > PRIME // 2:
        secret -= PRIME
    
    return int(secret)


### Model train and test
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
modelTreeRootNode = transform(model.tree_.feature, (model.tree_.threshold)*10, model.tree_.value, model.classes_)
#print("Model is ready!!")
mo=ModelOwner()
csp0 = CloudServiceProvider0()
csp1 = CloudServiceProvider1()
csu = CloudServiceUser()
p = Protocol()


###Prepare model(offline)
print("MO encrypt model and distrubte model shares")
start_time = time.time()
mo.input_model_and_split_into_shares(modelTreeRootNode,model.feature_names_in_,PRIME)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT model encryption:", execution_time, "mseconds")
mo.set_shares_to_two_parties(csp0,csp1)


### MSCDT preprocessing (online)
print("CSU encrypt query data")
#print("CSU data: \n",testingData.loc[0])

start_time = time.time()
#mo.input_model_and_split_into_shares(modelTreeRootNode,model.feature_names_in_,PRIME)
mo.set_shares_to_two_parties(csp0,csp1)
csu.set_query_data(testingData.loc[100],PRIME)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT user query encryption:", execution_time, "mseconds")

#print("CSU generate and send query data share")
csu.send_query_data_to_csp(csp0,csp1)

### MSCDT Node Evaluation
start_time = time.time()
p.node_eval(csp0,csp1,model.feature_names_in_,PRIME)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT node evaluation:", execution_time, "mseconds")

### MSCDT permute nodes

start_time = time.time()
#print("Tree before permute:\n")
#p.print_tree(csp0,csp1)
p.permute(csp0,csp1)
#print("Tree after permute:\n")
#p.print_tree(csp0,csp1)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT permute nodes:", execution_time, "mseconds")

### MSCDT server send model and total leave node number to user and decrypt elements
leaf_num=p.leafnode_num()
#print("Total leaf node number: ", p.leafnode_num())
start_time = time.time()
index=p.find_fake_index(csp0,csp1)
#print(index)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT generating fake index: ", execution_time, "mseconds")

fake_leaf_idx=np.zeros((leaf_num,),dtype=int)
fake_leaf_idx[index]=1
fake_leaf_idx_share0=np.zeros((leaf_num,),dtype=int)
fake_leaf_idx_share1=np.zeros((leaf_num,),dtype=int)

for i in range(leaf_num): #generate fake index vector shares
    fake_leaf_idx_share0[i] = random.randint(0,PRIME-1)
    fake_leaf_idx_share1[i] = (fake_leaf_idx[i]-fake_leaf_idx_share0[i])%PRIME


### MSCDT servers compute SDP function to obtain shared result

start_time = time.time()
z0, z1 = p.SDP(fake_leaf_idx_share0,fake_leaf_idx_share1)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT generating final index shares: ", execution_time, "mseconds")

### MSCDT user reconstruct result
start_time = time.time()
z = reconstruct_secret((z0,z1))
print("Reconstructed label: ", z)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
#print("MSCDT user reconstruct final index shares: ", execution_time, "mseconds")

print("Model depth: ", model.tree_.max_depth)
#print("MSCDT online inference time cost: ", execution_time, "mseconds")


#x = sy.Matrix([reconstruct_secret((x0[i],x1[i])) for i in range(leaf_node_num)])
#y = sy.Matrix([reconstruct_secret((y0[i],y1[i])) for i in range(leaf_node_num)])
#print("direct inner product: ", x.dot(y))