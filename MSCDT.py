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

from secure import PRIME 
import configparser

config = configparser.ConfigParser()
config.read('MSCDT.cfg')

r_config = configparser.ConfigParser()
r_config.set('DEFAULT', 'CSP0_SEVER_IP', os.environ.get('CSP0_SEVER_IP', config['DEFAULT']['CSP0_SEVER_IP']))
r_config.set('DEFAULT', 'CSP1_SEVER_IP', os.environ.get('CSP1_SEVER_IP', config['DEFAULT']['CSP1_SEVER_IP']))
r_config.set('DEFAULT', 'CSP0_SEVER_PORT', os.environ.get('CSP0_SEVER_PORT', config['DEFAULT']['CSP0_SEVER_PORT']))
r_config.set('DEFAULT', 'CSP1_SEVER_PORT', os.environ.get('CSP1_SEVER_PORT', config['DEFAULT']['CSP1_SEVER_PORT']))
r_config.set('DEFAULT', 'RETRIES', os.environ.get('RETRIES', config['DEFAULT']['RETRIES']))
# print(r_config.sections())
# print(r_config['DEFAULT']['CSP0_SEVER_IP'])
# print(r_config['DEFAULT']['CSP0_SEVER_PORT'])
# print(r_config['DEFAULT']['CSP1_SEVER_IP'])
# print(r_config['DEFAULT']['CSP1_SEVER_PORT'])

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



# SECUREPARAM = 20 
# PRIME = getPrime(SECUREPARAM+1, os.urandom)

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
print("Depth of the model: ", model.tree_.max_depth)
modelTreeRootNode = transform(model.tree_.feature, (model.tree_.threshold)*10, model.tree_.value, model.classes_)
print("Attribute length: ",len(model.feature_names_in_))
#print("Model is ready!!")

# X_train, X_test, y_train, y_test = train_test_split(x, y)
# model.fit(X_train, y_train)
# print("model predict: ", model.predict(X_test))
# exit()

v1_totaltime=[]

v2_totaltime=[]
v3_totaltime=[]
v4_totaltime=[]

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


version1_result_list = []
version2_result_list = []
version3_result_list = []
version4_result_list = []

for k in range(60):
    #print(k)
    #print("MSCDT v",v)
    #mo=ModelOwner(version=v)
    csp0 = CloudServiceProvider0(r_config, version=3)
    #time.sleep(1)
    csp1 = CloudServiceProvider1(r_config, version=3)
    csp1.start_connection()
    #time.sleep(1)
    mo=ModelOwner(r_config, version=3)
    mo.start_connection()

    csu = CloudServiceUser(r_config, version=3)
    csu.start_connection()
    p = Protocol(version=3)
    #print("All connection established")
    #print("version: ", v)
    ###Prepare model(offline)
    #print("MO encrypt model and distrubte model shares")
    #start_time = time.time()
    mo.input_model_and_split_into_shares(modelTreeRootNode,model.feature_names_in_,PRIME)
    #if v==3:
    A0,A1=mo.gen_permute_matrix(csp0,csp1) ########################### socket OK
#     if v==4:
#             AM, M_inv = mo.gen_pk_sk_matrix(csp0, csp1)
    #end_time = time.time()
    #execution_time = float(end_time - start_time)*1000

    #print("MSCDT model encryption:", execution_time, "mseconds")
    mo.set_shares_to_two_parties(csp0,csp1) ################################## socket OK
    mo.close_csp_connection()               ##[Socket Communication modified]
    # if v==1:
    #         v1_me_totaltime.append(execution_time)
    # if v==2:
    #         v2_me_totaltime.append(execution_time)
    # if v==3:
    #         v3_me_totaltime.append(execution_time)
    # if v==4:
    #         v4_me_totaltime.append(execution_time)


    ### MSCDT preprocessing (online)
    data=testingData.loc[k]
    #print("CSU data:\n",data)
    udata=np.zeros((len(model.feature_names_in_),),dtype=int)
    for i in range(len(udata)):
            udata[i]=data[i]
            
    start_time = time.time()
    #mo.input_model_and_split_into_shares(modelTreeRootNode,model.feature_names_in_,PRIME)
    # if v==1:
    #         csu.set_query_data(udata,PRIME)
    # if v==2:
    #         csu.set_query_data_v2(udata,PRIME)
    # if v==3:
    csu.set_query_data_v3(udata, PRIME)
    # if v==4:
    #         csu.set_query_data_v4(udata, AM, PRIME)

    # end_time = time.time()
    # execution_time = float(end_time - start_time)*1000
    # if v==1:
    #         v1_eud_totaltime.append(execution_time)
    # if v==2:
    #         v2_eud_totaltime.append(execution_time)
    # if v==3:
    #         v3_eud_totaltime.append(execution_time)
    # if v==4:
    #         v4_eud_totaltime.append(execution_time)
    #print("MSCDT user query encryption:", execution_time, "mseconds")

    #print("CSU generate and send query data share")
    csu.send_query_data_to_csp(csp0,csp1) ################################## socket OK
    csu.receive_result_items_and_compute_shared_result()  ##[Socket Communication modified]

    ### MSCDT Node Evaluation
    #start_time = time.time()
    eval_time= p.node_eval(csp0,csp1,model.feature_names_in_,PRIME)######################testing
    #print("node_eval end")
    # end_time = time.time()
    # execution_time = float(end_time - start_time)*1000
    # if v==1:
    #         v1_neval_totaltime.append(execution_time)
    # if v==2:
    #         v2_neval_totaltime.append(execution_time)
    # if v==3:
    #         v3_neval_totaltime.append(execution_time)
    # if v==4:
    #         v4_neval_totaltime.append(execution_time)
    #print("MSCDT node evaluation:", execution_time, "mseconds")

    ### MSCDT permute nodes

    start_time = time.time()
    #print("Tree before permute:\n")
    #p.print_tree(csp0,csp1)
    p.permute(csp0,csp1)
    #print("permute end")
    #print("Tree after permute:\n")
    #p.print_tree(csp0,csp1)
    # end_time = time.time()
    # execution_time = float(end_time - start_time)*1000
    # if v==1:
    #         v1_per_totaltime.append(execution_time)
    # if v==2:
    #         v2_per_totaltime.append(execution_time)
    # if v==3:
    #         v3_per_totaltime.append(execution_time)
    # if v==4:
    #         v4_per_totaltime.append(execution_time)
    #print("MSCDT permute nodes:", execution_time, "mseconds")

    ### MSCDT server send model and total leave node number to user and decrypt elements
    leaf_num=p.leafnode_num()
    #print("[origin] leaf_num: ", leaf_num)
    #print("Total leaf node number: ", p.leafnode_num())
    #start_time = time.time()
    index=p.find_fake_index(csp0,csp1)

    #print("[orignal] idex: ", index)
    #print(index)

    #print("MSCDT generating fake index: ", execution_time, "mseconds")

    fake_leaf_idx=np.zeros((leaf_num,),dtype=int)
    fake_leaf_idx[index]=1
    fake_leaf_idx_share0=np.random.randint(0,PRIME-1,(leaf_num,))
    fake_leaf_idx_share1=np.zeros((leaf_num,),dtype=int)
    fake_leaf_idx_share1 = (fake_leaf_idx-fake_leaf_idx_share0)%PRIME
    # end_time = time.time()

    # execution_time = float(end_time - start_time)*1000
    # if v==1:
    #         v1_fi_totaltime.append(execution_time)
    # if v==2:
    #         v2_fi_totaltime.append(execution_time)
    # if v==3:
    #         v3_fi_totaltime.append(execution_time)
    # if v==4:
    #         v4_fi_totaltime.append(execution_time)
    #for i in range(leaf_num): #generate fake index vector shares
    #       fake_leaf_idx_share0[i] = random.randint(0,PRIME-1)
    #        fake_leaf_idx_share1[i] = (fake_leaf_idx[i]-fake_leaf_idx_share0[i])%PRIME


    ### MSCDT servers compute SDP function to obtain shared result

    #start_time = time.time()
    z0, z1 = p.SDP(fake_leaf_idx_share0,fake_leaf_idx_share1)

    end_time = time.time()
    execution_time = float(end_time - start_time)*1000
    #print("Tree inference time: ", execution_time+eval_time)
    # if v==1:
    #         v1_fr_totaltime.append(execution_time)
    # if v==2:
    #         v2_fr_totaltime.append(execution_time)
    # if v==3:
    #         v3_fr_totaltime.append(execution_time)
    # if v==4:
    #         v4_fr_totaltime.append(execution_time)
    #print("MSCDT generating final index shares: ", execution_time, "mseconds")

    ### MSCDT user reconstruct result
    #start_time = time.time()
    z = reconstruct_secret((z0,z1))
    #print("Reconstructed label: ", z)
    end_time = time.time()
    execution_time = float(end_time - start_time)*1000
    # if v==1:
    #         v1_di_totaltime.append(execution_time)
    # if v==2:
    #         v2_di_totaltime.append(execution_time)
    # if v==3:
    #         v3_di_totaltime.append(execution_time)
    # if v==4:
    #         v4_di_totaltime.append(execution_time)
    #print("MSCDT user reconstruct final index shares: ", execution_time, "mseconds")
    #print("MSCDT online inference time cost: ", execution_time, "mseconds")

    # if v==1:
    #         v1_totaltime.append(execution_time)
    #         version1_result_list.append(csu.get_reconstruct_result())
    # if v==2:
    #         v2_totaltime.append(execution_time)
    #         version2_result_list.append(csu.get_reconstruct_result())
    # if v==3:
    v3_totaltime.append(execution_time)
    version3_result_list.append(csu.get_reconstruct_result())
    # if v==4:
    #         v4_totaltime.append(execution_time)
    #         version4_result_list.append(csu.get_reconstruct_result())
    #print("version ", v, " end")
    csu.close_csp_connection()
    #print("csu disconnected")
    csp0.cspthread.join()
    csp1.cspthread.join()
    #print("all csp joined")
    del mo
    del csu
    del csp0
    del csp1
    print("========================================================================")

#print("version1_result_list: ", version1_result_list)
#print("version2_result_list: ", version2_result_list)
print("version3_result_list: ", version3_result_list)
#print("version4_result_list: ", version4_result_list)

'''
print("v1 Average time: ", sum(v1_totaltime)/len(v1_totaltime))
print("v2 Average time: ", sum(v2_totaltime)/len(v2_totaltime))
print("v3 Average time: ", sum(v3_totaltime)/len(v3_totaltime))
print("v4 Average time: ", sum(v4_totaltime)/len(v4_totaltime))
'''

'''
print("v1 Average encrypting model time: ", sum(v1_me_totaltime)/len(v1_me_totaltime))
print("v1 Average encrypting user data time: ", sum(v1_eud_totaltime)/len(v1_eud_totaltime))
print("v1 Average node evaluation+permute node time: ", (sum(v1_neval_totaltime)+sum(v1_per_totaltime))/len(v1_neval_totaltime))
print("v1 Average find fake index+find result  time: ",(sum(v1_fi_totaltime)+sum(v1_fr_totaltime))/len(v1_fi_totaltime))
print("v1 Average decrypt result time: ",sum(v1_di_totaltime)/len(v1_di_totaltime))
print("\n")

print("v2 Average encrypting model time: ", sum(v2_me_totaltime)/len(v2_me_totaltime))
print("v2 Average encrypting user data time: ", sum(v2_eud_totaltime)/len(v2_eud_totaltime))
print("v2 Average node evaluation+permute node time: ", (sum(v2_neval_totaltime)+sum(v2_per_totaltime))/len(v2_neval_totaltime))
print("v2 Average find fake index+find result time: ",(sum(v2_fi_totaltime)+sum(v2_fr_totaltime))/len(v2_fi_totaltime))
print("v2 Average decrypt result time: ",sum(v2_di_totaltime)/len(v2_di_totaltime))
print("\n")

print("v3 Average encrypting model time: ", sum(v3_me_totaltime)/len(v3_me_totaltime))
print("v3 Average encrypting user data time: ", sum(v3_eud_totaltime)/len(v3_eud_totaltime))
print("v3 Average node evaluation+permute node time: ", (sum(v3_neval_totaltime)+sum(v3_per_totaltime))/len(v3_neval_totaltime))
print("v3 Average find fake index+find result time: ",(sum(v3_fi_totaltime)+sum(v3_fr_totaltime))/len(v3_fi_totaltime))
print("v3 Average decrypt result time: ",sum(v3_di_totaltime)/len(v3_di_totaltime))
print("\n")

print("v4 Average encrypting model time: ", sum(v4_me_totaltime)/len(v4_me_totaltime))
print("v4 Average encrypting user data time: ", sum(v4_eud_totaltime)/len(v4_eud_totaltime))
print("v4 Average node evaluation+permute node time: ", (sum(v4_neval_totaltime)+sum(v4_per_totaltime))/len(v4_neval_totaltime))
print("v4 Average find fake index+find result time: ",(sum(v4_fi_totaltime)+sum(v4_fr_totaltime))/len(v4_fi_totaltime))
print("v4 Average decrypt result time: ",sum(v4_di_totaltime)/len(v4_di_totaltime))
'''

                #x = sy.Matrix([reconstruct_secret((x0[i],x1[i])) for i in range(leaf_node_num)])
                #y = sy.Matrix([reconstruct_secret((y0[i],y1[i])) for i in range(leaf_node_num)])
                #print("direct inner product: ", x.dot(y))
