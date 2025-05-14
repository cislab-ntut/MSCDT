from phe import paillier
import numpy as np
import random
import os
import sys
import time
from Crypto.Util.number import getPrime
import sympy as sy



SECUREPARAM = 16
PRIME = getPrime(SECUREPARAM+1, os.urandom)
#PRIME=89

def dot_product_triples(n, x0, x1, y0, y1):#, Z0=0, Z1=0, X0=[], Y0=[], X1=[], Y1=[]):
    #if n > 0 & len(X0 = 0):
    X0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    X1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    Y0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    Y1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
    T = random.randint(0, 5)

    Z0 = (X0.dot(Y1)) %PRIME + T
    Z1 = (X1.dot(Y0)) %PRIME - T
    #print("Z", Z0, Z1)
    #print("X: ", X0, X1)
    #print("Y: ", Y0, Y1)
    p0x = (x0 + X0) %PRIME
    p0y = (y0 + Y0) %PRIME
    p1x = (x1 + X1) %PRIME
    p1y = (y1 + Y1) %PRIME
    #print("x0 + X0: ", p0x)
    #print("y0 + Y0: ", p0y)
    #print("x1 + X1: ", p1x)
    #print("y1 + Y1: ", p1y)

    z0 = (x0.dot((y0 + p1y)%PRIME))%PRIME - (Y0.dot(p1x))%PRIME + Z0
    z1 = (x1.dot((y1 + p0y)%PRIME))%PRIME - (Y1.dot(p0x))%PRIME + Z1
    #print("z0: ", z0)
    #print("z1: ", z1)
    return z0, z1


def reconstruct_secret(shares):
    secret = sum(shares) % PRIME
    
    if secret > PRIME // 2:
        secret -= PRIME
    
    return int(secret)

internal_node_num=127
leaf_node_num=128


model_threshold=sy.zeros(internal_node_num,1)
leaf_node=sy.zeros(leaf_node_num,1)
leaf_node_share0=sy.zeros(leaf_node_num,1)
leaf_node_share1=sy.zeros(leaf_node_num,1)
userquery=[-4, 11, 42, -4, 15, 8, -48, 80, 155, 3]# 10 attribtes
model_attr=[]
vb=["attr0","attr1","attr2","attr3","attr4","attr5","attr6","attr7","attr8","attr9"]


###Prepare model and query data
for i in range(internal_node_num):
    model_threshold[i]=i
    model_attr.append(vb[i%10])

for i in range(leaf_node_num):
    leaf_node[i]=i
    leaf_node_share0[i]=random.randint(0,PRIME-1)
    leaf_node_share1[i]=(leaf_node[i]-leaf_node_share0[i])% PRIME

### MSCDT preprocessing
model_share0=sy.zeros(internal_node_num,1)
model_share1=sy.zeros(internal_node_num,1)
start_time = time.time()
for i in range(internal_node_num):
    model_share0[i]=random.randint(0,PRIME-1)
    model_share1[i]=(model_threshold[i]-model_share0[i])% PRIME

end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("ASS enc model threshold 127 items:", execution_time, "mseconds")


user_share0=np.zeros((10,))
user_share1=np.zeros((10,))
start_time = time.time()
for i in range(10):
    user_share0[i]=random.randint(0,PRIME-1)
    user_share1[i]=(userquery[i]-user_share0[i]) % PRIME


end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("ASS enc user query 10 items:", execution_time, "mseconds")

### MSCDT Node Evaluation
start_time = time.time()
for i in range(internal_node_num):
    for a in range(10):
        if model_attr[i]==vb[a]:
            #print("query: ",userquery[a])
            #print("mt: ",model_threshold[i])
            #print("ms0: ",model_share0[i])
            #print("ms1: ",model_share1[i])
            #print("us0: ",user_share0[a])
            #print("us0: ",user_share1[a])
            #model_share0[i]=(user_share0[a]-model_share0[i])
            #model_share1[i]=(user_share1[a]-model_share1[i])
            model_share0[i] = user_share0[a]-model_share0[i]
            model_share1[i] = user_share1[a]-model_share1[i]
            #print(model_share0[i])
            #print(model_share1[i])
            #print(reconstruct_secret((model_share0[i],model_share1[i])))
            
            #print(userquery[a]-model_threshold[i])

end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT node evaluation:", execution_time, "mseconds")

### MSCDT permute nodes
start_time = time.time()
model_share0*=2
model_share1*=2
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT permute nodes:", execution_time, "mseconds")

### MSCDT send model to user and decrypt elements
i = 0
bit_string=[]
start_time = time.time()
while i<internal_node_num:      
    comb=model_share0[i]+model_share1[i]
    if comb<0:
        bit_string.append(0)
        i=i*2+2
    else:
        bit_string.append(1)
        i=i*2+1

label = (2**6)*bit_string[0]+(2**5)*bit_string[1]+(2**4)*bit_string[2]+(2**3)*bit_string[3]+(2**2)*bit_string[4]+(2**1)*bit_string[5]+(2**0)*bit_string[6]
leaf_idx = [0]*leaf_node_num
leaf_idx[label] = 1
#print("label: ",label)
leaf_idx_share0=sy.zeros(leaf_node_num,1)
leaf_idx_share1=sy.zeros(leaf_node_num,1)

for i in range(leaf_node_num):
    leaf_idx_share0[i]=random.randint(0,PRIME-1)
    leaf_idx_share1[i]=(leaf_idx[i]-leaf_idx_share0[i])  % PRIME

end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT generating fake index: ", execution_time, "mseconds")

### MSCDT servers permute and compute SDP function to obtain shared result
start_time = time.time()
#x0 = sy.Matrix([leaf_idx_share0[i] for i in range(leaf_node_num)])
#x1 = sy.Matrix([leaf_idx_share1[i] for i in range(leaf_node_num)])
#y0 = sy.Matrix([leaf_node_share0[i] for i in range(leaf_node_num)])
#y1 = sy.Matrix([leaf_node_share1[i] for i in range(leaf_node_num)])
x0 = leaf_idx_share0
x1 = leaf_idx_share1
y0 = leaf_node_share0
y1 = leaf_node_share1
z0, z1 = dot_product_triples(leaf_node_num, x0, x1, y0, y1)
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT generating final index shares: ", execution_time, "mseconds")

### MSCDT user reconstruct result
start_time = time.time()
z = reconstruct_secret((z0,z1))
end_time = time.time()
execution_time = float(end_time - start_time)*1000
print("MSCDT user reconstruct final index shares: ", execution_time, "mseconds")

print("Reconstructed label: ", z)

x = sy.Matrix([reconstruct_secret((x0[i],x1[i])) for i in range(leaf_node_num)])
y = sy.Matrix([reconstruct_secret((y0[i],y1[i])) for i in range(leaf_node_num)])
print("direct inner product: ", x.dot(y))