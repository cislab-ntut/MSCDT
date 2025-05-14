import socket
import time
from structure2 import Node, Timer
import numpy as np
#import galois
import random

from ot import *
import sympy as sy
from Crypto.Util.number import inverse
p_for_beaver=2**16

# CSP0_SEVER_IP = "127.0.0.1"                                                ##[Socket Communication modified]
# CSP0_SEVER_PORT = 8888                                                     ##[Socket Communication modified]
# CSP1_SEVER_IP = "127.0.0.1"                                                ##[Socket Communication modified]
# CSP1_SEVER_PORT = 8887                                                     ##[Socket Communication modified]
import pickle

import queue
import threading                                                           ##[Socket Communication modified]
from secure import PRIME, PRIME_SIZE                                       ##[Socket Communication modified]
def int_array_to_bytes(arr, byte_order='big', byte_size=PRIME_SIZE):       ##[Socket Communication modified]
    return b''.join(int(i).to_bytes(byte_size, byte_order) for i in arr)   ##[Socket Communication modified]

def bytes_to_int_array(byte_data, byte_order='big', byte_size=PRIME_SIZE): ##[Socket Communication modified]
    return [int.from_bytes(byte_data[i:i+byte_size], byte_order)           ##[Socket Communication modified]
            for i in range(0, len(byte_data), byte_size)]                  ##[Socket Communication modified]

def tuple_array_to_bytes(tuple_array, byte_size=1, byte_order='big'):
    return b''.join(
        b''.join(int(i).to_bytes(byte_size, byte_order, signed=True) for i in tup)
        for tup in tuple_array
    )

def bytes_to_tuple_array(byte_data, tuple_size, byte_size=1, byte_order='big'):
    step = byte_size * tuple_size
    return [
        tuple(int.from_bytes(byte_data[i+j:i+j+byte_size], byte_order,signed=True)
              for j in range(0, step, byte_size))
        for i in range(0, len(byte_data), step)
    ]


select_attr=0
tree_eval=0

### [Reconstruction + attribute-hiding]
### SecDT plus ver1: generate vb + matrix A --> A dot vb = 
### SecDT plus ver2: generate vb + matrix A + invertible(nonsingilar) matrix S + S^(-1)
def reconstruct_secret(shares,PRIME):
    secret = sum(shares) % PRIME
    
    if secret > PRIME // 2:
        secret -= PRIME
    
    return int(secret)

def dot_product_triples(n, x0, x1, y0, y1,PRIME):#, Z0=0, Z1=0, X0=[], Y0=[], X1=[], Y1=[]:

    X0 = np.random.randint(PRIME-1, size=(n,),dtype=int)
    X1 = np.random.randint(PRIME-1, size=(n,),dtype=int)
    Y0 = np.random.randint(PRIME-10, size=(n,),dtype=int)
    Y1 = np.random.randint(PRIME-1, size=(n,),dtype=int)
    T = random.randint(0, PRIME-1)
    Z0 = ((X0.dot(Y1)) + T)%PRIME
    Z1 = ((X1.dot(Y0)) - T)%PRIME

    p0x = (x0 + X0) %PRIME
    p0y = (y0 + Y0) %PRIME
    p1x = (x1 + X1) %PRIME
    p1y = (y1 + Y1) %PRIME

    z0 = (x0.dot((y0 + p1y)) - Y0.dot(p1x) + Z0)%PRIME
    z1 = (x1.dot((y1 + p0y)) - Y1.dot(p0x) + Z1)%PRIME

    return z0, z1

def MVM(x0, x1, y0, y1,csp0_triples,csp1_triples,PRIME):#, Z0=0, Z1=0, X0=[], Y0=[], X1=[], Y1=[]):

    p0x = (x0 + csp0_triples[0])%PRIME
    p0y = (y0 + csp0_triples[1])%PRIME
    p1x = (x1 + csp1_triples[0])%PRIME
    p1y = (y1 + csp1_triples[1])%PRIME

    z0=((np.dot(x0,(y0+p1y)))- (np.dot(p1x,csp0_triples[1]))+csp0_triples[2])%PRIME
    z1=((np.dot(x1,(y1+p0y)))- (np.dot(p0x,csp1_triples[1]))+csp1_triples[2])%PRIME

    return z0, z1

def ZDMVM(x0, x1, y0, y1,csp0_triples,csp1_triples,PRIME):
    
    E_0 = (x0 - csp0_triples[0])%PRIME
    E_1 = (x1 - csp1_triples[0])%PRIME
    f_0 = (y0 - csp0_triples[1])%PRIME
    f_1 = (y1 - csp1_triples[1])%PRIME 
    E = (E_0+E_1)%PRIME
    f = (f_0+f_1)%PRIME
    Ef = (np.dot(E,f))%PRIME
    Eg2_0 = (np.dot(E,csp0_triples[1]))%PRIME
    Eg2_1 = (np.dot(E,csp1_triples[1]))%PRIME
    G1_0f = (np.dot(csp0_triples[0],f))%PRIME
    G1_1f = (np.dot(csp1_triples[0],f))%PRIME

    z0=(0*Ef+Eg2_0+G1_0f+csp0_triples[2])%PRIME
    z1=(1*Ef+Eg2_1+G1_1f+csp1_triples[2])%PRIME
    
    return z0, z1

def additive_mul(a_0,a_1,b_0,b_1,t1_0,t1_1,t2_0,t2_1,t3_0,t3_1):
    e_0= (a_0-t1_0)%2
    e_1= (a_1-t1_1)%2
    f_0= (b_0-t2_0)%2
    f_1= (b_1-t2_1)%2
    e= (e_0+e_1)%2
    f= (f_0+f_1)%2
    ab_0= (0*e*f+t1_0*f+t2_0*e+t3_0)%2
    ab_1= (1*e*f+t1_1*f+t2_1*e+t3_1)%2
    return [ab_0,ab_1]

def additive_mul_prime(a_0,a_1,b_0,b_1,t1_0,t1_1,t2_0,t2_1,t3_0,t3_1,prime):
    e_0= (a_0-t1_0)%prime
    e_1= (a_1-t1_1)%prime
    f_0= (b_0-t2_0)%prime
    f_1= (b_1-t2_1)%prime
    e= (e_0+e_1)%prime
    f= (f_0+f_1)%prime
    ef = (e * f) % prime
    t1f_0 = (t1_0 * f) % prime
    t2e_0 = (t2_0 * e) % prime
    t1f_1 = (t1_1 * f) % prime
    t2e_1 = (t2_1 * e) % prime
    ab_0= (0*ef+t1f_0+t2e_0+t3_0)%prime
    ab_1= (1*ef+t1f_1+t2e_1+t3_1)%prime
    #ab_0= (0*e*f+t1_0*f+t2_0*e+t3_0)%prime
    #ab_1= (1*e*f+t1_1*f+t2_1*e+t3_1)%prime
    return [ab_0,ab_1]

class ModelOwner():

    def __init__(self, config, version=1):#(self, model_tree_root_node):
        self._root_node = None
        self.prime=None
        self.attrlist = None
        self._root_node_shares = None
        self.internal_node_attribute=[]
        self.internal_node_num=0
        self.leaf_node_num=0
        self.version = version

        self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        self.csp0_server.connect((config['DEFAULT']['CSP0_SEVER_IP'], int(config['DEFAULT']['CSP0_SEVER_PORT'])))           ##[Socket Communication modified]
        self.csp0_server.send('MO'.encode())                                 ##[Socket Communication modified]
        self.csp1_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        self.csp1_server.connect((config['DEFAULT']['CSP1_SEVER_IP'], int(config['DEFAULT']['CSP1_SEVER_PORT'])))           ##[Socket Communication modified]
        self.csp1_server.send('MO'.encode())                                 ##[Socket Communication modified]

    def input_model_and_split_into_shares(self, root_node,attrl,prime):
        self.prime=prime 
        self.attrlist=attrl
        self._root_node = root_node
        return self.split_model_into_shares()
    
    def get_model(self) -> Node:
        return self._root_node

    def split_model_into_shares(self) -> list:
        self.internal_node_num=0
        self.leaf_node_num=0
        if self._root_node == None:
            print("Please input model.")
            return None
        
        if self.version=="sdti":
            self._root_node_shares = self._build_shares_sd(self._root_node)
        
        if self.version=="zdwnn":
            self._root_node_shares = self._build_shares_zd(self._root_node)

        if self.version==1:
            self._root_node_shares = self._build_shares(self._root_node)
            #print("Internal nodes: ", self.internal_node_num)
            #print("Leaf nodes: ",self.leaf_node_num)
        if self.version==2:
            self._root_node_shares = self._build_shares_v2(self._root_node)
        if self.version==3:
            self._root_node_shares = self._build_shares_v3(self._root_node)
        if self.version==4:
            self._root_node_shares = self._build_shares_v4(self._root_node)
        #print("MO self._root_node_shares: ", self._root_node_shares)
        return self._root_node_shares

    # Combine copy1, cop2 into one function
    def _build_shares_zd(self, original_node):
        share0_left_child = None
        share0_right_child = None
        share1_left_child = None
        share1_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        if original_node.is_leaf_node():
            #self.leaf_node_num+=1
            labels0=random.randint(0,self.prime-1)
            #Split leaf nodes into shares
            return [Node(attribute=original_node.attribute(), 
                        threshold=labels0, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=(original_node.threshold() - labels0 ) % self.prime, is_leaf_node=True)]
        else:
            #self.internal_node_num+=1
            #print(original_node.threshold())
            thresholdShare0 = random.randint(0,self.prime-1)
            thresholdShare1 = (original_node.threshold()*10 - thresholdShare0 ) % self.prime

            share0_left_child, share1_left_child = self._build_shares_zd(original_node.left_child())
            share0_right_child, share1_right_child = self._build_shares_zd(original_node.right_child())

            self.internal_node_num+=1
            self.internal_node_attribute.append(original_node.attribute())
            return [Node(attribute=self.internal_node_num-1, #i-th internal node
                        threshold=thresholdShare0, 
                        left_child=share0_left_child, 
                        right_child=share0_right_child, 
                        is_leaf_node=False),
                    Node(attribute=self.internal_node_num-1, 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False)]

    def _build_shares_sd(self, original_node):
        share0_left_child = None
        share0_right_child = None
        share1_left_child = None
        share1_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        if original_node.is_leaf_node():
            #self.leaf_node_num+=1
            labels0=random.randint(0,2**16)
            #Split leaf nodes into shares
            return [Node(attribute=original_node.attribute(), 
                        threshold=labels0, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=(original_node.threshold() - labels0 ) % self.prime, is_leaf_node=True)]
        else:
            #self.internal_node_num+=1
            #print(original_node.threshold())
            attribute_share0 = random.randint(0,2**16)
            attribute_share1 = (original_node.attribute()-attribute_share0)%self.prime
            thresholdShare0 = random.randint(0,2**16)
            thresholdShare1 = (original_node.threshold()*10 - thresholdShare0 ) % self.prime

            share0_left_child, share1_left_child = self._build_shares_sd(original_node.left_child())
            share0_right_child, share1_right_child = self._build_shares_sd(original_node.right_child())

            return [Node(attribute=attribute_share0, 
                        threshold=thresholdShare0, 
                        left_child=share0_left_child, 
                        right_child=share0_right_child, 
                        is_leaf_node=False),
                    Node(attribute=attribute_share1, 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False)]
    
    
    def _build_shares(self, original_node):
        share0_left_child = None
        share0_right_child = None
        share1_left_child = None
        share1_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        if original_node.is_leaf_node():
            #self.leaf_node_num+=1
            labels0=random.randint(0,self.prime-1)
            #Split leaf nodes into shares
            return [Node(attribute=original_node.attribute(), 
                        threshold=labels0, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=(original_node.threshold() - labels0 ) % self.prime, is_leaf_node=True)]
        else:
            #self.internal_node_num+=1
            #print(original_node.threshold())
            thresholdShare0 = random.randint(0,self.prime-1)
            thresholdShare1 = (original_node.threshold()*10 - thresholdShare0 ) % self.prime

            share0_left_child, share1_left_child = self._build_shares(original_node.left_child())
            share0_right_child, share1_right_child = self._build_shares(original_node.right_child())

            return [Node(attribute=original_node.attribute(), 
                        threshold=thresholdShare0, 
                        left_child=share0_left_child, 
                        right_child=share0_right_child, 
                        is_leaf_node=False),
                    Node(attribute=original_node.attribute(), 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False)]
    

    def _build_shares_v2(self, original_node):
        share0_left_child = None
        share0_right_child = None
        share1_left_child = None
        share1_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        if original_node.is_leaf_node():
            labels0=random.randint(0,self.prime-1)
            #Split leaf nodes into shares
            return [Node(attribute=original_node.attribute(), 
                        threshold=labels0, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=(original_node.threshold() - labels0 ) % self.prime, is_leaf_node=True)]
        else:
            #print(len(self.attrlist))
            attribute_index_share0=np.zeros((len(self.attrlist),),dtype=int)
            attribute_index_share1=np.zeros((len(self.attrlist),),dtype=int)
            for i in range(len(self.attrlist)):
                if i != original_node.attribute():
                    attribute_index_share0[i] = random.randint(0,self.prime-1)
                    attribute_index_share1[i] = (0-attribute_index_share0[i])%self.prime
                else:
                    attribute_index_share0[i] = random.randint(0,self.prime-1)
                    attribute_index_share1[i] = (1-attribute_index_share0[i])%self.prime
                    
            thresholdShare0 = random.randint(0,self.prime-1)
            thresholdShare1 = (original_node.threshold()*10 - thresholdShare0 ) % self.prime
            #generate dot triples
            #X0 = np.array([random.randint(0, 10) for _ in range(len(self.attrlist))],dtype=int)
            #X1 = np.array([random.randint(0, 10) for _ in range(len(self.attrlist))],dtype=int)
            #Y0 = np.array([random.randint(0, 10) for _ in range(len(self.attrlist))],dtype=int)
            #Y1 = np.array([random.randint(0, 10) for _ in range(len(self.attrlist))],dtype=int)
            X0 = np.random.randint(10, size=(len(self.attrlist),),dtype=int)
            X1 = np.random.randint(10, size=(len(self.attrlist),),dtype=int)
            Y0 = np.random.randint(10, size=(len(self.attrlist),),dtype=int)
            Y1 = np.random.randint(10, size=(len(self.attrlist),),dtype=int)
            T = random.randint(0, 10)

            Z0 = ((X0.dot(Y1)) %self.prime + T)%self.prime
            Z1 = ((X1.dot(Y0)) %self.prime - T)%self.prime

            share0_left_child, share1_left_child = self._build_shares_v2(original_node.left_child())
            share0_right_child, share1_right_child = self._build_shares_v2(original_node.right_child())

            return [Node(attribute=[attribute_index_share0,X0,Y0,Z0], 
                        threshold=thresholdShare0, 
                        left_child=share0_left_child, 
                        right_child=share0_right_child, 
                        is_leaf_node=False),
                    Node(attribute=[attribute_index_share1,X1,Y1,Z1], 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False)]
        
    def _build_shares_v3(self, original_node):
        share0_left_child = None
        share0_right_child = None
        share1_left_child = None
        share1_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        if original_node.is_leaf_node():
            labels0=random.randint(0,self.prime-1)
            #Split leaf nodes into shares
            return [Node(attribute=original_node.attribute(), 
                        threshold=labels0, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=(original_node.threshold() - labels0 ) % self.prime, is_leaf_node=True)]
        else:
            thresholdShare0 = random.randint(0,self.prime-1)
            thresholdShare1 = (original_node.threshold()*10 - thresholdShare0 ) % self.prime

            share0_left_child, share1_left_child = self._build_shares_v3(original_node.left_child())
            share0_right_child, share1_right_child = self._build_shares_v3(original_node.right_child())
            self.internal_node_num+=1
            self.internal_node_attribute.append(original_node.attribute())
            return [Node(attribute=self.internal_node_num-1, #i-th internal node
                        threshold=thresholdShare0, 
                        left_child=share0_left_child, 
                        right_child=share0_right_child, 
                        is_leaf_node=False),
                    Node(attribute=self.internal_node_num-1, 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False)]

    def _build_shares_v4(self, original_node):
        share0_left_child = None
        share0_right_child = None
        share1_left_child = None
        share1_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        if original_node.is_leaf_node():
            labels0=random.randint(0,self.prime-1)
            #Split leaf nodes into shares
            return [Node(attribute=original_node.attribute(), 
                        threshold=labels0, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=(original_node.threshold() - labels0 ) % self.prime, is_leaf_node=True)]
        else:
            thresholdShare0 = random.randint(0,self.prime-1)
            thresholdShare1 = (original_node.threshold()*10 - thresholdShare0 ) % self.prime

            share0_left_child, share1_left_child = self._build_shares_v4(original_node.left_child())
            share0_right_child, share1_right_child = self._build_shares_v4(original_node.right_child())
            self.internal_node_num+=1
            self.internal_node_attribute.append(original_node.attribute())
            return [Node(attribute=self.internal_node_num-1, #i-th internal node
                        threshold=thresholdShare0, 
                        left_child=share0_left_child, 
                        right_child=share0_right_child, 
                        is_leaf_node=False),
                    Node(attribute=self.internal_node_num-1, 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False)]
    
    def gen_permute_matrix_zd(self, csp0, csp1):
        A=np.zeros((self.internal_node_num,len(self.attrlist))) 


        for i in range(self.internal_node_num):
            A[i][self.internal_node_attribute[i]]=1
        A0= np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        A1 = (A-A0)%self.prime
        csp0.A0=A0
        csp1.A1=A1

        G1 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        g2 = np.random.randint(0,self.prime-1,(len(self.attrlist)))
        g3 = np.dot(G1,g2)
        G1_0 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist))) 
        G1_1 = (G1-G1_0)%self.prime
        g2_0 = np.random.randint(0,self.prime-1,(len(self.attrlist)))
        g2_1 = (g2-g2_0)%self.prime
        g3_0 = np.random.randint(0,self.prime-1,(self.internal_node_num))
        g3_1 = (g3-g3_0)%self.prime

        csp0.MVM_triples=(G1_0,g2_0,g3_0)
        csp1.MVM_triples=(G1_1,g2_1,g3_1)
        return A0, A1


    def gen_permute_matrix(self, csp0, csp1):
        A=np.zeros((self.internal_node_num,len(self.attrlist)), dtype=np.int64) ##[Socket Communication modified]

        for i in range(self.internal_node_num):
            A[i][self.internal_node_attribute[i]]=1
        A0= np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        A1 = (A-A0)%self.prime
        #csp0.A0=A0                           ##[Socket Communication modified]
        #csp1.A1=A1                           ##[Socket Communication modified]
        # print(A0.shape)
        # print(A0)
        # print(A1.shape)
        # print(A1)
        self.csp0_server.send(int_array_to_bytes(A0.shape))     ##[Socket Communication modified]
        self.csp0_server.send(A0.tobytes())                     ##[Socket Communication modified]
        self.csp1_server.send(int_array_to_bytes(A1.shape))     ##[Socket Communication modified]
        self.csp1_server.send(A1.tobytes())                     ##[Socket Communication modified]

        X0 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        X1 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        Y0 = np.random.randint(0,self.prime-1,(len(self.attrlist),))
        Y1 = np.random.randint(0,self.prime-1,(len(self.attrlist),))
        T = np.random.randint(0,self.prime-1,(self.internal_node_num,))
        Z0 = (np.dot(X0,Y1) +T)%self.prime
        Z1 = (np.dot(X1,Y0) - T)%self.prime
        #csp0.MVM_triples=(X0,Y0,Z0) ##[Socket Communication modified]
        #csp1.MVM_triples=(X1,Y1,Z1) ##[Socket Communication modified]
        # print("X0 shape: ", X0.shape)
        # print("X0 ", X0)
        # print("Y0 shape: ", Y0.shape)
        # print("Y0 ", Y0)
        # print("Z0 shape: ", Z0.shape)
        # print("Z0 ", Z0)
        self.csp0_server.send(int_array_to_bytes(X0.shape)) ##[Socket Communication modified]
        self.csp0_server.send(X0.tobytes())                 ##[Socket Communication modified]
        self.csp0_server.send(Y0.tobytes())                 ##[Socket Communication modified]
        self.csp0_server.send(Z0.tobytes())                 ##[Socket Communication modified]
        self.csp1_server.send(int_array_to_bytes(X1.shape)) ##[Socket Communication modified]
        self.csp1_server.send(X1.tobytes())                 ##[Socket Communication modified]
        self.csp1_server.send(Y1.tobytes())                 ##[Socket Communication modified]
        self.csp1_server.send(Z1.tobytes())                 ##[Socket Communication modified]
        return A0, A1


    def gen_pk_sk_matrix(self, csp0, csp1):
        A=np.zeros((len(self.attrlist),self.internal_node_num)) 

        M = np.random.randint(self.prime-1, size=(self.internal_node_num, self.internal_node_num))
        M_inv = np.linalg.inv(M)

        for i in range(self.internal_node_num):
            A[self.internal_node_attribute[i]][i]=1

        AM= np.matmul(A,M)

        #csp0.M_inv=M_inv   ##[Socket Communication modified]
        #csp1.M_inv=M_inv   ##[Socket Communication modified]
        #print("M_inv: \n", M_inv)
        self.csp0_server.send(int_array_to_bytes(M_inv.shape))   ##[Socket Communication modified]
        self.csp0_server.send(M_inv.tobytes())                   ##[Socket Communication modified]
        self.csp1_server.send(int_array_to_bytes(M_inv.shape))   ##[Socket Communication modified]
        self.csp1_server.send(M_inv.tobytes())                   ##[Socket Communication modified]
        
        return AM, M_inv

    def set_shares_to_two_parties(self, csp0, csp1):
        self._root_node_shares=self._root_node_shares 
        #csp0.set_model_share0_root_node(self._root_node_shares[0],self.attrlist,self.prime) ##[Socket Communication modified]
        #csp1.set_model_share1_root_node(self._root_node_shares[1],self.attrlist,self.prime) ##[Socket Communication modified]

        ### send model tree share
        data = pickle.dumps(self._root_node_shares[0])
        self.csp0_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp0_server.send(data)
        data = pickle.dumps(self._root_node_shares[1])
        self.csp1_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp1_server.send(data)

        ### send attribute list
        #print(self.attrlist)
        data = str(self.attrlist)[2:-2].encode()
        self.csp0_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp0_server.send(data)
        data = str(self.attrlist)[2:-2].encode()
        self.csp1_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp1_server.send(data)

        ### send prime
        self.csp0_server.send(int(self.prime).to_bytes(PRIME_SIZE, 'big'))
        #print("PRIME: ", self.prime)
        self.csp1_server.send(int(self.prime).to_bytes(PRIME_SIZE, 'big'))

    def close_csp_connection(self):
        #self.csp0_server.shutdown()
        self.csp0_server.close()
        #self.csp1_server.shutdown()
        self.csp1_server.close()

class CloudServiceProvider0():

    def __init__(self, config, version=1):
        self.version=version
        self.model_share0_root = None
        self.prime=None
        self.attri_list = None
        self.A0= None
        self.rshare = 0
        self.M_inv= None
        self.MVM_triples=None

        self.config = config

        self.label_share0=[]

        self.queue_shared_MO_CSU = queue.Queue() ## for only one set of MO+CSU ##[Socket Communication modified]
        self.queue_shared_CSP0_to_CSP1 = queue.Queue() ## for only one set of MO+CSU ##[Socket Communication modified]
        self.queue_shared_CSP1_to_CSP0 = queue.Queue() ## for only one set of MO+CSU ##[Socket Communication modified]
        
        self.cspthread = threading.Thread(  ##[Socket Communication modified]
            target=self.server_threading,
            #args=(client,)
        )
        self.cspthread.start()
    
    # def __del__(self):
    #     print("[CSP0] destructed")

    def server_threading(self): ##[Socket Communication modified]
        self.s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_socket.bind((self.config['DEFAULT']['CSP0_SEVER_IP'], int(self.config['DEFAULT']['CSP0_SEVER_PORT'])))
        self.s_socket.listen(5)
        socket_count = 0

        #print("[CSP0] Listening on ", CSP0_SEVER_IP , CSP0_SEVER_PORT)
        while socket_count < 3:
            client, self.address = self.s_socket.accept()
            #print("[CSP0] Connected by ", self.address)
            
            #while True:
            data = client.recv(1024)
            if(data == b'MO'): ## MO connection thread
                socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_MO,
                    args=(client,)
                ).start()
            elif(data == b'CSU'):
                socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_CSU,
                    args=(client,)
                ).start()
            elif(data == b'CSP'):
                socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_CSP,
                    args=(client,)
                ).start()
            else:
                print("Can not distingusih MO CSU CSP.")
    
    def server_handle_recv_MO(self, client: socket.socket):
        #print("[CSP0] MO")
        if(self.version == 3):
            ### receive A0
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            A0_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] A0_dim", A0_dim)
            data = client.recv(A0_dim[0] * A0_dim[1] * 8, socket.MSG_WAITALL)
            self.A0 = np.frombuffer(data, dtype=(np.int64)).reshape(A0_dim)
            #print("[CSP] A0 szie: ", self.A0.shape)
            #print("[CSP] A0", self.A0)

            ### receive triples X0, Y0, Z0
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            X0_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] triples size: ", X0_dim)
            X0 = np.frombuffer(client.recv(X0_dim[0] * X0_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(X0_dim)
            #print("[CSP] X0: ", X0)
            Y0 = np.frombuffer(client.recv(X0_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64))
            #print("[CSP] Y0: ", Y0)
            Z0 = np.frombuffer(client.recv(X0_dim[0] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64))
            #print("[CSP] Z0: ", Z0)
            self.MVM_triples = (X0, Y0, Z0)
            
            self.queue_shared_MO_CSU.put(self.A0)
            self.queue_shared_MO_CSU.put((X0, Y0, Z0))

        if(self.version == 4):
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            M_inv_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] triples size: ", X0_dim)
            self.M_inv = np.frombuffer(client.recv(M_inv_dim[0] * M_inv_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(M_inv_dim)
            #print("[CSP0] M_inv:\n", self.M_inv)
            
            self.queue_shared_MO_CSU.put(self.M_inv)

        ### receive model tree share, attribute list and prime
        tree_share_pickle_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(tree_share_pickle_size, socket.MSG_WAITALL)
        #print("[CSP] tree share size: ", len(data))
        #self.model_share0_root = pickle.loads(data)
        self.queue_shared_MO_CSU.put(data)
        attri_list_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(attri_list_size, socket.MSG_WAITALL)
        self.attri_list = data.decode().split("' '")
        #print("[CSP] attribute list: ", self.attri_list)
        self.prime = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        #print("[CSP0] PRIME: ", self.prime)

        self.queue_shared_MO_CSU.put(self.attri_list)
        self.queue_shared_MO_CSU.put(self.prime)

    def server_handle_recv_CSU(self, client: socket.socket):
        #print("[CSP0] CSU")
        ### receive CSU query data
        query_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        self.qDataShare0 = np.frombuffer(client.recv(query_size*8, socket.MSG_WAITALL),dtype=(np.int64))
        #print("[CSP0]-CSU qDataShare0: ", self.qDataShare0)
        #print("[CSP0]-CSU prime: ", self.prime)

        if self.version==1:
            self.minus(self.qDataShare0,p0_node)
            return tree_eval
        
        # if self.version==2:
        #     self.minus_v2(csp0, p0_node)
        #     #print("Attribute selection time: ", select_attr)
        #     return tree_eval

        ### start node_eval
        if self.version==3:
            self.A0 = self.queue_shared_MO_CSU.get()
            self.MVM_triples = self.queue_shared_MO_CSU.get()
            data = self.queue_shared_MO_CSU.get()
            self.model_share0_root = pickle.loads(data)
            self.attri_list = self.queue_shared_MO_CSU.get()
            self.prime = self.queue_shared_MO_CSU.get()

            p0_node = self.model_share0_root

            #start_time = time.time()

            #z0, z1= MVM(csp0.A0,csp1.A1, csp0.qDataShare0,csp1.qDataShare1,csp0.MVM_triples, csp1.MVM_triples,self.prime)
            #def     MVM(     x0,     x1,               y0,              y1,    csp0_triples,     csp1_triples,     PRIME)
            p0x = (self.A0 + self.MVM_triples[0])%self.prime
            p0y = (self.qDataShare0 + self.MVM_triples[1])%self.prime
            # p1x = (x1 + csp1_triples[0])%self.prime
            # p1y = (y1 + csp1_triples[1])%self.prime

            self.queue_shared_CSP0_to_CSP1.put((p0x, p0y))
            (p1x, p1y) = self.queue_shared_CSP1_to_CSP0.get()

            #z0=((np.dot(x0,(y0+p1y)))- (np.dot(p1x,csp0_triples[1]))+csp0_triples[2])%self.prime
            z0=((np.dot(self.A0,(self.qDataShare0+p1y)))- (np.dot(p1x,self.MVM_triples[1]))+self.MVM_triples[2])%self.prime
            # z1=((np.dot(x1,(y1+p0y)))- (np.dot(p0x,csp1_triples[1]))+csp1_triples[2])%self.prime
            #print("[CSP0] z0: ", z0)

            ### generate a, b for permutation
            b_set = [x for x in range(-5, 5) if x != 0]
            self.ab_tuple_list = [(random.randint(-10, 10), random.choice(b_set)) for _ in range(self.A0.shape[0])]
            self.ab_tuple_list_idx = 0
            self.queue_shared_CSP0_to_CSP1.put(self.ab_tuple_list)

            #end_time = time.time()
            #execution_time = float(end_time - start_time)*1000
            #print("Attribute selection time: ", execution_time)

            #start_time = time.time()
            self.minus_v4(z0,p0_node)
            #end_time = time.time()
            #execution_time = float(end_time - start_time)*1000
            # self.permute()

            # data = pickle.dumps(self.model_share0_root)
            # client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
            # client.send(data)
            # client.send(len(self.label_share0).to_bytes(PRIME_SIZE, 'big')) 
            # client.send(int_array_to_bytes(self.label_share0)) 

        if self.version==4:
            self.M_inv = self.queue_shared_MO_CSU.get()
            data = self.queue_shared_MO_CSU.get()
            self.model_share0_root = pickle.loads(data)
            self.attri_list = self.queue_shared_MO_CSU.get()
            self.prime = self.queue_shared_MO_CSU.get()

            p0_node = self.model_share0_root

            start_time = time.time()
            QA0 = (np.round(np.matmul(self.qDataShare0, self.M_inv)))%self.prime
            #QA1 = (np.round(np.matmul(csp1.qDataShare1, csp1.M_inv)))%self.prime
            #print(QA0)
            #print(QA1)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            print("Attribute selection time: ", execution_time)
            #QA=(QA0+QA1)%self.prime
            #print(QA)
            
            self.minus_v4(QA0,p0_node)

        self.permute()

        data = pickle.dumps(self.model_share0_root)
        client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        client.send(data)
        client.send(len(self.label_share0).to_bytes(PRIME_SIZE, 'big')) 
        client.send(int_array_to_bytes(self.label_share0)) 
        return tree_eval

    def server_handle_recv_CSP(self, client: socket.socket):
        #print("[CSP0] CSP1")
        if self.version==3:
            ### receive p1x p1y from CSP1
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            p1x_dim = tuple(bytes_to_int_array(data))
            p1x = np.frombuffer(client.recv(p1x_dim[0] * p1x_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(p1x_dim)
            p1y_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
            p1y = np.frombuffer(client.recv(p1y_size*8, socket.MSG_WAITALL),dtype=(np.int64))

            self.queue_shared_CSP1_to_CSP0.put((p1x,p1y))
            (p0x, p0y) = self.queue_shared_CSP0_to_CSP1.get()

            #print("[CSP0] send (p0x, p0y)")
            ### sned (p0x, p0y) to CSP1
            client.send(int_array_to_bytes(p0x.shape)) 
            client.send(p0x.tobytes())
            client.send(len(p0y).to_bytes(PRIME_SIZE, 'big')) 
            client.send(p0y.tobytes())
            #print("[CSP0] send (p0x, p0y) end")

            ab_tuple_list = self.queue_shared_CSP0_to_CSP1.get()
            client.send(tuple_array_to_bytes(ab_tuple_list))

    def dfs_permute(self,share0,):
        if share0.is_leaf_node():
            self.label_share0.append(share0.threshold())
            share0.set_threshold(self.leaf_idx)
            self.leaf_idx+=1
            return
        else:
            # a=random.randint(-10,10)
            # b=random.randint(-5,5)
            # while b==0:
            #     b=random.randint(-5,5)

            # share0.set_threshold((((share0.threshold()*b)%self.prime)+a)%self.prime)
            # if b<0: #permute
            share0.set_threshold((((share0.threshold()*self.ab_tuple_list[self.ab_tuple_list_idx][1])%self.prime)+self.ab_tuple_list[self.ab_tuple_list_idx][0])%self.prime)
            if self.ab_tuple_list[self.ab_tuple_list_idx][1]<0: #permute
                temp=share0.left_child()
                share0.set_left_child(share0.right_child())
                share0.set_right_child(temp)

            self.ab_tuple_list_idx += 1

            self.dfs_permute(share0.left_child())
            self.dfs_permute(share0.right_child())

    def permute(self):
        self.leaf_idx=0
        self.label_share0.clear()
        p0_node=self.model_share0_root
        self.dfs_permute(p0_node)

    def minus(self, qDataShare0, share0):
        global select_attr
        global tree_eval
        if share0.is_leaf_node():
            #print("A leaf node")
            return
        else:
            attribute=share0.attribute()
            #s_time=time.time()
            start_time = time.time()
            share0.set_threshold((qDataShare0[attribute]-share0.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            #print("MSCDT eval a node:", execution_time, "mseconds")
            self.minus(qDataShare0,share0.left_child())
            self.minus(qDataShare0,share0.right_child())

    def minus_v4(self, QA0, share0):
        global select_attr
        global tree_eval
        if share0.is_leaf_node():
            return
        else:
            start_time = time.time()
            #print(share0.attribute())
            share0.set_threshold((QA0[share0.attribute()]-share0.threshold())%self.prime)
            #share1.set_threshold((QA1[share1.attribute()]-share1.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            self.minus_v4(QA0,share0.left_child())
            self.minus_v4(QA0,share0.right_child())
            #print(share0.attribute())
    
    def resultshare0(self):
        return self.rshare
    
    def set_model_share0_root_node(self, share0,attrlist,prime):
        self.attri_list=attrlist
        self.prime=prime
        self.model_share0_root = share0
    
    def set_query_data_share0(self, qDataShare0):
        self.qDataShare0 = qDataShare0
    def set_resultshare0(self, val):
        self.rshare=val

class CloudServiceProvider1():

    def __init__(self, config, version=1):
        self.version=version
        self.model_share1_root = None
        self.attri_list = None
        self.prime = None
        self.rshare = 0
        self.A1= None
        self.M_inv= None
        self.MVM_triples=None

        self.config = config

        self.label_share1=[]

        self.queue_shared_MO_CSU = queue.Queue() ## for only one set of MO+CSU ##[Socket Communication modified]

        self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        self.csp0_server.connect((config['DEFAULT']['CSP0_SEVER_IP'], int(config['DEFAULT']['CSP0_SEVER_PORT'])))           ##[Socket Communication modified]
        self.csp0_server.send('CSP'.encode())                                ##[Socket Communication modified]

        self.cspthread = threading.Thread(                   ##[Socket Communication modified]
            target=self.server_threading,
            #args=(client,)
        )
        self.cspthread.start()
    
    # def __del__(self):
    #     print("[CSP1] destructed")

    def server_threading(self): ##[Socket Communication modified]
        self.s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_socket.bind((self.config['DEFAULT']['CSP1_SEVER_IP'], int(self.config['DEFAULT']['CSP1_SEVER_PORT'])))
        self.s_socket.listen(5)

        self.socket_count = 0

        #print("[CSP1] Listening on ", CSP1_SEVER_IP , CSP1_SEVER_PORT)
        while self.socket_count < 2:
            client, address = self.s_socket.accept()
            #print("[CSP1] Connected by ", address)

            data = client.recv(1024)
            #print(data)
            if(data == b'MO'):
                self.socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_MO,
                    args=(client,)
                ).start()
            elif(data == b'CSU'):
                self.socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_CSU,
                    args=(client,)
                ).start()

    def server_handle_recv_MO(self, client:socket.socket):
        if(self.version == 3):
            data = client.recv(6, socket.MSG_WAITALL)
            #print(data)
            A1_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] A1_dim",A1_dim)
            data = client.recv(A1_dim[0]*A1_dim[1]*8, socket.MSG_WAITALL)
            self.A1 = np.frombuffer(data, dtype=(np.int64)).reshape(A1_dim)
            #print("[CSP] A1 szie: ", self.A1.shape)
            #print("[CSP] A1",self.A1)
            data = client.recv(6, socket.MSG_WAITALL)
            X1_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] triples size: ", X1_dim)
            X1 = np.frombuffer(client.recv(X1_dim[0]*X1_dim[1]*8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(X1_dim)
            #print("[CSP] X1: ", X1)
            Y1 = np.frombuffer(client.recv(X1_dim[1]*8, socket.MSG_WAITALL),
                            dtype=(np.int64))
            #print("[CSP] Y1: ", Y1)
            Z1 = np.frombuffer(client.recv(X1_dim[0]*8, socket.MSG_WAITALL),
                            dtype=(np.int64))
            #print("[CSP] Z1: ", Z1)
            self.MVM_triples = (X1, Y1, Z1)
            self.queue_shared_MO_CSU.put(self.A1)
            self.queue_shared_MO_CSU.put((X1, Y1, Z1))

        if(self.version == 4):
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            M_inv_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] triples size: ", X0_dim)
            self.M_inv = np.frombuffer(client.recv(M_inv_dim[0] * M_inv_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(M_inv_dim)
            #print("[CSP1] M_inv:\n", self.M_inv)
            self.queue_shared_MO_CSU.put(self.M_inv)

        ### receive MO model tree share, attribute list and prime
        tree_share_pickle_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(tree_share_pickle_size, socket.MSG_WAITALL)
        #print("[CSP] tree share size: ", len(data))
        #self.model_share1_root = pickle.loads(data)
        self.queue_shared_MO_CSU.put(data)
        attri_list_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(attri_list_size, socket.MSG_WAITALL)
        self.attri_list = data.decode().split("' '")
        #print("[CSP] attribute list: ", self.attri_list)
        self.prime = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')

        self.queue_shared_MO_CSU.put(self.attri_list)
        self.queue_shared_MO_CSU.put(self.prime)

    def server_handle_recv_CSU(self, client:socket.socket):
        #print("[CSP1] CSU")
        ### receive CSU query data
        query_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        self.qDataShare1 = np.frombuffer(client.recv(query_size*8, socket.MSG_WAITALL),dtype=(np.int64))
        # print("[CSP1]-CSU qDataShare1: ", self.qDataShare1)
        # print("[CSP1]-CSU prime: ", self.prime)

        if self.version==1:
            self.minus(self.qDataShare1,p1_node)
            return tree_eval
        
        # if self.version==2:
        #     self.minus_v2(csp1, p1_node)
        #     #print("Attribute selection time: ", select_attr)
        #     return tree_eval

        ### start node_eval
        if self.version==3:
            self.A1 = self.queue_shared_MO_CSU.get()
            self.MVM_triples = self.queue_shared_MO_CSU.get()
            data = self.queue_shared_MO_CSU.get()
            self.model_share1_root = pickle.loads(data)
            self.attri_list = self.queue_shared_MO_CSU.get()
            self.prime = self.queue_shared_MO_CSU.get()

            p1_node = self.model_share1_root

            #start_time = time.time()

            #z0, z1= MVM(csp0.A0,csp1.A1, csp0.qDataShare0,csp1.qDataShare1,csp0.MVM_triples, csp1.MVM_triples,self.prime)
            #def     MVM(x0,     x1,      y0,              y1,              csp0_triples,csp1_triples,PRIME)
            #p0x = (self.A0 + self.MVM_triples[0])%self.prime
            #p0y = (self.qDataShare0 + self.MVM_triples[1])%self.prime
            p1x = (self.A1 + self.MVM_triples[0])%self.prime
            p1y = (self.qDataShare1 + self.MVM_triples[1])%self.prime

            #print("p1x: ", p1x, type(p1x))
            #print("p1y: ", p1y, type(p1y))

            ### CSP1 send p1x p1y to CSP0
            self.csp0_server.send(int_array_to_bytes(p1x.shape)) 
            self.csp0_server.send(p1x.tobytes())
            self.csp0_server.send(len(p1y).to_bytes(PRIME_SIZE, 'big')) 
            self.csp0_server.send(p1y.tobytes())

            ### receive p0x p0y from CSP0
            #print("[CSP1] receive (p0x, p0y)")
            data = self.csp0_server.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            p0x_dim = tuple(bytes_to_int_array(data))
            p0x = np.frombuffer(self.csp0_server.recv(p0x_dim[0] * p0x_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(p0x_dim)
            p0y_size = int.from_bytes(self.csp0_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
            p0y = np.frombuffer(self.csp0_server.recv(p0y_size*8, socket.MSG_WAITALL),dtype=(np.int64))
            #print("[CSP1] receive (p0x, p0y) end")

            z1=((np.dot(self.A1,(self.qDataShare1+p0y)))- (np.dot(p0x,self.MVM_triples[1]))+self.MVM_triples[2])%self.prime
            #print("[CSP1] z1: ", z1)

            #end_time = time.time()
            #execution_time = float(end_time - start_time)*1000
            #print("Attribute selection time: ", execution_time)

            #start_time = time.time()
            self.minus_v4(z1,p1_node)
            #end_time = time.time()
            #execution_time = float(end_time - start_time)*1000

            data = self.csp0_server.recv(self.A1.shape[0]*2, socket.MSG_WAITALL)
            self.ab_tuple_list = bytes_to_tuple_array(data, tuple_size=2, byte_size=1)
            self.ab_tuple_list_idx = 0

            # self.permute()

            # data = pickle.dumps(self.model_share1_root)
            # client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
            # client.send(data)
            # client.send(self.leaf_idx.to_bytes(PRIME_SIZE, 'big'))
            # client.send(len(self.label_share1).to_bytes(PRIME_SIZE, 'big')) 
            # client.send(int_array_to_bytes(self.label_share1)) 

            # return tree_eval
        if self.version==4:
            self.M_inv = self.queue_shared_MO_CSU.get()
            data = self.queue_shared_MO_CSU.get()
            self.model_share1_root = pickle.loads(data)
            self.attri_list = self.queue_shared_MO_CSU.get()
            self.prime = self.queue_shared_MO_CSU.get()

            start_time = time.time()
            #QA0 = (np.round(np.matmul(csp0.qDataShare0, csp0.M_inv)))%self.prime
            QA1 = (np.round(np.matmul(self.qDataShare1, self.M_inv)))%self.prime
            #print(QA0)
            #print(QA1)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            print("Attribute selection time: ", execution_time)
            #QA=(QA0+QA1)%self.prime
            #print(QA)
            
            self.minus_v4(QA1,p1_node)
            
            #return tree_eval
        self.permute()

        data = pickle.dumps(self.model_share1_root)
        client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        client.send(data)
        client.send(self.leaf_idx.to_bytes(PRIME_SIZE, 'big'))
        client.send(len(self.label_share1).to_bytes(PRIME_SIZE, 'big')) 
        client.send(int_array_to_bytes(self.label_share1)) 

        return tree_eval

    def dfs_permute(self,share1):
        if share1.is_leaf_node():
            self.label_share1.append(share1.threshold())
            share1.set_threshold(0)
            self.leaf_idx+=1
            return
        else:
            a=random.randint(-10,10)
            b=random.randint(-5,5)
            while b==0:
                b=random.randint(-5,5)
            
            # share1.set_threshold((share1.threshold()*b)%self.prime)
            # if b<0: #permute
            share1.set_threshold((share1.threshold()*self.ab_tuple_list[self.ab_tuple_list_idx][1])%self.prime)
            if self.ab_tuple_list[self.ab_tuple_list_idx][1]<0: #permute
                temp=share1.left_child()
                share1.set_left_child(share1.right_child())
                share1.set_right_child(temp)

            self.ab_tuple_list_idx += 1

            self.dfs_permute(share1.left_child())
            self.dfs_permute(share1.right_child())

    def permute(self):
        self.leaf_idx=0
        #self.label_share0.clear()
        self.label_share1.clear()
        #p0_node=csp0.model_share0_root
        p1_node=self.model_share1_root
        self.dfs_permute(p1_node)

    def minus(self,qDataShare1, share1):
        global select_attr
        global tree_eval
        if share1.is_leaf_node():
            #print("A leaf node")
            return
        else:
            attribute=share1.attribute()
            #s_time=time.time()
            start_time = time.time()
            share1.set_threshold((qDataShare1[attribute]-share1.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            #print("MSCDT eval a node:", execution_time, "mseconds")
            self.minus(qDataShare1,share1.left_child())
            self.minus(qDataShare1,share1.right_child())

    # def minus_v2(self, csp1, share1):
    #     global select_attr
    #     global tree_eval
    #     if share1.is_leaf_node():
    #         return
    #     else:
    #         #attribute0=share0.attribute() #attribute=[attribute_index_share0 or 1,X0(1),Y0(1),Z0(1)
    #         attribute1=share1.attribute()

    #         start_time = time.time()
    #         p1x = (csp1.qDataShare1 + attribute1[1]) %self.prime
    #         p1y = (attribute1[0] + attribute1[2]) %self.prime
        
    #         z1 = (csp1.qDataShare1.dot((attribute1[0] + p0y)) - attribute1[2].dot(p0x) + attribute1[3])%self.prime
    #         end_time = time.time()
    #         execution_time = float(end_time - start_time)*1000
    #         select_attr+=execution_time

    #         #print((csp0.qDataShare0+csp1.qDataShare1)%self.prime)
    #         #print((attribute0[0]+attribute1[0])%self.prime)
    #         #a=(attribute0[0]+attribute1[0])%self.prime
    #         #b=(csp0.qDataShare0+csp1.qDataShare1)%self.prime
    #         #print(a.dot(b))
    #         #print("z0+z1:",(z0+z1)%self.prime)
    #         #print("origin th: ",(share0.threshold()+share1.threshold())%self.prime)
    #         start_time = time.time()
    #         share1.set_threshold((z1-share1.threshold())%self.prime)
    #         end_time = time.time()
    #         execution_time = float(end_time - start_time)*1000
    #         tree_eval+=execution_time
    #         secret=(share0.threshold()+share1.threshold())%self.prime
    #         if secret > self.prime // 2:
    #             secret -= self.prime
    #         #print("after:",secret)
    #         self.minus_v2(csp1, share1.left_child())
    #         self.minus_v2(csp1, share1.right_child())

    def minus_v4(self, QA1, share1):
        global select_attr
        global tree_eval
        if share1.is_leaf_node():
            return
        else:
            start_time = time.time()
            #print(share0.attribute())
            share1.set_threshold((QA1[share1.attribute()]-share1.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            self.minus_v4(QA1,share1.left_child())
            self.minus_v4(QA1,share1.right_child())
            #print(share0.attribute())

    def resultshare1(self):
        return self.rshare

    def set_model_share1_root_node(self, share1,attrlist,prime):
        self.attri_list=attrlist
        self.prime=prime
        self.model_share1_root = share1
    
    def set_query_data_share1(self, qDataShare1):
        self.qDataShare1 = qDataShare1

    def set_resultshare1(self, val):
        self.rshare=val


class CloudServiceUser():
    
    def __init__(self, config, version=1):
        self.version=version
        self.qData = None
        self.prime = None

        self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        self.csp0_server.connect((config['DEFAULT']['CSP0_SEVER_IP'], int(config['DEFAULT']['CSP0_SEVER_PORT'])))           ##[Socket Communication modified]
        self.csp0_server.send('CSU'.encode())                                ##[Socket Communication modified]
        self.csp1_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        self.csp1_server.connect((config['DEFAULT']['CSP1_SEVER_IP'], int(config['DEFAULT']['CSP1_SEVER_PORT'])))           ##[Socket Communication modified]
        self.csp1_server.send('CSU'.encode())                                ##[Socket Communication modified]

        self.z_reconstruct_queue = queue.Queue()

    def set_seed(self, seed):
        self.seed = seed

    def set_query_data_sd(self, data,prime):
        self.prime=prime
        self.qData = data*100
        
        self.qDataShare0 = np.random.randint(2**31, size=(len(data),))
        self.qDataShare1 = (self.qData-self.qDataShare0)%self.prime    

    def set_query_data_zd(self, data,prime):
        self.prime=prime
        self.qData = data*100
        
        self.qDataShare0 = np.random.randint(self.prime-1, size=(len(data),))
        self.qDataShare1 = (self.qData-self.qDataShare0)%self.prime    

    def set_query_data(self, data,prime):
        self.prime=prime
        self.qData = data*100
        
        self.qDataShare0 = np.random.randint(self.prime-1, size=(len(data),))
        self.qDataShare1 = (self.qData-self.qDataShare0)%self.prime
    
    
    def set_query_data_v2(self, data,prime):
        self.prime=prime
        self.qData = data*100
        self.qDataShare0=np.zeros((len(data),),dtype=int)
        self.qDataShare1=np.zeros((len(data),),dtype=int)
        
        self.qDataShare0 = np.random.randint(self.prime-1, size=(len(data),))
        self.qDataShare1=(self.qData-self.qDataShare0)%self.prime
        #for i in range(len(data)):
        #    self.qDataShare0[i] = random.randint(0,self.prime-1)
        #    self.qDataShare1[i] = (self.qData[i]-self.qDataShare0[i]) % self.prime
        
    def set_query_data_v3(self, data,prime):
        self.prime=prime
        self.qData = data*100
        self.qDataShare0=np.zeros((len(data),),dtype=int)
        self.qDataShare1=np.zeros((len(data),),dtype=int)
        
        self.qDataShare0 = np.random.randint(self.prime-1, size=(len(data),))
        self.qDataShare1=(self.qData-self.qDataShare0)%self.prime
        #for i in range(len(data)):
        #    self.qDataShare0[i] = random.randint(0,self.prime-1)
        #    self.qDataShare1[i] = (self.qData[i]-self.qDataShare0[i]) % self.prime

    def set_query_data_v4(self, data, AM, prime):
        self.prime=prime
        self.qData = data*100
        #print(self.qData)
        QAM = np.matmul(self.qData, AM)
        #print(QAM)

        self.qDataShare0 = np.random.randint(self.prime-1, size=(len(QAM),))
        #self.qDataShare0 = sy.Matrix(self.qDataShare0).transpose()
        self.qDataShare1 = (QAM - self.qDataShare0)
        #self.qDataShare1 = (QAM - self.qDataShare0)%self.prime

    def send_query_data_to_csp(self, csp0,csp1):
        csp0.set_query_data_share0(self.qDataShare0)  ##[Socket Communication modified]
        csp1.set_query_data_share1(self.qDataShare1)  ##[Socket Communication modified]
        self.csp0_server.send(len(self.qDataShare0).to_bytes(PRIME_SIZE, 'big')) ##[Socket Communication modified]
        self.csp0_server.send(self.qDataShare0.tobytes())                        ##[Socket Communication modified]
        #print("[CSU] qDataShare0: ", self.qDataShare0)
        self.csp1_server.send(len(self.qDataShare1).to_bytes(PRIME_SIZE, 'big')) ##[Socket Communication modified]
        self.csp1_server.send(self.qDataShare1.tobytes())                        ##[Socket Communication modified]
        #print("[CSU] qDataShare1: ", self.qDataShare1)

    def receive_result_items_and_compute_shared_result(self,):
        tree_share_pickle_size = int.from_bytes(self.csp0_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big') #model_share0_root
        data = self.csp0_server.recv(tree_share_pickle_size, socket.MSG_WAITALL)          #model_share0_root
        model_share0_root = pickle.loads(data)                                            #model_share0_root
        label_share0_size = int.from_bytes(self.csp0_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')   #label_share0
        data = self.csp0_server.recv(label_share0_size*PRIME_SIZE, socket.MSG_WAITALL)                     #label_share0
        label_share0 = bytes_to_int_array(data)                                        #label_share0

        tree_share_pickle_size = int.from_bytes(self.csp1_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big') #model_share1_root
        data = self.csp1_server.recv(tree_share_pickle_size, socket.MSG_WAITALL)          #model_share1_root
        model_share1_root = pickle.loads(data)                                            #model_share1_root
        leaf_num = int.from_bytes(self.csp1_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')       #leaf_num
        #print("[CSU] leaf_num: ", leaf_num)
        label_share1_size = int.from_bytes(self.csp1_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')  #label_share1
        data = self.csp1_server.recv(label_share1_size*PRIME_SIZE, socket.MSG_WAITALL)                    #label_share1
        label_share1 = bytes_to_int_array(data)                                       #label_share1
        

        index = self.find_fake_index(model_share0_root, model_share1_root)
        #print("[CSU] index: ", index)

        fake_leaf_idx=np.zeros((leaf_num,),dtype=int)
        fake_leaf_idx[index]=1
        fake_leaf_idx_share0=np.random.randint(0,PRIME-1,(leaf_num,))
        fake_leaf_idx_share1=np.zeros((leaf_num,),dtype=int)
        fake_leaf_idx_share1 = (fake_leaf_idx-fake_leaf_idx_share0)%PRIME

        z0, z1 = self.SDP(fake_leaf_idx_share0,fake_leaf_idx_share1, label_share0, label_share1)
        z = reconstruct_secret((z0,z1), self.prime)

        self.z_reconstruct_queue.put(z)
        #print("[CSU Socket] Reconstructed label: ", z)

    def get_reconstruct_result(self):
        return self.z_reconstruct_queue.get()

    def find_fake_index(self,p0_node,p1_node):
        # p0_node=csp0.model_share0_root
        # p1_node=csp1.model_share1_root
        while p0_node.is_leaf_node()!=True:
            recon=reconstruct_secret([p0_node.threshold(),p1_node.threshold()],self.prime)#reconstruct current node secret 
            #print(recon)
            if recon<=0: #go left
                p0_node=p0_node.left_child()
                p1_node=p1_node.left_child()
            else:
                p0_node=p0_node.right_child()
                p1_node=p1_node.right_child()
        return p0_node.threshold()+p1_node.threshold()
    
    def SDP(self,fake_leaf_idx_share0,fake_leaf_idx_share1, label_share0, label_share1):
        #for i in range(self.leaf_idx):
        #    print(reconstruct_secret([self.label_share0[i],self.label_share1[i]],self.prime))
        x0 = fake_leaf_idx_share0
        x1 = fake_leaf_idx_share1
        y0 = label_share0
        y1 = label_share1
        z0, z1 = dot_product_triples(len(x0), x0, x1, y0, y1,self.prime)
        return z0,z1
    
    def close_csp_connection(self):
        #self.csp0_server.shutdown()
        self.csp0_server.close()
        #self.csp1_server.shutdown()
        self.csp1_server.close()

class Protocol():
    
    def __init__(self, version=1):
        self.prime = None
        self.attrilist=None
        self.label_share0=[]
        self.label_share1=[]
        self.leaf_idx=0
        self.version=version

    def minus_zd(self, z0,z1,share0, share1):
        
        t1_0= 0

        t1_1= 1

        t2_0= 1

        t2_1= 1

        t3_0= 0

        t3_1= 0
        if share0.is_leaf_node() and share1.is_leaf_node():
            #print("A leaf node")
            return
        else:
            attribute=share0.attribute()
            #print(attribute)
            #s_time=time.time()
            share0.set_threshold((z0[attribute]-share0.threshold())%self.prime)
            share1.set_threshold((z1[attribute]-share1.threshold())%self.prime)

            # secret =(share0.threshold()+share1.threshold())%self.prime
            # if secret > self.prime // 2:
            #    secret -= self.prime
            # print()
            # print(secret)

            
            p=int(share0.threshold())
            # print(a)
            # print(bin(a))
            
            q=int(share1.threshold())
            # print(b)
            # print(bin(b))
            # secret =(p+q)%self.prime
            # if secret > self.prime // 2:
            #     secret -= self.prime
            # print()
            # print(secret)
            

            bit_length = 32
            p_0= [(p >> i) & 1 for i in range(bit_length)]
            #p_1= [(0 >> i) & 1 for i in range(bit_length)]
            #q_0= [(0 >> i) & 1 for i in range(bit_length)]
            q_1= [(q >> i) & 1 for i in range(bit_length)]
            w_0= p_0
            w_1= q_1
            
            
            #c1
            '''
            t1=np.random.randint(0,2)
            t2=np.random.randint(0,2)
            t3=t1*t2
            
            t1_0= np.random.randint(0,2)
            print(t1_0)
            t1_1= (t1-t1_0)%2
            print(t1_1)
            t2_0= np.random.randint(0,2)
            print(t2_0)
            t2_1= (t2-t2_0)%2
            print(t2_1)
            t3_0= np.random.randint(0,2)
            print(t3_0)
            t3_1= (t3-t3_0)%2
            print(t3_1)
            '''
            carry= additive_mul(p_0[0],0,0,q_1[0],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)


            for i in range(1, bit_length-1):
                '''
                t1=np.random.randint(0,2)
                t2=np.random.randint(0,2)
                t3=t1*t2

                t1_0= np.random.randint(0,2)
                t1_1= (t1-t1_0)%2
                t2_0= np.random.randint(0,2)
                t2_1= (t2-t2_0)%2
                t3_0= np.random.randint(0,2)
                t3_1= (t3-t3_0)%2
                '''
                d= additive_mul(p_0[i],0,0,q_1[i],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                d[0]= (d[0]+1)%2
                #d[1] = (d[0])%2

                e= additive_mul(w_0[i],w_1[i],carry[0],carry[1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                e[0]= (e[0]+1)%2
                #e[1] = (e[0])%2

                carry= additive_mul(e[0],e[1],d[0],d[1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                carry[0]= (carry[0]+1)%2
                #carry[1] = (carry[0])%2

            MSB_0= (w_0[19]+carry[0])%2
            MSB_1= (w_1[19]+carry[1])%2     

            share0.set_threshold(MSB_0)
            share1.set_threshold(MSB_1)
            #print("MSB: ",(share0.threshold()+share1.threshold())%2)
            

            #Change domain
            

            a_0= MSB_0
            a_1= 0
            b_0= 0
            b_1= MSB_1

            t1_0p=9889
            t1_1p=53080
            t2_0p=19644
            t2_1p=44576
            t3_0p=5948
            t3_1p=554176
            '''
            t1=np.random.randint(0,2**16)
            t2=np.random.randint(0,2**16)
            t3=t1*t2

            t1_0= np.random.randint(0,2**16)
            print(t1_0)
            t1_1= (t1-t1_0)%self.prime
            print(t1_1)
            t2_0= np.random.randint(0,2**16)
            print(t2_0)
            t2_1= (t2-t2_0)%self.prime
            print(t2_1)
            t3_0= np.random.randint(0,2**16)
            print(t3_0)
            t3_1= (t3-t3_0)%self.prime
            print(t3_1)
            '''
            temp= additive_mul_prime(a_0,a_1,b_0,b_1,t1_0p,t1_1p,t2_0p,t2_1p,t3_0p,t3_1p,self.prime)

            vj_0=(a_0+b_0-2*temp[0])%self.prime
            vj_1=(a_1+b_1-2*temp[1])%self.prime
            share0.set_threshold([(1-vj_0)%self.prime, vj_0])
            share1.set_threshold([(-vj_1)%self.prime, vj_1])

            #print("comparison result: ",(share0.threshold()[0]+share1.threshold()[0])%self.prime)
            #print("MSB: ",(share0.threshold()[0]+share0.threshold()[1])%self.prime)
            #e_time=time.time()
            #execution_time = float(e_time - s_time)*1000
            #print("MSCDT eval a node:", execution_time, "mseconds")
            self.minus_zd(z0,z1,share0.left_child(),share1.left_child())
            self.minus_zd(z0,z1,share0.right_child(),share1.right_child())

        
    def minus_sd(self, csp0,csp1,share0, share1):
        global select_attr
        global tree_eval
        t1_0p=9889
        t1_1p=53080
        t2_0p=19644
        t2_1p=44576
        t3_0p=5948
        t3_1p=554176
        t1_0= 0

        t1_1= 1

        t2_0= 1

        t2_1= 1

        t3_0= 0

        t3_1= 0
        if share0.is_leaf_node() and share1.is_leaf_node():
            #print("A leaf node")
            return
        else:
            p_0= []
            p_1= []
            start_time = time.time()
            s_0=random.randint(0,2**16)
            s_1=random.randint(0,2**16)
            r_0=random.randint(0,2**16)
            r_1=random.randint(0,2**16)

            #s_0=0
            #s_1=1
            #r_0=0
            #r_1=0

            for i in range(len(csp0.qDataShare0)):
                p_0.append((csp0.qDataShare0[((i+s_0)%self.prime)%len(csp0.qDataShare0)]+r_0)%self.prime)
            # print(csp0.qDataShare0)
            # print(p_0)
            # print()
            for i in range(len(csp1.qDataShare1)):
                p_1.append((csp1.qDataShare1[((i+s_1)%self.prime)%len(csp1.qDataShare1)]+r_1)%self.prime)
            # print(csp1.qDataShare1)
            # print(p_1)
            
            #C0
            r=random.randint(0,2**16)
            i_0= (share0.attribute()+r)%self.prime
            
            i_p= (i_0+share1.attribute()-s_1)%self.prime

            i_p_1= ((i_p-r)%self.prime)%len(csp0.qDataShare0)
            #print(int(i_p_1))

            alphabet = "abcdefghijklmnopqrstuvwxyz"
            #secrets = [bytes("".join(random.choice(alphabet) for _ in range(secret_length)), "ASCII") for __ in range(8)]
            secrets=[]
            for i in range(len(csp1.qDataShare1)):
                secrets.append(str(p_1[i]).zfill(13).encode("utf-8"))
            #secrets = [b'Secret message 1', b'Secret message 2', b'Secret message 3', b'Secret message 4']
            secret_length = len(secrets[0])
            t = 1
            alice = Alice(secrets, t, secret_length)
            bob = Bob([int(i_p_1)])

            alice.setup()
            bob.setup()
            alice.transmit()
            M_prime = bob.receive()
            p_p_1=int(M_prime[0])
            #print(p_p_1)

            #C1
            r=random.randint(0,2**16)
            i_1= (share1.attribute()+r)%self.prime
            
            i_p= (i_1+share0.attribute()-s_0)%self.prime

            i_p_0= ((i_p-r)%self.prime)%len(csp1.qDataShare1)
            # print(csp0.qDataShare0)
            # print(int(i_p_0))

            alphabet = "abcdefghijklmnopqrstuvwxyz"
            #secrets = [bytes("".join(random.choice(alphabet) for _ in range(secret_length)), "ASCII") for __ in range(8)]
            secrets=[]
            for i in range(len(csp0.qDataShare0)):
                secrets.append(str(p_0[i]).zfill(13).encode("utf-8"))
            #secrets = [b'Secret message 1', b'Secret message 2', b'Secret message 3', b'Secret message 4']
            secret_length = len(secrets[0])
            t = 1
            alice = Alice(secrets, t, secret_length)
            bob = Bob([int(i_p_0)])

            alice.setup()
            bob.setup()
            alice.transmit()
            M_prime = bob.receive()
            p_p_0=int(M_prime[0])
            #print(p_p_0)

            r_p=random.randint(0,2**16)
            feature_share1=r_p

            p_star_0=(p_p_0-r_1-r_p)%self.prime
            
            feature_share0=p_star_0+p_p_1-r_0
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            select_attr+=execution_time
            #print((feature_share0+feature_share1)%self.prime)
            start_time = time.time()
            attribute=share0.attribute()
            share0.set_threshold((feature_share0-share0.threshold())%self.prime)
            share1.set_threshold((feature_share1-share1.threshold())%self.prime)
            #secret =(share0.threshold()+share1.threshold())%self.prime
            #if secret > self.prime // 2:
            #    secret -= self.prime
            #print()
            #print(secret)

            
            a=int(share0.threshold())
            # print(a)
            # print(bin(a))
            
            b=int(share1.threshold())
            # print(b)
            # print(bin(b))
            
            
            bit_length = 165
            a_0= [(a >> i) & 1 for i in range(bit_length)]
            #a_1= [0]*32
            #b_0= [0]*32
            b_1= [(b >> i) & 1 for i in range(bit_length)]
            w_0= a_0
            w_1= b_1
            
            G_0= []
            G_1= []
            P_0= [(0 >> i) & 1 for i in range(bit_length)]
            P_1= [(0 >> i) & 1 for i in range(bit_length)]

            for i in range(bit_length):
                '''
                t1=np.random.randint(0,2)
                t2=np.random.randint(0,2)
                t3=t1*t2

                t1_0= np.random.randint(0,2)
                t1_1= (t1-t1_0)%2
                t2_0= np.random.randint(0,2)
                t2_1= (t2-t2_0)%2
                t3_0= np.random.randint(0,2)
                t3_1= (t3-t3_0)%2
                '''
                r= additive_mul(a_0[i],0,0,b_1[i],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                G_0.append(r[0])
                G_1.append(r[1])

            P_0= [(a_0[i]) for i in range(bit_length)]
            P_1= [(b_1[i]) for i in range(bit_length)]
            
            level1_0= []
            level1_1= []
            level2_0= []
            level2_1= []
            level3_0= [] 
            level3_1= [] 
            level4_0= [] 
            level4_1= []
            level1_0.append([G_0[0], P_0[0]])
            level1_1.append([G_1[0], P_1[0]])

            for i in range(1, 16):
                '''
                t1=np.random.randint(0,2)
                t2=np.random.randint(0,2)
                t3=t1*t2

                t1_0= np.random.randint(0,2)
                t1_1= (t1-t1_0)%2
                t2_0= np.random.randint(0,2)
                t2_1= (t2-t2_0)%2
                t3_0= np.random.randint(0,2)
                t3_1= (t3-t3_0)%2
                '''
                temp= additive_mul(G_0[2*i-1],G_1[2*i-1],P_0[2*i],P_1[2*i],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                temp2= additive_mul(P_0[2*i],P_1[2*i],P_0[2*i-1],P_1[2*i-1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)

                level1_0.append( [ (G_0[2*i]+(temp[0]))%2 , temp2[0] ] )
                level1_1.append( [ (G_1[2*i]+(temp[1]))%2 , temp2[1] ] )
            
            for i in range(0, 8):
                '''
                t1=np.random.randint(0,2)
                t2=np.random.randint(0,2)
                t3=t1*t2

                t1_0= np.random.randint(0,2)
                t1_1= (t1-t1_0)%2
                t2_0= np.random.randint(0,2)
                t2_1= (t2-t2_0)%2
                t3_0= np.random.randint(0,2)
                t3_1= (t3-t3_0)%2
                '''
                temp= additive_mul(level1_0[2*i][0],level1_1[2*i][0],level1_0[2*i+1][1],level1_1[2*i+1][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                temp2= additive_mul(level1_0[2*i+1][1],level1_1[2*i+1][1],level1_0[2*i][1],level1_1[2*i][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)

                level2_0.append( [ (level1_0[2*i+1][0]+temp[0])%2 , temp2[0] ] )
                level2_1.append( [ (level1_1[2*i+1][0]+temp[1])%2 , temp2[1] ] )

            for i in range(0, 4):
                '''
                t1=np.random.randint(0,2)
                t2=np.random.randint(0,2)
                t3=t1*t2

                t1_0= np.random.randint(0,2)
                t1_1= (t1-t1_0)%2
                t2_0= np.random.randint(0,2)
                t2_1= (t2-t2_0)%2
                t3_0= np.random.randint(0,2)
                t3_1= (t3-t3_0)%2
                '''
                temp= additive_mul(level2_0[2*i][0],level2_1[2*i][0],level2_0[2*i+1][1],level2_1[2*i+1][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                temp2= additive_mul(level2_0[2*i+1][1],level2_1[2*i+1][1],level2_0[2*i][1],level2_1[2*i][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)

                level3_0.append( [ (level2_0[2*i+1][0]+temp[0])%2 , temp2[0] ] )
                level3_1.append( [ (level2_1[2*i+1][0]+temp[1])%2 , temp2[1] ] )

            for i in range(0, 2):
                '''
                t1=np.random.randint(0,2)
                t2=np.random.randint(0,2)
                t3=t1*t2

                t1_0= np.random.randint(0,2)
                t1_1= (t1-t1_0)%2
                t2_0= np.random.randint(0,2)
                t2_1= (t2-t2_0)%2
                t3_0= np.random.randint(0,2)
                t3_1= (t3-t3_0)%2
                '''
                temp= additive_mul(level3_0[2*i][0],level3_1[2*i][0],level3_0[2*i+1][1],level3_1[2*i+1][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)
                temp2= additive_mul(level3_0[2*i+1][1],level3_1[2*i+1][1],level3_0[2*i][1],level3_1[2*i][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)

                level4_0.append( [ (level3_0[2*i+1][0]+temp[0])%2 , temp2[0] ] )
                level4_1.append( [ (level3_1[2*i+1][0]+temp[1])%2 , temp2[1] ] )

            '''
            t1=np.random.randint(0,2)
            t2=np.random.randint(0,2)
            t3=t1*t2

            t1_0= np.random.randint(0,2)
            t1_1= (t1-t1_0)%2
            t2_0= np.random.randint(0,2)
            t2_1= (t2-t2_0)%2
            t3_0= np.random.randint(0,2)
            t3_1= (t3-t3_0)%2
            '''
            temp= additive_mul(level4_0[0][0],level4_1[0][0],level4_0[1][1],level4_1[1][1],t1_0,t1_1,t2_0,t2_1,t3_0,t3_1)

            finalG_0=(level4_0[1][0]+temp[0])%2
            finalG_1=(level4_1[1][0]+temp[1])%2

            #finalG_0=(level4_0[1][0]+(level4_0[0][0]*level4_0[1][1]))%2
            #finalG_1=(level4_1[1][0]+(level4_1[0][0]*level4_1[1][1]))%2

            MSB_0= (w_0[31]+finalG_0)%2
            MSB_1= (w_1[31]+finalG_1)%2  
            #print(MSB_0)     

            share0.set_threshold(MSB_0)
            share1.set_threshold(MSB_1)
            #print("MSB: ",(share0.threshold()+share1.threshold())%2)
            

            #Change domain
            

            a_0= MSB_0
            a_1= 0
            b_0= 0
            b_1= MSB_1
            '''
            t1=np.random.randint(0,2**16)
            t2=np.random.randint(0,2**16)
            t3=t1*t2

            t1_0= np.random.randint(0,2**16)
            t1_1= (t1-t1_0)%self.prime
            t2_0= np.random.randint(0,2**16)
            t2_1= (t2-t2_0)%self.prime
            t3_0= np.random.randint(0,2**16)
            t3_1= (t3-t3_0)%self.prime
            '''
            
            temp= additive_mul_prime(a_0,0,0,b_1,t1_0p,t1_1p,t2_0p,t2_1p,t3_0p,t3_1p,self.prime)

            vj_0=(a_0+b_0-2*temp[0])%self.prime
            vj_1=(a_1+b_1-2*temp[1])%self.prime
            share0.set_threshold([(1-vj_0)%self.prime, vj_0])
            share1.set_threshold([(-vj_1)%self.prime, vj_1])
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            #print(share0.threshold()[0])
            
            #print("comparison result: ",(share0.threshold()[0]+share1.threshold()[0])%self.prime)
            #print("MSB: ",(share0.threshold()[0]+share0.threshold()[1])%self.prime)
            #e_time=time.time()
            #execution_time = float(e_time - s_time)*1000
            #print("MSCDT eval a node:", execution_time, "mseconds")
            self.minus_sd(csp0,csp1,share0.left_child(),share1.left_child())
            self.minus_sd(csp0,csp1,share0.right_child(),share1.right_child())



    def minus(self, csp0,csp1,share0, share1):
        global select_attr
        global tree_eval
        if share0.is_leaf_node() and share1.is_leaf_node():
            #print("A leaf node")
            return
        else:
            attribute=share0.attribute()
            #s_time=time.time()
            start_time = time.time()
            share0.set_threshold((csp0.qDataShare0[attribute]-share0.threshold())%self.prime)
            share1.set_threshold((csp1.qDataShare1[attribute]-share1.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            #print("MSCDT eval a node:", execution_time, "mseconds")
            self.minus(csp0,csp1,share0.left_child(),share1.left_child())
            self.minus(csp0,csp1,share0.right_child(),share1.right_child())
    
    def minus_v2(self, csp0, csp1,share0, share1):
        global select_attr
        global tree_eval
        if share0.is_leaf_node() and share1.is_leaf_node():
            return
        else:
            attribute0=share0.attribute() #attribute=[attribute_index_share0 or 1,X0(1),Y0(1),Z0(1)
            attribute1=share1.attribute()

            start_time = time.time()
            p0x = (csp0.qDataShare0 + attribute0[1]) %self.prime
            p0y = (attribute0[0] + attribute0[2]) %self.prime
            p1x = (csp1.qDataShare1 + attribute1[1]) %self.prime
            p1y = (attribute1[0] + attribute1[2]) %self.prime
        
            z0 = (csp0.qDataShare0.dot((attribute0[0] + p1y)) - attribute0[2].dot(p1x) + attribute0[3])%self.prime
            z1 = (csp1.qDataShare1.dot((attribute1[0] + p0y)) - attribute1[2].dot(p0x) + attribute1[3])%self.prime
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            select_attr+=execution_time

            #print((csp0.qDataShare0+csp1.qDataShare1)%self.prime)
            #print((attribute0[0]+attribute1[0])%self.prime)
            #a=(attribute0[0]+attribute1[0])%self.prime
            #b=(csp0.qDataShare0+csp1.qDataShare1)%self.prime
            #print(a.dot(b))
            #print("z0+z1:",(z0+z1)%self.prime)
            #print("origin th: ",(share0.threshold()+share1.threshold())%self.prime)
            start_time = time.time()
            share0.set_threshold((z0-share0.threshold())%self.prime)
            share1.set_threshold((z1-share1.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            secret=(share0.threshold()+share1.threshold())%self.prime
            if secret > self.prime // 2:
                secret -= self.prime
            print("after:",secret)
            self.minus_v2(csp0,csp1,share0.left_child(),share1.left_child())
            self.minus_v2(csp0,csp1,share0.right_child(),share1.right_child())

    
    def minus_v4(self, QA0, QA1, share0, share1):
        global select_attr
        global tree_eval
        if share0.is_leaf_node() and share1.is_leaf_node():
            return
        else:
            start_time = time.time()
            #print(share0.attribute())
            share0.set_threshold((QA0[share0.attribute()]-share0.threshold())%self.prime)
            share1.set_threshold((QA1[share1.attribute()]-share1.threshold())%self.prime)
            end_time = time.time()
            execution_time = float(end_time - start_time)*1000
            tree_eval+=execution_time
            self.minus_v4(QA0,QA1,share0.left_child(),share1.left_child())
            self.minus_v4(QA0,QA1,share0.right_child(),share1.right_child())
            #print(share0.attribute())
        
    def node_eval(self, csp0, csp1,attri,prime):
        global select_attr
        global tree_eval
        select_attr=0
        tree_eval=0
        p0_node = csp0.model_share0_root
        p1_node = csp1.model_share1_root # root node of [M]2 
        self.prime=prime
        self.attrilist=attri
        if p0_node.is_leaf_node() and p1_node.is_leaf_node():
            return
        else:
            if self.version=="sdti":
                self.minus_sd(csp0,csp1,p0_node,p1_node)
                print("Attribute selection time: ", select_attr)
                return tree_eval
            if self.version=="zdwnn":
                start_time = time.time()
                z0, z1= ZDMVM(csp0.A0,csp1.A1, csp0.qDataShare0,csp1.qDataShare1,csp0.MVM_triples, csp1.MVM_triples,self.prime)
                end_time = time.time()
                execution_time = float(end_time - start_time)*1000
                print("Attribute selection time: ", execution_time)
                start_time = time.time()
                self.minus_zd(z0,z1,p0_node,p1_node)
                end_time = time.time()
                execution_time = float(end_time - start_time)*1000
                return execution_time
            if self.version==1:
                self.minus(csp0,csp1,p0_node,p1_node)
                return tree_eval
                
            if self.version==2:
                self.minus_v2(csp0,csp1,p0_node,p1_node)
                print("Attribute selection time: ", select_attr)
                return tree_eval
            if self.version==3:
                start_time = time.time()
                z0, z1= MVM(csp0.A0,csp1.A1, csp0.qDataShare0,csp1.qDataShare1,csp0.MVM_triples, csp1.MVM_triples,self.prime)
                #print("[origin] z0: ", z0)
                #print("[origin] z1: ", z1)
                end_time = time.time()
                execution_time = float(end_time - start_time)*1000
                print("Attribute selection time: ", execution_time)

                #start_time = time.time()
                self.minus_v4(z0,z1,p0_node,p1_node)
                #end_time = time.time()
                #execution_time = float(end_time - start_time)*1000
                return tree_eval
            if self.version==4:
                start_time = time.time()
                QA0 = (np.round(np.matmul(csp0.qDataShare0, csp0.M_inv)))%self.prime
                QA1 = (np.round(np.matmul(csp1.qDataShare1, csp1.M_inv)))%self.prime
                #print(QA0)
                #print(QA1)
                end_time = time.time()
                execution_time = float(end_time - start_time)*1000
                print("Attribute selection time: ", execution_time)
                #QA=(QA0+QA1)%self.prime
                #print(QA)
                
                self.minus_v4(QA0,QA1,p0_node,p1_node)
                
                return tree_eval

    def dfs_permute(self,share0,share1):
        if share0.is_leaf_node() and share0.is_leaf_node():
            self.label_share0.append(share0.threshold())
            self.label_share1.append(share1.threshold())
            share0.set_threshold(self.leaf_idx)
            share1.set_threshold(0)
            self.leaf_idx+=1
            return
        else:
            a=random.randint(-10,10)
            b=random.randint(-5,5)
            while b==0:
                b=random.randint(-5,5)
            #b=20
            #a=0
            share0.set_threshold((((share0.threshold()*b)%self.prime)+a)%self.prime)
            share1.set_threshold((share1.threshold()*b)%self.prime)
            if b<0: #permute
                temp=share0.left_child()
                share0.set_left_child(share0.right_child())
                share0.set_right_child(temp)
                temp=share1.left_child()
                share1.set_left_child(share1.right_child())
                share1.set_right_child(temp)

            self.dfs_permute(share0.left_child(),share1.left_child())
            self.dfs_permute(share0.right_child(),share1.right_child())

    def permute(self,csp0,csp1):
        self.leaf_idx=0
        self.label_share0.clear()
        self.label_share1.clear()
        p0_node=csp0.model_share0_root
        p1_node=csp1.model_share1_root
        self.dfs_permute(p0_node,p1_node)

    def leafnode_num(self):
        return self.leaf_idx
    
    def dfs_print_tree(self,share0,share1):
        if share0.is_leaf_node() and share0.is_leaf_node():
            print("label: ",reconstruct_secret([share0.threshold(),share1.threshold()],self.prime))
            return
        else:

            print(reconstruct_secret([share0.threshold(),share1.threshold()],self.prime))

            self.dfs_print_tree(share0.left_child(),share1.left_child())
            self.dfs_print_tree(share0.right_child(),share1.right_child())

    def print_tree(self,csp0,csp1):
        p0_node=csp0.model_share0_root
        p1_node=csp1.model_share1_root
        self.dfs_print_tree(p0_node,p1_node)
    
    def find_fake_index(self,csp0,csp1):
        p0_node=csp0.model_share0_root
        p1_node=csp1.model_share1_root
        while p0_node.is_leaf_node()!=True:
            recon=reconstruct_secret([p0_node.threshold(),p1_node.threshold()],self.prime)#reconstruct current node secret 
            #print(recon)
            if recon<=0: #go left
                p0_node=p0_node.left_child()
                p1_node=p1_node.left_child()
            else:
                p0_node=p0_node.right_child()
                p1_node=p1_node.right_child()
        return p0_node.threshold()+p1_node.threshold()
    
    def SDP(self,fake_leaf_idx_share0,fake_leaf_idx_share1):
        #for i in range(self.leaf_idx):
        #    print(reconstruct_secret([self.label_share0[i],self.label_share1[i]],self.prime))
        x0 = fake_leaf_idx_share0
        x1 = fake_leaf_idx_share1
        y0 = self.label_share0
        y1 = self.label_share1
        z0, z1 = dot_product_triples(len(x0), x0, x1, y0, y1,self.prime)
        return z0,z1        


    def dfs_ifgen(self,csp0,csp1,share0,share1,val):
        t1_0p=9889
        t1_1p=53080
        t2_0p=19644
        t2_1p=44576
        t3_0p=5948
        t3_1p=554176
        if share0.is_leaf_node() and share0.is_leaf_node():
            '''
            t1=np.random.randint(0,p_for_beaver)
            t2=np.random.randint(0,p_for_beaver)
            t3=t1*t2

            t1_0= np.random.randint(0,p_for_beaver)
            t1_1= (t1-t1_0)%self.prime
            t2_0= np.random.randint(0,p_for_beaver)
            t2_1= (t2-t2_0)%self.prime
            t3_0= np.random.randint(0,p_for_beaver)
            t3_1= (t3-t3_0)%self.prime
            '''
            #print((val[0]+val[1])%self.prime)
            result= additive_mul_prime(val[0],val[1],share0.threshold(),share1.threshold(),t1_0p,t1_1p,t2_0p,t2_1p,t3_0p,t3_1p,self.prime)
            #print((share0.threshold()+share1.threshold())%self.prime)
            csp0.set_resultshare0((csp0.resultshare0()+result[0])%self.prime)
            csp1.set_resultshare1((csp1.resultshare1()+result[1])%self.prime)
            #csp0.set_resultshare0((csp0.resultshare0()+share0.threshold()*val[0])%self.prime)
            #csp1.set_resultshare1((csp1.resultshare1()+share1.threshold()*val[0])%self.prime)
            #print("current result: ",(csp0.resultshare0()+csp1.resultshare1())%self.prime)
            return
        else:
            share0.set_polyval(val[0])
            share1.set_polyval(val[1])
            #print((val[0]+val[1])%self.prime)
            
            if(val[0]==1 and val[1]==1):
                self.dfs_ifgen(csp0,csp1,share0.left_child(),share1.left_child(),[(1-share0.threshold()[0])%self.prime, (-share1.threshold()[0])%self.prime])
                self.dfs_ifgen(csp0,csp1,share0.right_child(),share1.right_child(),[(1-share0.threshold()[1])%self.prime, (-share1.threshold()[1])%self.prime])
            else:
                '''
                t1=np.random.randint(0,p_for_beaver)
                t2=np.random.randint(0,p_for_beaver)
                t3=t1*t2

                t1_0= np.random.randint(0,p_for_beaver)
                t1_1= (t1-t1_0)%self.prime
                t2_0= np.random.randint(0,p_for_beaver)
                t2_1= (t2-t2_0)%self.prime
                t3_0= np.random.randint(0,p_for_beaver)
                t3_1= (t3-t3_0)%self.prime
                '''
                self.dfs_ifgen(csp0,csp1,share0.left_child(),share1.left_child(),additive_mul_prime(share0.pval(),share1.pval(),(1-share0.threshold()[0])%self.prime,(-share1.threshold()[0])%self.prime,t1_0p,t1_1p,t2_0p,t2_1p,t3_0p,t3_1p,self.prime))                                
                #self.dfs_ifgen(csp0,csp1,share0.left_child(),share1.left_child(),[(share0.pval()*(1-share0.threshold()[0])), (share1.pval()*(-share1.threshold()[0]))])
                '''
                t1=np.random.randint(0,p_for_beaver)
                t2=np.random.randint(0,p_for_beaver)
                t3=t1*t2

                t1_0= np.random.randint(0,p_for_beaver)
                t1_1= (t1-t1_0)%self.prime
                t2_0= np.random.randint(0,p_for_beaver)
                t2_1= (t2-t2_0)%self.prime
                t3_0= np.random.randint(0,p_for_beaver)
                t3_1= (t3-t3_0)%self.prime
                '''
                self.dfs_ifgen(csp0,csp1,share0.right_child(),share1.right_child(),additive_mul_prime(share0.pval(),share1.pval(),(1-share0.threshold()[1])%self.prime,(-share1.threshold()[1])%self.prime,t1_0p,t1_1p,t2_0p,t2_1p,t3_0p,t3_1p,self.prime))

