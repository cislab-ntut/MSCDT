import socket
import time
from structure2 import Node, Timer
import numpy as np
import random
import pickle
from secure import PRIME_SIZE

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("MO.log", "w"),
        logging.StreamHandler()
    ]
)
MO_Logger = logging.getLogger("MO")

def int_array_to_bytes(arr, byte_order='big', byte_size=PRIME_SIZE):
    return b''.join(int(i).to_bytes(byte_size, byte_order) for i in arr)

def bytes_to_int_array(byte_data, byte_order='big', byte_size=PRIME_SIZE):
    return [int.from_bytes(byte_data[i:i+byte_size], byte_order)
            for i in range(0, len(byte_data), byte_size)]

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

class ModelOwner():

    def __init__(self, config, version=1):
        self._root_node = None
        self.prime=None
        self.attrlist = None
        self._root_node_shares = None
        self.internal_node_attribute=[]
        self.internal_node_num=0
        self.leaf_node_num=0
        self.version = version
        self.config = config
    
    def start_connection(self):
        retry_counts = 0
        self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        RETRIES = int(self.config['DEFAULT']['RETRIES'])
        while retry_counts < RETRIES:
            try:
                self.csp0_server.connect((self.config['DEFAULT']['CSP0_SEVER_IP'], int(self.config['DEFAULT']['CSP0_SEVER_PORT']))) 
                self.csp0_server.send(' MO'.encode())
                break
            except Exception as e:
                print("MO failed to connect CSP0. Retry", retry_counts)
                print(e)
                time.sleep(0.3)
                retry_counts += 1
                if retry_counts == int(self.config['DEFAULT']['RETRIES']):
                    print("Suspend at MO connect to CSP0.\n Enter anything to reset and retry the connection.")
                    input()
                    retry_counts = 0
        retry_counts = 0
        self.csp1_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while retry_counts < RETRIES:
            try:
                self.csp1_server.connect((self.config['DEFAULT']['CSP1_SEVER_IP'], int(self.config['DEFAULT']['CSP1_SEVER_PORT'])))
                self.csp1_server.send(' MO'.encode())
                break
            except Exception as e:
                print("MO failed to connect CSP1. Retry", retry_counts)
                print(e)
                time.sleep(0.3)
                retry_counts += 1
                if retry_counts == int(self.config['DEFAULT']['RETRIES']):
                    print("Suspend at MO connect to CSP1.\n Enter anything to reset and retry the connection.")
                    input()
                    retry_counts = 0

    def input_model_and_split_into_shares(self, root_node, attrl, prime):
        self.prime=prime 
        self.attrlist=attrl
        self._root_node = root_node
        return self.split_model_into_shares()
    
    def get_model(self) -> Node:
        return self._root_node

    def split_model_into_shares(self) -> list:
        MO_Logger.info("[MO] Start model transforming ... (split into shares)")
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
        MO_Logger.info("[MO] End model transforming ... (split into shares)")
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
    
    def gen_permute_matrix_zd(self):
        A=np.zeros((self.internal_node_num,len(self.attrlist))) 


        for i in range(self.internal_node_num):
            A[i][self.internal_node_attribute[i]]=1
        A0= np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        A1 = (A-A0)%self.prime
        #csp0.A0=A0
        #csp1.A1=A1

        G1 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        g2 = np.random.randint(0,self.prime-1,(len(self.attrlist)))
        g3 = np.dot(G1,g2)
        G1_0 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist))) 
        G1_1 = (G1-G1_0)%self.prime
        g2_0 = np.random.randint(0,self.prime-1,(len(self.attrlist)))
        g2_1 = (g2-g2_0)%self.prime
        g3_0 = np.random.randint(0,self.prime-1,(self.internal_node_num))
        g3_1 = (g3-g3_0)%self.prime

        #csp0.MVM_triples=(G1_0,g2_0,g3_0)
        #csp1.MVM_triples=(G1_1,g2_1,g3_1)
        return A0, A1


    def gen_permute_matrix(self):
        A=np.zeros((self.internal_node_num,len(self.attrlist)), dtype=np.int64)

        for i in range(self.internal_node_num):
            A[i][self.internal_node_attribute[i]]=1
        A0= np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        A1 = (A-A0)%self.prime
        #csp0.A0=A0
        #csp1.A1=A1
        #print(A0.shape)
        #print(A0)
        #print(A1.shape)
        #print(A1)
        MO_Logger.info("[MO] Send share of permutation matrix (A0) to CSP0... ")
        self.csp0_server.send(int_array_to_bytes(A0.shape))
        self.csp0_server.send(A0.tobytes())
        MO_Logger.info("[MO] Success sending A0 to CSP0.")
        MO_Logger.info("[MO] Send share of permutation matrix (A1) to CSP1... ")
        self.csp1_server.send(int_array_to_bytes(A1.shape))
        self.csp1_server.send(A1.tobytes())
        MO_Logger.info("[MO] Success sending A1 to CSP1.")
        

        X0 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        X1 = np.random.randint(0,self.prime-1,(self.internal_node_num,len(self.attrlist)))
        Y0 = np.random.randint(0,self.prime-1,(len(self.attrlist),))
        Y1 = np.random.randint(0,self.prime-1,(len(self.attrlist),))
        T = np.random.randint(0,self.prime-1,(self.internal_node_num,))
        Z0 = (np.dot(X0,Y1) +T)%self.prime
        Z1 = (np.dot(X1,Y0) - T)%self.prime
        #csp0.MVM_triples=(X0,Y0,Z0)
        #csp1.MVM_triples=(X1,Y1,Z1)
        # print("X0 shape: ", X0.shape)
        # print("X0 ", X0)
        # print("Y0 shape: ", Y0.shape)
        # print("Y0 ", Y0)
        # print("Z0 shape: ", Z0.shape)
        # print("Z0 ", Z0)
        MO_Logger.info("[MO] Send dot-product triples to CSP0... ")
        self.csp0_server.send(int_array_to_bytes(X0.shape))
        self.csp0_server.send(X0.tobytes())
        self.csp0_server.send(Y0.tobytes())
        self.csp0_server.send(Z0.tobytes())
        MO_Logger.info("[MO] Success sending dot-product triples to CSP0... ")
        MO_Logger.info("[MO] Send dot-product triples to CSP1... ")
        self.csp1_server.send(int_array_to_bytes(X1.shape))
        self.csp1_server.send(X1.tobytes())
        self.csp1_server.send(Y1.tobytes())
        self.csp1_server.send(Z1.tobytes())
        MO_Logger.info("[MO] Success sending dot-product triples to CSP1... ")
        return A0, A1


    def gen_pk_sk_matrix(self):
        A=np.zeros((len(self.attrlist),self.internal_node_num)) 

        M = np.random.randint(self.prime-1, size=(self.internal_node_num, self.internal_node_num))
        M_inv = np.linalg.inv(M)

        for i in range(self.internal_node_num):
            A[self.internal_node_attribute[i]][i]=1

        AM= np.matmul(A,M)

        #csp0.M_inv=M_inv
        #csp1.M_inv=M_inv
        #print("M_inv: \n", M_inv)
        self.csp0_server.send(int_array_to_bytes(M_inv.shape))
        self.csp0_server.send(M_inv.tobytes())
        self.csp1_server.send(int_array_to_bytes(M_inv.shape))
        self.csp1_server.send(M_inv.tobytes())
        
        return AM, M_inv

    def set_shares_to_two_parties(self):
        self._root_node_shares=self._root_node_shares 
        #csp0.set_model_share0_root_node(self._root_node_shares[0],self.attrlist,self.prime)
        #csp1.set_model_share1_root_node(self._root_node_shares[1],self.attrlist,self.prime)

        ### send prime
        MO_Logger.info("[MO] Send share to CSP0...")
        self.csp0_server.send(int(self.prime).to_bytes(PRIME_SIZE, 'big'))
        #print("PRIME: ", self.prime)
        MO_Logger.info("[MO] Success sending share to CSP0.")
        MO_Logger.info("[MO] Send share to CSP1...")
        self.csp1_server.send(int(self.prime).to_bytes(PRIME_SIZE, 'big'))
        MO_Logger.info("[MO] Success sending share to CSP1.")

        ### send model tree share
        MO_Logger.info("[MO] Send model share to CSP0...")
        data = pickle.dumps(self._root_node_shares[0])
        self.csp0_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp0_server.send(data)
        MO_Logger.info("[MO] Success sending model share to CSP0.")
        MO_Logger.info("[MO] Send model share to CSP1...")
        data = pickle.dumps(self._root_node_shares[1])
        self.csp1_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp1_server.send(data)
        MO_Logger.info("[MO] Success sending model share to CSP1.")

        ### send attribute list
        #print(self.attrlist)
        MO_Logger.info("[MO] Send attribute list to CSP0...")
        data = str(self.attrlist)[2:-2].encode()
        self.csp0_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp0_server.send(data)
        MO_Logger.info("[MO] Success sending attribute list to CSP0.")
        MO_Logger.info("[MO] Send attribute list to CSP1...")
        data = str(self.attrlist)[2:-2].encode()
        self.csp1_server.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        self.csp1_server.send(data)
        MO_Logger.info("[MO] Success sending attribute list to CSP1.")

    def close_csp_connection(self):
        MO_Logger.info("[MO] Close CSP0 connection...")
        self.csp0_server.close()
        MO_Logger.info("[MO] Success close CSP0 connection.")
        MO_Logger.info("[MO] Close CSP1 connection...")
        self.csp1_server.close()
        MO_Logger.info("[MO] Success close CSP1 connection.")