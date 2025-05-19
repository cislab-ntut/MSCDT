import socket
import time
import numpy as np
import random
import pickle
import queue

from secure import PRIME_SIZE

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("CSU.log", "w"),
        logging.StreamHandler()
    ]
)
CSU_Logger = logging.getLogger("CSU")

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


class CloudServiceUser():
    
    def __init__(self, config, version=1):
        self.version=version
        self.qData = None
        self.prime = None
        self.config = config

        # self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        # self.csp0_server.connect((config['DEFAULT']['CSP0_SEVER_IP'], int(config['DEFAULT']['CSP0_SEVER_PORT'])))           ##[Socket Communication modified]
        # self.csp0_server.send('CSU'.encode())                                ##[Socket Communication modified]
        # self.csp1_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        # self.csp1_server.connect((config['DEFAULT']['CSP1_SEVER_IP'], int(config['DEFAULT']['CSP1_SEVER_PORT'])))           ##[Socket Communication modified]
        # self.csp1_server.send('CSU'.encode())                                ##[Socket Communication modified]

        self.z_reconstruct_queue = queue.Queue()

    def start_connection(self):
        retry_counts = 0
        self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        RETRIES = int(self.config['DEFAULT']['RETRIES'])
        while retry_counts < RETRIES:
            try:
                self.csp0_server.connect((self.config['DEFAULT']['CSP0_SEVER_IP'], int(self.config['DEFAULT']['CSP0_SEVER_PORT']))) ##[Socket Communication modified]
                self.csp0_server.send('CSU'.encode())
                ### receive prime
                self.prime = int.from_bytes(self.csp0_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
                break
            except Exception as e:
                print("CSU failed to connect CSP0. Retry", retry_counts)
                print(e)
                retry_counts += 1
                time.sleep(0.3)
                if retry_counts == int(self.config['DEFAULT']['RETRIES']):
                    print("Suspend at CSU connect to CSP0.\n Enter anything to reset and retry the connection.")
                    input()
                    retry_counts = 0
        # if retry_counts == self.config['DEFAULT']['RETRIES']:
        #     print("Suspend at MO connect to CSP0.\n Enter anything to manually retry the connection.")
        #     input()
        retry_counts = 0
        self.csp1_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ##[Socket Communication modified]
        while retry_counts < RETRIES:
            try:
                self.csp1_server.connect((self.config['DEFAULT']['CSP1_SEVER_IP'], int(self.config['DEFAULT']['CSP1_SEVER_PORT']))) ##[Socket Communication modified]
                self.csp1_server.send('CSU'.encode())
                break
            except Exception as e:
                print("CSU failed to connect CSP1.", retry_counts)
                print(e)
                retry_counts += 1
                time.sleep(0.3)
                if retry_counts == int(self.config['DEFAULT']['RETRIES']):
                    print("Suspend at CSU connect to CSP1.\n Enter anything to reset and retry the connection.")
                    input()
                    retry_counts = 0

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
        
    def set_query_data_v3(self, data):
        #self.prime=prime
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

    def send_query_data_to_csp(self):
        # csp0.set_query_data_share0(self.qDataShare0)  ##[Socket Communication modified]
        # csp1.set_query_data_share1(self.qDataShare1)  ##[Socket Communication modified]
        CSU_Logger.info("[CSU] Send query data share to CSP0...")
        self.csp0_server.send(len(self.qDataShare0).to_bytes(PRIME_SIZE, 'big')) ##[Socket Communication modified]
        self.csp0_server.send(self.qDataShare0.tobytes())                        ##[Socket Communication modified]
        CSU_Logger.info("[CSU] Success sending model share to CSP0...")
        #print("[CSU] qDataShare0: ", self.qDataShare0)
        CSU_Logger.info("[CSU] Send query data share to CSP1...")
        self.csp1_server.send(len(self.qDataShare1).to_bytes(PRIME_SIZE, 'big')) ##[Socket Communication modified]
        self.csp1_server.send(self.qDataShare1.tobytes())                        ##[Socket Communication modified]
        CSU_Logger.info("[CSU] Success sending model share to CSP1...")
        #print("[CSU] qDataShare1: ", self.qDataShare1)

    def receive_result_items_and_compute_shared_result(self,):
        tree_share_pickle_size = int.from_bytes(self.csp0_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big') #model_share0_root
        data = self.csp0_server.recv(tree_share_pickle_size, socket.MSG_WAITALL)          #model_share0_root
        model_share0_root = pickle.loads(data)                                            #model_share0_root
        CSU_Logger.info("[CSU] Received model share from CSP0.")
        label_share0_size = int.from_bytes(self.csp0_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')   #label_share0
        data = self.csp0_server.recv(label_share0_size*PRIME_SIZE, socket.MSG_WAITALL)                     #label_share0
        label_share0 = bytes_to_int_array(data)                                        #label_share0
        CSU_Logger.info("[CSU] Received label share from CSP0.")

        tree_share_pickle_size = int.from_bytes(self.csp1_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big') #model_share1_root
        data = self.csp1_server.recv(tree_share_pickle_size, socket.MSG_WAITALL)          #model_share1_root
        model_share1_root = pickle.loads(data)                                            #model_share1_root
        CSU_Logger.info("[CSU] Received model share from CSP1.")
        leaf_num = int.from_bytes(self.csp1_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')       #leaf_num
        CSU_Logger.info("[CSU] Received leaf number from CSP1.")
        #print("[CSU] leaf_num: ", leaf_num)
        label_share1_size = int.from_bytes(self.csp1_server.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')  #label_share1
        data = self.csp1_server.recv(label_share1_size*PRIME_SIZE, socket.MSG_WAITALL)                    #label_share1
        label_share1 = bytes_to_int_array(data)                                       #label_share1
        CSU_Logger.info("[CSU] Received label share from CSP1.")
        
        CSU_Logger.info("[CSU] Start finding fake index.")
        index = self.find_fake_index(model_share0_root, model_share1_root)
        CSU_Logger.info("[CSU] End finding fake index.")
        #print("[CSU] index: ", index)

        fake_leaf_idx=np.zeros((leaf_num,),dtype=int)
        fake_leaf_idx[index]=1
        fake_leaf_idx_share0=np.random.randint(0,self.prime-1,(leaf_num,))
        fake_leaf_idx_share1=np.zeros((leaf_num,),dtype=int)
        fake_leaf_idx_share1 = (fake_leaf_idx-fake_leaf_idx_share0)%self.prime

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