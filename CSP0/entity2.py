import socket
import time
import numpy as np
import random
import pickle
import queue
import threading
from secure import PRIME_SIZE

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("CSP0.log", "w"),
        logging.StreamHandler()
    ]
)
CSP0_Logger = logging.getLogger("CSP0")

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

        self.all_var_ready_for_CSU = threading.Event()
        
        self.cspthread = threading.Thread(  ##[Socket Communication modified]
            target=self.server_threading,
            name="CSP0_main",
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

        CSP0_Logger.info("[CSP0] Listening on  {}:{}".format(self.config['DEFAULT']['CSP0_SEVER_IP'], int(self.config['DEFAULT']['CSP0_SEVER_PORT'])))
        while socket_count < 3:
            client, self.address = self.s_socket.accept()
            CSP0_Logger.info("[CSP0] Connected by  {}".format(self.address))
            
            #while True:
            data = client.recv(3)
            if(data == b' MO'): ## MO connection thread
                socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_MO,
                    name="CSP0_MO",
                    args=(client,)
                ).start()
            elif(data == b'CSU'):
                socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_CSU,
                    name="CSP0_CSU",
                    args=(client,)
                ).start()
            elif(data == b'CSP'):
                socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_CSP,
                    name="CSP0_CSP",
                    args=(client,)
                ).start()
            else:
                print("[CSP0] Can not distingusih MO CSU CSP: ", data)
    
    def server_handle_recv_MO(self, client: socket.socket):
        CSP0_Logger.info("[CSP0] MO connected")
        ### receive model tree share, attribute list and prime
        self.prime = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        CSP0_Logger.info("[CSP0] Received prime from MO.")
        #print("[CSP0] PRIME: ", self.prime)
        tree_share_pickle_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(tree_share_pickle_size, socket.MSG_WAITALL)
        CSP0_Logger.info("[CSP0] Received model share from MO.")
        #print("[CSP] tree share size: ", len(data))
        #self.model_share0_root = pickle.loads(data)
        self.queue_shared_MO_CSU.put(data)
        data = client.recv(PRIME_SIZE, socket.MSG_WAITALL)
        attri_list_size = int.from_bytes(data, 'big')
        data = client.recv(attri_list_size, socket.MSG_WAITALL)
        CSP0_Logger.info("[CSP0] Received attribute list from MO.")
        self.attri_list = data.decode().split("' '")
        #print("[CSP] attribute list: ", self.attri_list)

        if(self.version == 3):
            ### receive A0
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            A0_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] A0_dim", A0_dim)
            data = client.recv(A0_dim[0] * A0_dim[1] * 8, socket.MSG_WAITALL)
            self.A0 = np.frombuffer(data, dtype=(np.int64)).reshape(A0_dim)
            CSP0_Logger.info("[CSP0] Received share of permutation matrix (A0) from MO.")
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
            CSP0_Logger.info("[CSP0] Received dot-product triples from MO.")
            
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

        self.queue_shared_MO_CSU.put(self.attri_list)
        self.queue_shared_MO_CSU.put(self.prime)
        self.all_var_ready_for_CSU.set()

    def server_handle_recv_CSU(self, client: socket.socket):
        CSP0_Logger.info("[CSP0] CSU connected")
        ### blocking until necessary variable assigned
        self.all_var_ready_for_CSU.wait()
        ## send prime
        CSP0_Logger.info("[CSP0] Send prime to CSU...")
        client.send(int(self.prime).to_bytes(PRIME_SIZE, 'big'))
        CSP0_Logger.info("[CSP0] Success sending  prime to CSU...")
        ### receive CSU query data
        query_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        self.qDataShare0 = np.frombuffer(client.recv(query_size*8, socket.MSG_WAITALL),dtype=(np.int64))
        CSP0_Logger.info("[CSP0] Received CSU query data share from CSU.")
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
            data = self.queue_shared_MO_CSU.get()
            self.model_share0_root = pickle.loads(data)
            self.A0 = self.queue_shared_MO_CSU.get()
            self.MVM_triples = self.queue_shared_MO_CSU.get()
            self.attri_list = self.queue_shared_MO_CSU.get()
            self.prime = self.queue_shared_MO_CSU.get()

            p0_node = self.model_share0_root
            CSP0_Logger.info("[CSP0] Start NodeEval.")
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
            CSP0_Logger.info("[CSP0] End NodeEval.")
            #end_time = time.time()
            #execution_time = float(end_time - start_time)*1000
            # self.permute()

            # data = pickle.dumps(self.model_share0_root)
            # client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
            # client.send(data)
            # client.send(len(self.label_share0).to_bytes(PRIME_SIZE, 'big')) 
            # client.send(int_array_to_bytes(self.label_share0)) 

        if self.version==4:
            data = self.queue_shared_MO_CSU.get()
            self.model_share0_root = pickle.loads(data)
            self.M_inv = self.queue_shared_MO_CSU.get()
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
        CSP0_Logger.info("[CSP0] Start NodePermute.")
        self.permute()
        CSP0_Logger.info("[CSP0] End NodePermute.")

        CSP0_Logger.info("[CSP0] Send model share to CSU...")
        data = pickle.dumps(self.model_share0_root)
        client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        client.send(data)
        CSP0_Logger.info("[CSP0] Success sending model share to CSU...")
        CSP0_Logger.info("[CSP0] Send label share to CSU...")
        client.send(len(self.label_share0).to_bytes(PRIME_SIZE, 'big')) 
        client.send(int_array_to_bytes(self.label_share0))
        CSP0_Logger.info("[CSP0] Success sending label share to CSU...")
        return tree_eval

    def server_handle_recv_CSP(self, client: socket.socket):
        CSP0_Logger.info("[CSP0] CSP1 connected")
        if self.version==3:
            ### receive p1x p1y from CSP1
            data = client.recv(PRIME_SIZE * 2, socket.MSG_WAITALL)
            p1x_dim = tuple(bytes_to_int_array(data))
            p1x = np.frombuffer(client.recv(p1x_dim[0] * p1x_dim[1] * 8, socket.MSG_WAITALL),
                            dtype=(np.int64)).reshape(p1x_dim)
            CSP0_Logger.info("[CSP0] Received dot-product element x1+X1 from CSP1.")
            p1y_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
            p1y = np.frombuffer(client.recv(p1y_size*8, socket.MSG_WAITALL),dtype=(np.int64))
            CSP0_Logger.info("[CSP0] Received dot-product element y1+Y1 from CSP1.")

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