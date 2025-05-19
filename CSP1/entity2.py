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
        logging.FileHandler("CSP1.log", "w"),
        logging.StreamHandler()
    ]
)
CSP1_Logger = logging.getLogger("CSP1")

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

        self.queue_shared_MO_CSU = queue.Queue()

        self.all_var_ready_for_CSU = threading.Event()

        self.cspthread = threading.Thread(
            target=self.server_threading,
            name="CSP1_main",
            #args=(client,)
        )
        self.cspthread.start()
    
    def start_connection(self):
        retry_counts = 0
        self.csp0_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        RETRIES = int(self.config['DEFAULT']['RETRIES'])
        while retry_counts < RETRIES:
            try:
                self.csp0_server.connect((self.config['DEFAULT']['CSP0_SEVER_IP'], int(self.config['DEFAULT']['CSP0_SEVER_PORT'])))
                self.csp0_server.send('CSP'.encode())
                break
            except Exception as e:
                print("CSU failed to connect CSP0. Retry", retry_counts)
                print(e)
                retry_counts += 1
                time.sleep(0.3)
                if retry_counts == int(self.config['DEFAULT']['RETRIES']):
                    print("Suspend at CSP1 connect to CSP0.\n Enter anything to reset and retry the connection.")
                    input()
                    retry_counts = 0


    def server_threading(self): ##[Socket Communication modified]
        self.s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_socket.bind((self.config['DEFAULT']['CSP1_SEVER_IP'], int(self.config['DEFAULT']['CSP1_SEVER_PORT'])))
        self.s_socket.listen(5)

        self.socket_count = 0

        CSP1_Logger.info("[CSP1] Listening on {}:{}".format(self.config['DEFAULT']['CSP1_SEVER_IP'], int(self.config['DEFAULT']['CSP1_SEVER_PORT'])))
        while self.socket_count < 2:
            client, address = self.s_socket.accept()
            CSP1_Logger.info("[CSP1] Connected by {}".format(address))

            data = client.recv(3)
            if(data == b' MO'):
                self.socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_MO,
                    name="CSP0_MO",
                    args=(client,)
                ).start()
            elif(data == b'CSU'):
                self.socket_count += 1
                threading.Thread(
                    target=self.server_handle_recv_CSU,
                    name="CSP0_CSU",
                    args=(client,)
                ).start()
            else:
                print("[CSP1] Can not distingusih MO CSU: ", data)

    def server_handle_recv_MO(self, client:socket.socket):
        CSP1_Logger.info("[CSP1] MO connected")
        ### receive MO model tree share, attribute list and prime
        self.prime = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        CSP1_Logger.info("[CSP1] Received prime from MO.")
        tree_share_pickle_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(tree_share_pickle_size, socket.MSG_WAITALL)
        CSP1_Logger.info("[CSP1] Received model share from MO.")
        #print("[CSP] tree share size: ", len(data))
        self.queue_shared_MO_CSU.put(data)
        attri_list_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        data = client.recv(attri_list_size, socket.MSG_WAITALL)
        CSP1_Logger.info("[CSP1] Received attribute list from MO.")
        self.attri_list = data.decode().split("' '")
        #print("[CSP] attribute list: ", self.attri_list)

        if(self.version == 3):
            data = client.recv(6, socket.MSG_WAITALL)
            #print(data)
            A1_dim = tuple(bytes_to_int_array(data))
            #print("[CSP] A1_dim",A1_dim)
            data = client.recv(A1_dim[0]*A1_dim[1]*8, socket.MSG_WAITALL)
            self.A1 = np.frombuffer(data, dtype=(np.int64)).reshape(A1_dim)
            CSP1_Logger.info("[CSP1] Received share of permutation matrix (A1) from MO.")
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
            CSP1_Logger.info("[CSP1] Received dot-product triples from MO.")
            
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

        self.queue_shared_MO_CSU.put(self.attri_list)
        self.queue_shared_MO_CSU.put(self.prime)
        self.all_var_ready_for_CSU.set()

    def server_handle_recv_CSU(self, client:socket.socket):
        CSP1_Logger.info("[CSP1] CSU connected")
        ### blocking until necessary variable assigned
        self.all_var_ready_for_CSU.wait()
        ### receive CSU query data
        query_size = int.from_bytes(client.recv(PRIME_SIZE, socket.MSG_WAITALL), 'big')
        self.qDataShare1 = np.frombuffer(client.recv(query_size*8, socket.MSG_WAITALL),dtype=(np.int64))
        CSP1_Logger.info("[CSP1] Received CSU query data share from CSU.")
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
            data = self.queue_shared_MO_CSU.get()
            self.model_share1_root = pickle.loads(data)
            self.A1 = self.queue_shared_MO_CSU.get()
            self.MVM_triples = self.queue_shared_MO_CSU.get()
            self.attri_list = self.queue_shared_MO_CSU.get()
            self.prime = self.queue_shared_MO_CSU.get()

            p1_node = self.model_share1_root
            CSP1_Logger.info("[CSP1] Start NodeEval.")
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
            CSP1_Logger.info("[CSP1] End NodeEval.")
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
            data = self.queue_shared_MO_CSU.get()
            self.model_share1_root = pickle.loads(data)
            self.M_inv = self.queue_shared_MO_CSU.get()
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
        CSP1_Logger.info("[CSP1] Start NodePermute.")
        self.permute()
        CSP1_Logger.info("[CSP1] End NodePermute.")

        CSP1_Logger.info("[CSP1] Send model share to CSU...")
        data = pickle.dumps(self.model_share1_root)
        client.send(len(data).to_bytes(PRIME_SIZE, 'big'))
        client.send(data)
        CSP1_Logger.info("[CSP1] Success sending model share to CSU...")
        CSP1_Logger.info("[CSP1] Send label share to CSU...")
        client.send(self.leaf_idx.to_bytes(PRIME_SIZE, 'big'))
        client.send(len(self.label_share1).to_bytes(PRIME_SIZE, 'big')) 
        client.send(int_array_to_bytes(self.label_share1)) 
        CSP1_Logger.info("[CSP1] Success sending label share to CSU...")
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