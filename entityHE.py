import socket
import time
from structure2 import Node, Timer
import numpy as np
#import galois
import random
import sympy as sy
import gmpy2
from phe import paillier


##测试代码，进行同态加密运算时间计算
import time
import numpy as np
from phe import paillier
start_time = time.time()
u_a=np.random.uniform(low=2,high=8,size=(1,20000))  
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
encrypted_u_a = np.asarray([public_key.encrypt(x) for x in u_a[0]])
elapsed_time = time.time() - start_time
print('elapsed_time: {:.3f}s'.format(elapsed_time))


class ModelOwner():

    def __init__(self):#(self, model_tree_root_node):
        self._root_node = None
        self.attrlist = None
        self._root_node_enc = None
        self.leaf_node_label=[]
        self.mpk = None
        self.leaf_node_num=0
        self.msk = None
        self.upk=None
        self.zciph= None
        self.ociph=None

    def gen_keypair(self):
        self.mpk, self.msk = paillier.generate_paillier_keypair()

    def gen_z_o_cipher(self):
        self.zciph=self.upk.encrypt(0)
        self.ociph=self.upk.encrypt(1)
    def get_upk(self,csu):
        self.upk=csu.upk
        self.gen_z_o_cipher()
    def input_model_and_encrypt(self, root_node,attrl):
        self.attrlist=attrl
        self._root_node = root_node
        return self.encrypt_model()
    
    def get_model(self) -> Node:
        return self._root_node

    def encrypt_model(self) -> list:
        self.leaf_node_num = 0
        if self._root_node == None:
            print("Please input model.")
            return None
        self._root_node_enc = self._encrypt_node(self._root_node)
        #print("MO self._root_node_shares: ", self._root_node_shares)
        return self._root_node_enc

    # Combine copy1, cop2 into one function
    
    def _encrypt_node(self, original_node):
        enc_left_child = None
        enc_right_child = None

        if original_node.is_leaf_node():
            self.leaf_node_num+=1
            self.leaf_node_label.append(original_node.threshold().item())
            #Split leaf nodes into shares
            return Node(attribute=original_node.attribute(), 
                        threshold=-2, is_leaf_node=True)
        else:
            enc_left_child = self._encrypt_node(original_node.left_child())
            enc_right_child = self._encrypt_node(original_node.right_child())

            return Node(attribute=original_node.attribute(), 
                        threshold=self.mpk.encrypt(original_node.threshold()), 
                        left_child=enc_left_child, 
                        right_child=enc_right_child, 
                        is_leaf_node=False)
    
    def set_model_to_csp(self, csp):
        self._root_node_enc=self._root_node_enc
        csp.set_model_root_node(self._root_node_enc,self.attrlist)



class CloudServiceProvider():

    def __init__(self):
        self.model_root = None
        self.attri_list = None
        self.prime = None
        self.upk=None
        self.refresh=None

    def set_model_root_node(self, _root_node_enc,attrlist):
        self.attri_list=attrlist
        self.model_root = _root_node_enc
    
    def set_upk(self,pk):
        self.upk=pk
        self.refresh=self.upk.encrypt(0)
    def get_refresh(self):
        return self.refresh
    
    def set_query_data_enc(self, qDataEnc):
        self.qDataEnc = qDataEnc


class CloudServiceUser():
    
    def __init__(self):
        self.qData = None
        self.prime = None
        self.upk = None 
        self.usk = None

    def gen_keypair(self):
        self.upk,self.usk = paillier.generate_paillier_keypair()

    def set_query_data(self, data, mo):
        self.qData = data*10
        self.qDataEnc = [mo.mpk.encrypt(self.qData[i].item()) for i in range(len(self.qData))]

    def send_query_data_to_csp(self, csp):
        csp.set_query_data_enc(self.qDataEnc)
        csp.set_upk(self.upk)
        


class Protocol():
    
    def __init__(self):
        self.prime = None
        self.attrilist=None
        self.permute_table=[]
        self.real_index_vector=[]
        self.leaf_idx=0
        self.result=0

    def minus(self, csp, p_node):
        if p_node.is_leaf_node():
            return
        else:
            attribute=p_node.attribute()
            #start_time = time.time()
            p_node.set_threshold(csp.qDataEnc[attribute]-p_node.threshold())
            #end_time=time.time()
            #execution_time = float(end_time - start_time)*1000
            #print("HEDT evaluate a node:", execution_time, "mseconds")
            self.minus(csp,p_node.left_child())
            self.minus(csp,p_node.right_child())
            

    def node_eval(self, csp, attri):
        p_node = csp.model_root
        self.attrilist=attri
        if p_node.is_leaf_node():
            return
        else:
            self.minus(csp, p_node)

    def dfs_permute(self,p_node):
        if p_node.is_leaf_node() :
            p_node.set_threshold(self.leaf_idx)
            self.leaf_idx+=1
            return
        else:
            a=random.randint(-10,10)
            b=random.randint(-5,5)
            a=0
            b=1
            while b==0:
                b=random.randint(-5,5)
            p_node.id=len(self.permute_table)
            self.permute_table.append(b)
            #b=1
            #a=0
            
            #start_time = time.time()
            p_node.set_threshold((p_node.threshold()*b)+a)
            #end_time=time.time()
            #execution_time = float(end_time - start_time)*1000
            #print("HEDT permute a node:", execution_time, "mseconds")
            if b<0: #permute
                temp=p_node.left_child()
                p_node.set_left_child(p_node.right_child())
                p_node.set_right_child(temp)

            self.dfs_permute(p_node.left_child())
            self.dfs_permute(p_node.right_child())

    def permute(self,csp):
        self.permute_table=[]
        self.leaf_idx=0
        p_node=csp.model_root
        self.dfs_permute(p_node)

    def dfs_reverse_permute(self,p_node):
        if p_node.is_leaf_node():
            return
        else:
            self.dfs_reverse_permute(p_node.right_child())
            self.dfs_reverse_permute(p_node.left_child())
            if self.permute_table[p_node.id]<0: #permute
                temp=p_node.left_child()
                p_node.set_left_child(p_node.right_child())
                p_node.set_right_child(temp)

    def reverse_permute(self,csp):
        p_node=csp.model_root
        self.dfs_reverse_permute(p_node)


    def dfs_reverse_permute_fake_index(self,p_node,refresh,fake_index):
        if p_node.is_leaf_node():
            self.real_index_vector.append(fake_index[p_node.threshold()]+refresh)
            return
        else:
            self.dfs_reverse_permute_fake_index(p_node.left_child(),refresh,fake_index)
            self.dfs_reverse_permute_fake_index(p_node.right_child(),refresh,fake_index)
        return self.real_index_vector


    def reverse_permute_fake_index(self,csp,csu,fake_index):
        self.real_index_vector=[]
        
        p_node=csp.model_root
        return self.dfs_reverse_permute_fake_index(p_node,csp.get_refresh(),fake_index)

    def hedt_find_result(self,mo,index_vector):
        self.result=0
        c=[a*b for a,b in zip(index_vector, mo.leaf_node_label)]
        self.result=sum(c)
        #for i in range(len(mo.leaf_node_label)):
        #    self.result+=index_vector[i]*mo.leaf_node_label[i].item()
        return self.result
    def user_decrypt_result(self,csu,res):
        return csu.usk.decrypt(res)

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
    
    def find_fake_index(self,mo,csp,csu):
        p_node=csp.model_root
        while p_node.is_leaf_node()!=True:
            if mo.msk.decrypt(p_node.threshold())<=0: #go left
                p_node=p_node.left_child()
    
            else:
                p_node=p_node.right_child()
        index=p_node.threshold
        print(index)
        fake_index_vector=[]
        
        for i in range(mo.leaf_node_num):
            if i==p_node.threshold():
                fake_index_vector.append(mo.ociph)
                #fake_index_vector.append(csu.upk.encrypt(1))
            else:
                fake_index_vector.append(mo.zciph)
                #fake_index_vector.append(csu.upk.encrypt(0))
        
        return fake_index_vector
    
