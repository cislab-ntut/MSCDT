
import time

class Node():
    def __init__(self, attribute=None, threshold=None, left_child=None, right_child=None, is_leaf_node=False):
        #self._data = data
        self._attribute = attribute
        self._polyval = 0
        self.id=0
        self._thre = threshold
        self._lc = left_child
        self._rc = right_child
        self._is_leaf_node = is_leaf_node

    def threshold(self):
        return self._thre

    def attribute(self):
        return self._attribute

    def pval(self):
        return self._polyval
     
    def is_leaf_node(self):
        return self._is_leaf_node

    def left_child(self):
        return self._lc

    def right_child(self):
        return self._rc
    def set_left_child(self,node):
        self._lc=node
        
    def set_right_child(self,node):
        self._rc=node
    
    def set_attribute(self, attri):
        self._attribute = attri

    def set_threshold(self, val):
        self._thre = val
    def set_polyval(self, val):
        self._polyval=val

class Timer():
    def __init__(self, detail=None):
        self.start_time = time.time()
        self.detail = detail

    def reset(self, detail=None):
        self.start_time = time.time()
        self.detail = detail

    def end(self, detail=None):
        if detail is not None:
            self.detail = detail
        interval = (time.time() - self.start_time) * 1000
        #print(f"{self.detail}共花費 {interval:.4f}豪秒")
        return interval
