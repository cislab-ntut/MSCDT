from preprocessing import transform4
from structure2 import Node


## leaf node attribute = -1, threshold = classifier result
def build_node(skl_tree_attri_arr, skl_tree_threshold_arr, skl_tree_classifier_arr, idx, attri_list):
    newNode = None
    idx_ = None
    if skl_tree_threshold_arr[idx] != -20:
        #print(skl_tree_threshold_arr[idx])
        left, idx_ = build_node(skl_tree_attri_arr, skl_tree_threshold_arr, skl_tree_classifier_arr, idx+1, attri_list)
        right, idx_ = build_node(skl_tree_attri_arr, skl_tree_threshold_arr, skl_tree_classifier_arr, idx_+1, attri_list)
        newNode = Node(attribute=skl_tree_attri_arr[idx], threshold=skl_tree_threshold_arr[idx], left_child=left, right_child=right, is_leaf_node=False)
        return newNode, idx_
    else:
        # print(skl_tree_classifier_arr[idx])
        #print(skl_tree_classifier_arr[idx][0])
        #print([i for i, e in enumerate(skl_tree_classifier_arr[idx][0]) if e != 0])
        #print("leaf attri: ", attri_list[[i for i, e in enumerate(skl_tree_classifier_arr[idx][0]) if e != 0][0]])
        max=0
        index=0
        for i, e in enumerate(skl_tree_classifier_arr[idx][0]) :
            if e != 0:
                if e>max:
                    index=i
                    max=e
        newNode = Node(attribute=-1, 
                       threshold=attri_list[[index][0]], 
                       is_leaf_node=True)
        #print("leaf: ",newNode.threshold())
        return newNode, idx

def transform(skl_tree_attri_arr, skl_tree_threshold_arr, skl_tree_classifier_arr, attri_list):
    ##tree = [] ## 
    root, _ = build_node(skl_tree_attri_arr, skl_tree_threshold_arr, skl_tree_classifier_arr, 0, attri_list)
    return root