# -*- coding: utf-8 -*-
"""
@author: alper
"""


from fp_tree import *


# Exploit common set
def discover_path(temp:list, root:Node,value:int):
    if root.key == 'root':
        return
    temp.append((root.key,value))
    parent = root.parent
    discover_path(temp,parent,value)
def Conditional_pattern(header_table:list, beta:str):
    # Conditional Pattern Base
    conditional_PB = []
  
    for i in header_table:
        if i.key == beta:
            alpha = i.root
         
            while alpha is not None:
                temp = []
                discover_path(temp, alpha.parent, alpha.value)
                temp.reverse()
                if len(temp) != 0:
                    conditional_PB.append(temp)
                alpha = alpha.root
            break
    return conditional_PB
def FP_Growth(FP_Tree:Node,header_table:list,final_result,item_list,value_list, beta:str,value, minsup):

    item_templist = item_list.copy() # The list contains items
    value_templist = value_list.copy() # List contains values
    item_templist.append(beta)
    value_templist.append(value)
    final_result.append((item_templist,min(value_templist)))
    conditional_PB = Conditional_pattern(header_table,beta)
    if len(conditional_PB) == 0:
        return


    # frequent item for Conditional_PB
    frequent_item_in_CPB = {}
    for i in conditional_PB: # i is a list of nodes, each leaf node tuple
        for j in i: # Search each node, each leaf node tuple
            if j[0] not in frequent_item_in_CPB:
                frequent_item_in_CPB[j[0]] = j[1]
            else:
                frequent_item_in_CPB[j[0]] += j[1]
    for key in list(frequent_item_in_CPB.keys()):
        if frequent_item_in_CPB[key] < minsup:
            del frequent_item_in_CPB[key]
    sorted_frequent_item = [(k, frequent_item_in_CPB[k]) for k in sorted(frequent_item_in_CPB, key=frequent_item_in_CPB.get, reverse=True)]

    # Create the FP Tree
    
    head_SubTable = []
    root = Node('root', 1, None, None)  # Create the root node
    for i in sorted_frequent_item:
        head_SubTable.append(Node(i[0],i[1]))

    # Create frequent trans in each Conditional_FB
    for i in conditional_PB: # Search each path
        frequent_trans = []
        for j in i: # Search each node of the path
            for k in sorted_frequent_item:
                if j[0] == k[0]:
                    frequent_trans.append(j)
                    break
        insert_Tree(root,head_SubTable,frequent_trans)
    for i in head_SubTable:
        FP_Growth(root,head_SubTable,final_result,item_templist,value_templist,i.key,i.value,minsup)



def Run_FPGrowth(root,minsup,header_table):
    size = len(header_table) - 1
    final = []
    itemlist = []
    valuelist = []
    while size >= 0:
        FP_Growth(root,header_table,final,itemlist,valuelist,header_table[size].key,header_table[size].value,minsup)
        size -= 1
    return final

