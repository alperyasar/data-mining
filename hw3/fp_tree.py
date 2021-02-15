# -*- coding: utf-8 -*-
"""

@author: alper
"""

from node import Node


def find_frequent_itemset(item, minsup, D):
    '''
    Find 1-itemset menu
     : param item: Tap the item
     : param minsup: Source of support
     : param D: number data
     : return: Tap menu 1-itemset, dictionary N
    '''
    itemset = []
    for i in item:
        count = 0
        for _,values in D.items():
            if i in values:
                count += 1
        if count >= minsup:
            itemset.append((i,count))
    return itemset


# Sort the itemset
def sort(itemset):
    size = len(itemset)
    for i in range(0,size - 1):
        for j in range(i + 1, size):
            if itemset[i][1] < itemset[j][1]:
                temp = itemset[i]
                itemset[i] = itemset[j]
                itemset[j] = temp
    return itemset


# Raise with the head
def create_HeaderTable(itemset):
    header_table = []
    for i in itemset:
        temp = Node(i[0],i[1],None,None)
        header_table.append(temp)
    return header_table


# Create an FP tree
def frequent_item(itemset):
    # The function to find the frequent list series in the itemset
    frequent = []
    for i in itemset:
        frequent.append(i[0])
        print("%6s %15d"% (str(i[0]), int(i[1])))
    return frequent

def insert_HeadTable(root:Node, node:Node):
    if root.root is None:
        root.root = node
    else:
        temp = root.root # Di toi node tiep theo
        insert_HeadTable(temp,node)
def insert_Tree(root:Node,header_table:list,frequent_trans:list):
    '''
    Insert the tree at the root node
    :param root: root
    :param header_table: head of table
    :param frequent_trans: Summary of transaction, there is a list of tuple ('id', count)
    :return:
    '''
    if len(frequent_trans) == 0: # If the frequent_trans string does not count how much content
        return
    first_item = frequent_trans[0][0]
    remaining_item = frequent_trans[1:]
    if root.isChild(first_item) == False: # If first_item is not a child of root
        newNode = Node(first_item,frequent_trans[0][1],root,None) # Create a new node
        for i in header_table: # Added by head at element i
            if newNode == i:
                insert_HeadTable(i,newNode)
                break
        root.append(newNode)
        temp = newNode
        insert_Tree(temp,header_table,remaining_item)
    else: # If first_item is a child of root
        root.increase(first_item,frequent_trans[0][1])
        temp = root.getNode(first_item)
        insert_Tree(temp,header_table,remaining_item)
def create_FPTree(D, frequent_list, header_table, root):
    # Find frequent items in transactions and be ordered as frequent_string
    for _,value in D.items(): # Search each transaction
        frequent_trans = []
        for i in frequent_list:
            if i in value:
                frequent_trans.append((i,1))
        insert_Tree(root,header_table,frequent_trans)
    return root

