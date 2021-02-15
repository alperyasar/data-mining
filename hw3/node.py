# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:25:05 2020

@author: alper
"""

# Lop Node
class Node:
    def __init__(self,key,value = 1,parent = None,root = None):
        self.key = key
        self.value = value
        self.parent = parent
        self.root = root
        self.Node_list = []
    def append(self,newNode):
        self.Node_list.append(newNode)
    def item(self):
        return (self.key,self.value)
    def __str__(self):
        rep = self.key + ':' + str(self.value) + '\n'
        if self.parent is not None:
            rep += '\tParent: ' + self.parent.key + str(self.parent.value) + '\n'
        else:
            rep += '\tParent: None\n'
        if self.root is not None:
            rep += '\troot: ' + self.root.key + str(self.root.value) + '\n'
        else:
            rep += '\troot: None\n'
        rep += '\t' + "Node list: "
        for i in self.Node_list:
            item = i.item()
            rep += '(' + item[0] + ':' + str(item[1]) + ')' + ','
        rep = rep[:-1]
        return rep
    def __eq__(self, other):
        if self.key == other.key:
            return True
        return False
    def isChild(self, item):
        '''Check whether the string of items appears in Node_list 
            or check the child node'''
        for i in self.Node_list:
            if item == i.key:
                return True
        return False
    def getNode(self,item):
        for i in self.Node_list:
            if item == i.key:
                return i
        return None
    def increase(self,item,increment):
        '''Increase value by 1 unit'''
        for i in self.Node_list:
            if item == i.key:
                i.value += increment