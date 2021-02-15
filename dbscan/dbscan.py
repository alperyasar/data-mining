# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:28:00 2020

@author: alper
"""

import numpy

# D: adata set containing n objects
# ϵ(eps): the radius parameter
#Minpts: the neighborhood density threshold 

def MyDBSCAN(D, eps, MinPts):

    # make all objects as unvisited
    points = [0]*len(D)

    # C is the ID of the current cluster.    
    C = 0
    
    # select point
    for P in range(0, len(D)):
    
        # If the point's label is not 0, it's mean visited, 
        # if visisted continue to the next point.
        if not (points[P] == 0):
           continue
        
        
        # check how many has neighbor points
        NeighborPoits = neighborhood(D, P, eps)
        
        # If the number is below MinPts, this point is noise. 
        if len(NeighborPoits) < MinPts:
            points[P] = -1
        #if the ϵ-neighborhood of p hasat least MinPts objects   
        else: 
            # create new C, and add p to C
            C += 1
            cluster(D, points, P, NeighborPoits, C, eps, MinPts)
    
    # All data has been clustered!
    return points


def cluster(D, points, P, NeighborPoits, C, eps, MinPts):

    # Assign the cluster label to the seed point.
    points[P] = C
    

    i = 0
    while i < len(NeighborPoits):    
        
        # Get the next point from the queue.        
        Pn = NeighborPoits[i]
       
        if points[Pn] == -1:
           points[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif points[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            points[Pn] = C
            
            # Find all the neighbors of Pn
            PnNeighborPts = neighborhood(D, Pn, eps)
            
            # If Pn has at least MinPts neighbors, it's a branch point! 
            if len(PnNeighborPts) >= MinPts:
                NeighborPoits = NeighborPoits + PnNeighborPts

        i += 1        
    
    # We've finished growing cluster C!


def neighborhood(D, P, eps):
    
    # Find all points in dataset `D` within distance `eps` of point `P`.

    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors