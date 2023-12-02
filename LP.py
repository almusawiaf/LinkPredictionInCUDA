# local and global LP models

import numpy as np
import math
import networkx as nx
from random import sample

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import train_test_split

from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

def getU(lst):
    return list(combinations(lst, 2))

def get_AUC(true_labels, predicted_scores):
    return roc_auc_score(true_labels, predicted_scores)

def get_AUPRC(true_labels, predicted_scores):
    return average_precision_score(true_labels, predicted_scores)

def get_ACC(true_labels, predicted_scores, threshold=0.5):
    predicted_labels = [1 if score >= threshold else 0 for score in predicted_scores]
    return accuracy_score(true_labels, predicted_labels)

#-------------------------------------------------------------------------
def CN(G, x, y, Nx= None, Ny=None):
    'common neighbor'
    ax = set(x)
    ay = set(y)
    return  abs(len(ax.intersection(ay)))

#-------------------------------------------------------------------------
def AA(G,x,y, Nx= None, Ny=None):
    'Adamic-Adar index'
    ax = set(x)
    ay = set(y)
    az = ax.intersection(ay)
    sum = 0
    for z in az:
        L = math.log(len(list(G.neighbors(z))))
        # print (L)
        if L != 0 :
            sum = sum + (1/L)
    return sum 

#-------------------------------------------------------------------------
def RA(G, x, y, Nx= None, Ny=None):
    ax = set(x)
    ay = set(y)
    sum = 0 
    for z in (ax.intersection(ay)):
        sum = sum + abs(1/len(list(G.neighbors(z))))
    return sum 

    
#-------------------------------------------------------------------------
def PA(G, x, y, Nx= None, Ny=None):
    'Preferential Attachment'
    ax = set(x)
    ay = set(y)
    return  len(ax)*len(ay)

#-------------------------------------------------------------------------
def JA(G, x, y, Nx= None, Ny=None):
    'Jaccard Index'
    ax = set(x)
    ay = set(y)
    return  len(ax.intersection(ay))/len(ax.union(ay))

#-------------------------------------------------------------------------
def SA(G, x, y, Nx= None, Ny=None):
    'Salton Index'
    ax = set(x)
    ay = set(y)
    return  len(ax.intersection(ay))/math.sqrt(len(ax)*len(ay))


#-------------------------------------------------------------------------
def SO(G, x, y, Nx= None, Ny=None):
    'Sorensen Index'
    ax = set(x)
    ay = set(y)
    return  2* len(ax.intersection(ay))/(len(ax)+len(ay))

#-------------------------------------------------------------------------
def HPI(G, x, y, Nx= None, Ny=None):
    'Hub Pronoted Index'
    ax = set(x)
    ay = set(y)
    return  len(ax.intersection(ay))/min(len(ax), len(ay))

#-------------------------------------------------------------------------
def HDI(G, x, y, Nx= None, Ny=None):
    'Hub Depressed Index'
    ax = set(x)
    ay = set(y)
    return  len(ax.intersection(ay))/max(len(ax), len(ay))

#-------------------------------------------------------------------------
def LLHN(G, x, y, Nx= None, Ny=None):
    'Local Leicht-Homle-Newman Index'
    ax = set(x)
    ay = set(y)
    return  len(ax.intersection(ay))/len(ax)*len(ay)

#-------------------------------------------------------------------------
def CAR(G, x, y, Nx=None, Ny=None):
    ax = set(Nx)
    ay = set(Ny)
    sum = 0 
    for z in (ax.intersection(ay)):
        az = G.neighbors(z)
        if len(list(az)) != 0:
            dom = len(ax.intersection(ay.intersection(set(G.neighbors(z)))))
            nom = len(list(G.neighbors(z)))
            sum = sum + (dom/nom)
    return sum


#-------------------------------------------------------------------------
def CH2_L2(G, x, y, Nx=None, Ny=None):
    """Nx and Ny: neighbor of x, y"""
    # start = time.time()
    ai = set(Nx)
    aj = set(Ny)
    S = 0
    for x in (ai.intersection(aj)):
        c_x = len(set(G.neighbors(x)).intersection(ai.intersection(aj)))
        A = set(G.neighbors(x))
        B = set(ai.intersection(aj))
        ij= set([x,y])
        o_x = len(A-B-ij)
        S += (1 + c_x)/(1 + o_x)
    # end = time.time()
    # print(f'L2 = {end-start}')
    if S== None:
        return 0
    else:
        return S
    
#-------------------------------------------------------------------------
def CH2_L3(G, x, y, Nx=None, Ny=None):
    """Nx and Ny: neighbor of x, y"""
    # start = time.time()
    ai = set(Nx)
    aj = set(Ny)
    S = 0
    con = [i for i in list(nx.all_simple_paths(G, x, y, cutoff=3)) if len(i)==4]
    connects = set()
    for i in con:
        connects.union(set(i))
    
    if not G.has_edge(x, y):
        for i in ai:
            for j in aj:
                if G.has_edge(i,j):
                    Ci = len(set(G.neighbors(i)).intersection(connects))
                    Cj = len(set(G.neighbors(j)).intersection(connects))
                    
                    Oi = len(set(G.neighbors(i)).intersection(G.nodes - connects))
                    Oj = len(set(G.neighbors(j)).intersection(G.nodes - connects))
                    
                    S+= math.sqrt((1+Ci)*(1+Cj))/math.sqrt((1+Oi)*(1+Oj))
    # end = time.time()
    # print(f'L3 = {end-start}')
    if S== None:
        return 0
    else:
        return S


#-------------------------------------------------------------------------
def sampling(X, Total):
    return round(X*Total)

#-------------------------------------------------------------------------
def info(G):
    return f'|V|\t=\t{len(G.nodes())}\t|E|\t=\t{len(G.edges())}'

#-------------------------------------------------------------------------
def divide_randomly(perc, G):
    """G: Graph
    gT: training graph
    gP: Probe Graph"""
    E = list(G.edges())
    
    pT = sampling(perc, len(E))
    ET = sample(E, pT)
    EP = list(G.edges() - list(ET))
    # print("E = {}\t eT = {} \teP = {}".format(len(E), len(ET), len(EP)))
    gT, gP = nx.Graph(), nx.Graph()
    gT.add_nodes_from(list(G.nodes()))
    gP.add_nodes_from(list(G.nodes()))
    gT.add_edges_from(ET)    
    gP.add_edges_from(EP)
    # print('\nG = ',  info(G))
    # print('gT = ', info(gT))
    # print('gP = ', info(gP))

    return gT, gP

#-------------------------------------------------------------------------
def LP_exmperiment(G, perc, ev):
    """G: Graph, 
    K: number of simulation, 
    perc: sampling percentage (0.2)
    ev: evaluation model 0: AUC, 1 : AUPRC"""
    
    # ------------------------------------------------
    gT, gP = divide_randomly(perc, G)
    # ------------------------------------------------
    real_y = []
    S_E = []     
    Nodes = sorted(list(G.nodes()))
    for i in range(len(Nodes)-1):
        for j in range(i+1, len(Nodes)):
            v, u = Nodes[i], Nodes[j]
            if (u,v) in gT.edges() or u==v:
                continue
            Nv, Nu = list(gT.neighbors(v)), list(gT.neighbors(u))
            Sxy_CAR =    CAR(gT, v, u, Nv, Nu)
            Sxy_L3  = CH2_L3(gT, v, u, Nv, Nu)

            S_E.append([Sxy_CAR, Sxy_L3])
            if gP.has_edge(v, u):
                real_y.append(1)
            else:
                real_y.append(0) 
    newSxy = np.array(S_E)
    if ev==0:
        results1 = get_AUC(real_y, newSxy)
    else:
        results1 = get_AUPRC(real_y, newSxy)
    print(results1)
    print('\n----------------------------------------------------------')
    return results1


# #-------------------------------------------------------------------------
# def get_AUC(real_y, newSxy):
#     # print('real y = {}\newSxy = {}\n{}\n{}'.format(real_y, newSxy, len(real_y), len(newSxy)))
#     results = []
#     for c in newSxy.T:
#         try:
#             rr = roc_auc_score(real_y, c)
#         except ValueError:
#             rr = 0
#         results.append(rr)
#     return results

# #-------------------------------------------------------------------------
# def get_AUPRC(real_y, newSxy):
#     # URL = "https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/#:~:text=The%20AUPRC%20is%20calculated%20as,make%20use%20of%20the%20TPR."
#     return [average_precision_score(real_y, np.array(c)) for c in newSxy.T]
#     # for c in newSxy.T:
#     #     try:
#     #         rr = average_precision_score(real_y, np.array(c))
#     #     except ValueError:
#     #         rr = 0
#     #     results.append(rr)
#     # return results

#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
