# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de 3i026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# importation de LabeledSet
from . import LabeledSet as ls

from iads import Classifiers as cl

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1

def plot_frontiere(set,classifier,step=10):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])
    
# ------------------------ 

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ 
        rend un LabeledSet 2D généré aléatoirement.
        Arguments:
        - positive_center (vecteur taille 2): centre de la gaussienne des points positifs
        - positive_sigma (matrice 2*2): variance de la gaussienne des points positifs
        - negative_center (vecteur taille 2): centre de la gaussienne des points négative
        - negative_sigma (matrice 2*2): variance de la gaussienne des points négative
        - nb_points (int):  nombre de points de chaque classe à générer
    """
    une_base = ls.LabeledSet(2)
    for i in range(nb_points):
        une_base.addExample(np.random.multivariate_normal(positive_center,positive_sigma),1)
        une_base.addExample(np.random.multivariate_normal(negative_center,negative_sigma),-1)
    return une_base
    
# ------------------------ 

def createXOR(nb_points,var):
    data1 = createGaussianDataset(np.array([0,0]),np.array([[var,0],[0,var]]),np.array([1,0]),np.array([[var,0],[0,var]]),nb_points)
    data2 = createGaussianDataset(np.array([1,1]),np.array([[var,0],[0,var]]),np.array([0,1]),np.array([[var,0],[0,var]]),nb_points)
    for i in range(data1.size()):
        data2.addExample(data1.getX(i),data1.getY(i))
        
    return data2

# ------------------------

def split(labeledSet,p):
    dim = labeledSet.getInputDimension()
    base_train = ls.LabeledSet(dim)
    base_test = ls.LabeledSet(dim)
    
    indice = list(range(labeledSet.size()))
    random.shuffle(indice)
    
    p_train = (int)(labeledSet.size()*p)
    p_test = (int)(labeledSet.size()*(1-p))
    for i in range(p_train):
        base_train.addExample(labeledSet.getX(indice[i]), labeledSet.getY(indice[i]))
        
            
    for i in range(p_train, p_test+p_train):
        base_test.addExample(labeledSet.getX(indice[i]), labeledSet.getY(indice[i])) 
        
    return (base_train,base_test)

# ------------------------

def classe_majoritaire(ls):
    pos = 0
    neg = 0
    for i in range(ls.size()):
        if(ls.getY(i)==1):
            pos +=1
        elif(ls.getY(i)==-1):
            neg +=1
    if(neg > pos):
        return -1
    else:
        return 1


# ------------------------

def entropie(ls):
    pos = 0
    neg = 0
    for i in range(ls.size()):
        if(ls.getY(i)==1):
            pos +=1
        elif(ls.getY(i)==-1):
            neg +=1
    pneg = neg/ls.size()
    ppos = pos/ls.size()
    return shannon([ppos,pneg])


# ------------------------

def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)


# ------------------------

def divise(LSet,att,seuil):
    """ LSet: LabeledSet
        att : numéro d'attribut
        seuil : valeur de seuil
        rend le tuple contenant les 2 sous-LabeledSet obtenus par la
        division de LSet selon le seuil sur l'attribut att
    """
    Linf = ls.LabeledSet(LSet.getInputDimension())
    Lsup = ls.LabeledSet(LSet.getInputDimension())
    
    for i in range(LSet.size()):
        if(seuil > LSet.getX(i)[att]):
            Linf.addExample(LSet.getX(i),LSet.getY(i))
        else:
            Lsup.addExample(LSet.getX(i),LSet.getY(i))
    return (Linf,Lsup)        

# ------------------------

def construit_AD(LSet, epsilon):
    """ LSet : LabeledSet
        epsilon : seuil d'entropie pour le critère d'arrêt 
    """
    a = cl.ArbreBinaire()
    e = entropie(LSet)
    min_seuil = 1.1
    min_entropie = 1.1
    att = 0
    if(e < epsilon):
        a.ajoute_feuille(classe_majoritaire(LSet))
    else:
        for i in range(LSet.getInputDimension()):
            
            minseuil_tmp, minentropie_tmp = discretise(LSet,i)
            
            if(minentropie_tmp < min_entropie):
                min_seuil = minseuil_tmp
                min_entropie = minentropie_tmp
                att = i
                
        Lsup, Linf = divise(LSet,att,min_seuil)        
        ABinf = construit_AD(Linf,epsilon)
        ABsup = construit_AD(Lsup,epsilon)
        
        a.ajoute_fils(ABinf,ABsup,att,min_seuil)
    return a
