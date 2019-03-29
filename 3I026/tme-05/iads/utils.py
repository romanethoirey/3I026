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

# importation de LabeledSet
from . import LabeledSet as ls

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
