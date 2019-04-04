# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(df):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    normalised_data = (df - df.min() )/(df.max() - df.min());
    return normalised_data

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.linalg.norm(v1-v2)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(df):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return pd.DataFrame(df.mean()).T

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    centr = centroide(df)
    somme = 0
    for i in range(len(df)):
        somme += dist_vect(df.iloc[i],centr)*dist_vect(df.iloc[i],centr)
    return somme


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,df):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    return df.sample(K)


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(Exe,Centres):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    min = 3
    ind = 0
    for i in range(len(Centres)):
        dist = dist_vect(Exe, Centres.iloc[i])
        if(dist < min):
            min = dist
            ind = i
    return ind

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(Base,Centres):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    dictionnaire = {}
    for i in range(len(Centres)):
        dictionnaire[i]=[]
    for i in range(len(Base)):
        cluster_curr = plus_proche(Base.iloc[i], Centres)
        dictionnaire[cluster_curr].append(i)
    return dictionnaire

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(Base,U):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    new_df = pd.DataFrame()
    centr = pd.DataFrame()
    
    for i in U.keys():
        for j in range(len(U[i])):
            new_df = new_df.append(Base.iloc[U[i][j]])
            
        centr = centr.append(centroide(new_df), ignore_index=True)
        new_df = pd.DataFrame()
    return centr

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(Base, U):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    inertie_tot = 0
    
    new_df = pd.DataFrame()
    centr = pd.DataFrame()
    
    for i in U.keys():
        for j in range(len(U[i])):
            new_df = new_df.append(Base.iloc[U[i][j]])
        inertie_tot += inertie_cluster(new_df)
        new_df = pd.DataFrame()
    return inertie_tot
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(K, Base, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    nouveaux = initialisation(K,Base)
    affecte = affecte_cluster(Base,nouveaux)
    inertie = inertie_globale(Base,affecte)
    inertie_prec = inertie + epsilon + 2
    i = 0
    
    while(((inertie_prec - inertie) > epsilon) or (i<iter_max)):
        affecte = affecte_cluster(Base,nouveaux)
        nouveaux = nouveaux_centroides(Base, affecte)
        inertie_prec = inertie
        inertie = inertie_globale(Base,affecte)
        i+=1
        #print("iteration", i, "Inertie :",inertie,"Difference: ",(inertie - inertie_prec),")")
    return (nouveaux, affecte)       
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    label = Base.columns.values
    x = label[0]
    y = label[1]
    for i in Affect.values():
        tmp = Base.iloc[i]
        c = np.random.rand(3)
        plt.scatter(tmp[x],tmp[y],color=c)
    plt.scatter(Centres['X'],Centres['Y'],color='r',marker='x')
# -------
