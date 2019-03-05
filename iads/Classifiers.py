# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        
        raise NotImplementedError("Please Implement this method")

        
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        res = 0
        for i in range(dataset.size()):
            if(self.predict(dataset.getX(i))==dataset.getY(i)):
                res+=1
        return res/dataset.size()
    
    def getInputDimension(self):
        """ Renvoie la dimension de l'espace d'entrée
        """
        return self.input_dimension
    

# ---------------------------
class ClassifierRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
   
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w = np.random.randn(input_dimension)
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        score = np.dot(x,self.w)
        if(score>0):
            return 1
        else:
            return -1
        
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        print("Pas d'apprentissage pour ce classifieur")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        res = 0
        for i in range(dataset.size()):
            if(self.predict(dataset.getX(i))==dataset.getY(i)):
                res+=1
        return res/dataset.size()
    
    def getInputDimension(self):
        """ Renvoie la dimension de l'espace d'entrée
        """
        return self.input_dimension
    
# ---------------------------
class ClassifierKNN(Classifier):
    
   
    def __init__(self, input_dimension,k):
        """ Constructeur de ClassifierKNN
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        b = self.labeledSet
        tab = []
        for i in range(b.size()):
            distance = ((b.getX(i)[0]-x[0])**2)+((b.getX(i)[1]-x[1])**2)
            tab.append(distance)
        ind = np.argsort(tab)
        
        labelpos = 0
        labelneg = 0
        for i in range(self.k):
            if(b.getY(ind[i]) == 1):
                labelpos+=1
            else:
                labelneg+=1
        if(labelpos>labelneg):
            return 1
        else:
            return -1
        
        
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet = labeledSet
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        res = 0
        for i in range(dataset.size()):
            if(self.predict(dataset.getX(i))==dataset.getY(i)):
                res+=1
        return res/dataset.size()
    
    def getInputDimension(self):
        """ Renvoie la dimension de l'espace d'entrée
        """
        return self.input_dimension
    
# ---------------------------

class ClassifierPerceptronRandom(Classifier):
    def __init__(self, input_dimension):
        """ Argument:
                - input_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z
        
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("No training needed")

# ---------------------------

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v

   

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        ##TODO
        score = np.dot(x,self.w)
        if(score>0):
            return 1
        else:
            return -1
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ##TODO
        r = list(range(labeledSet.size()))
        random.shuffle(r)
        for i in r:
            self.w = self.w + self.learning_rate*labeledSet.getX(i)*labeledSet.getY(i)
        
 # ---------------------------   

class ClassifierPerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.dimension_kernel = dimension_kernel
        self.learning_rate = learning_rate
        self.kernel = kernel
        v = np.random.rand(dimension_kernel)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v
   
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        score = np.dot(self.w,self.kernel.transform(x))
        if(score>0):
            return 1
        else:
            return -1

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ##TODO
        r = list(range(labeledSet.size()))
        random.shuffle(r)
        for i in r:
            self.w += self.learning_rate *self.kernel.transform(labeledSet.getX(i))*(labeledSet.getY(i)-self.predict(labeledSet.getX(i)))
  
  
