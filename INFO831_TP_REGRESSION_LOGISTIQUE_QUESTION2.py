# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:25:02 2025

@author: atto

IDU_INFO831 TP REGRESSION LOGISTIQUE (LOGIT) - QUESTION 2 
"""


import os
import numpy as np
import cv2  # OpenCV pour la lecture des images
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt


 
#%% Fonction Sigmoide
def Sigmoide(z):
    # s = Sigmoide(z)
    #s = 1 / (1 - np.exp(z)) ERREUR
    s = 1 / (1+np.exp(-z))  
    return s
#
# Test de la fonction Sigmoide
print ("Sigmoide([0, 2]) = " + str(Sigmoide(np.array([0,2]))))
#
#%% Fonction d'initialisaion: Initialiser_Les_Parametres
def Initialiser_Les_Parametres(dim):
    # w = vecteur de zeros [shape (dim, 1)] et biais scalaire b = 0.    
    w = np.zeros([dim, 1])
    b = 0.0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))  
    return w, b
#
dim = 2
w, b = Initialiser_Les_Parametres(dim)
print ("w = " + str(w))
print ("b = " + str(b))
#
#%% model de regression logistique
def Modele_De_Regression_Logistique(X, Y, w, b):
    """
    Donnees et labels:
        X: tableau de donnees d'entree de taille (nb_features, nombre d"exemples)
        Y: vecteur de vrais "labels" (1 pour la cible, 0 pour tout autre) de taille (1, nombre d"exemples)
    Parametres du model                                                  
        w: params poids de taille (nb_features, 1)
        b: biais scalaire
    Sorties:
        fonction_cout: cf regression logistique
        dw: gradient de la fonction_cout par rapport à w, meme taille que w
        db: gradient de la fonction_cout par rapport à b, meme taille que b
    """
    
    m = X.shape[1]  # nombre d'exemples
    
    # FORWARD (De X vers la fonction de cout)
    Ye = Sigmoide(np.dot(w.T, X) + b)                # 
    fonction_cout = -np.sum(Y * np.log(Ye) + (1 - Y) * np.log(1 - Ye)) / m     # 
    
    # BACKWARD (Pour le calcul du gradient)
    dw = np.dot(X, (Ye - Y).T) / m    # X.T => transpose de X
    db = np.sum(Ye - Y) / m
    fonction_cout = np.squeeze(fonction_cout)

    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    assert(fonction_cout.shape == ())
    
    grads = {"dw": dw, "db": db}
    
    return grads, fonction_cout
#
# Test calcul de sortie et gradients
w=np.array([[1],[2]])
b=2
X=np.array([[1,2],[3,4]])
Y=np.array([[1,0]])
grads, fonction_cout = Modele_De_Regression_Logistique(X, Y, w, b)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("fonction_cout = " + str(fonction_cout))
#
#%% Optimisation
def Optimisation_Par_Descente_De_Gradient_Logit(w, b, X, Y, nb_iterations, taux_descente, print_fonction_cout = False):
    """
    Descente de gradient de f = f(w,b) 
    nb_iterations -- nombre d'itérations de la boucle d'optimisation
    taux_descente -- learning rate de la règle de mise à jour de la descente de gradient
    print_fonction_cout -- pour imprimer la fonctio de coût tous les 100 itérations
    
    Returns:
    params --dictionnaire contenant les poids w et le biais b
    grads -- dictionnaire contenant les gradients des poids et des biais par rapport à la fonction fonction_cout
    fonction_couts -- liste de toutes les fonction_couts calculées lors de l'optimisation, cela sera utilisé pour tracer la courbe d'Apprentissage_Logit.
    
    Itérations de:
        1) Calculer la fonction_cout et le gradient pour les paramètres courants. Utiliser Modele_De_Regression_Logistique().
        2) Mettre à jour les paramètres à l'aide de la règle de descente de gradient pour w et b.
    """
    
    fonction_couts = []
    
    for i in range(nb_iterations):
        grads, fonction_cout = Modele_De_Regression_Logistique(X, Y, w, b)     
        dw = grads["dw"]
        db = grads["db"]
        #w = w + taux_descente * dw ERREUR
        w = w - taux_descente * dw
        b = b - taux_descente * db
        # Record the fonction_couts
        if i % 100 == 0:
            fonction_couts.append(fonction_cout)
        
        # Print fonction_cout (tous les 100 exemples)
        if print_fonction_cout and i % 100 == 0:
            print ("fonction_cout apres iteration %i: %f" %(i, fonction_cout))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, fonction_couts
#
# Test de la routine d'optim
params, grads, fonction_couts = Optimisation_Par_Descente_De_Gradient_Logit(w, b, X, Y, nb_iterations= 10, taux_descente = 0.009, print_fonction_cout = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
#
#%% Fonction de prediction
def Predire_Par_Seuil_De_Decision_Sigmoide(w, b, X):
    # Sortie: Y_prediction -- a numpy array (vecteur) contanant toutes les predictions (0/1) pour les exemples issus de X    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    # Calcul du vecteur "A" correspondant à une proba d"être de la classe "cible"
    A = Sigmoide(np.dot(w.T, X) + b)   
    for i in range(A.shape[1]):
        # Convertir les proba A[0,i] en predictions p[0,i]
        Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
#
# Test de fonction Predire_Par_Seuil_De_Decision_Sigmoide
print ("predictions = " + str(Predire_Par_Seuil_De_Decision_Sigmoide(w, b, X)))
#
#%%  Merge all functions

def Apprentissage_Logit(X_train, Y_train, X_test, Y_test, nb_iterations = 2, taux_descente = 0.5, print_fonction_cout = False):
    """
    Phase d"Apprentissage
    
    Arguments:
    X_train -- data de training : numpy array de shape (nb_features, N_train)
    Y_train -- labels de training : numpy array (vecteur) de shape (1, N_train)
    X_test -- data de test : numpy array de shape (nb_features, N_test)
    Y_test -- labels de test : numpy array (vecteur) de shape (1, N_test)
    nb_iterations -- hyperparameter représentant le nombre d'itérations pour trouver les paramètres optimaux
    taux_descente -- hyperparameter représentant le taux d'Apprentissage utilisé dans la règle de mise à jour de Optimisation_Par_Descente_De_Gradient_Logit()
    print_fonction_cout -- Mettre à "True" pour afficher "fonction_cout" toutes les 100 iterations
    
    Returns:
    d -- dictionnaire contenant des informations sur la phase : Apprentissage_Logit.
    """
    
    ### APPRENDRE ###
    
    # initialiser les parameters 
    w, b = Initialiser_Les_Parametres(X_train.shape[0])

    # Descente de Gradient
    parameters, grads, fonction_couts = Optimisation_Par_Descente_De_Gradient_Logit(w, b, X_train, Y_train, nb_iterations, taux_descente, print_fonction_cout)
    
    # Extraire les parameters w et b du dictionaire "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predire : test/train sur des examples
    Y_prediction_test = Predire_Par_Seuil_De_Decision_Sigmoide(w, b, X_test)
    Y_prediction_train = Predire_Par_Seuil_De_Decision_Sigmoide(w, b, X_train)

    ### END APPRENDRE ###

    # Afficher train/test performance
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"fonction_couts": fonction_couts,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "taux_descente" : taux_descente,
         "nb_iterations": nb_iterations}
    
    return d
#
#  
#
#%%
#%% Les noms des couches de xception (au cas où vous souhaiteriez changer la procedure de vectorisation)
# X= ['input_layer_13', 'block1_conv1', 'block1_conv1_bn', 'block1_conv1_act', 'block1_conv2', 'block1_conv2_bn', 'block1_conv2_act', 'block2_sepconv1', 'block2_sepconv1_bn', 'block2_sepconv2_act', 'block2_sepconv2', 'block2_sepconv2_bn', 'conv2d_36', 'block2_pool', 'batch_normalization_36', 'add_108', 'block3_sepconv1_act', 'block3_sepconv1', 'block3_sepconv1_bn', 'block3_sepconv2_act', 'block3_sepconv2', 'block3_sepconv2_bn', 'conv2d_37', 'block3_pool', 'batch_normalization_37', 'add_109', 'block4_sepconv1_act', 'block4_sepconv1', 'block4_sepconv1_bn', 'block4_sepconv2_act', 'block4_sepconv2', 'block4_sepconv2_bn', 'conv2d_38', 'block4_pool', 'batch_normalization_38', 'add_110', 'block5_sepconv1_act', 'block5_sepconv1', 'block5_sepconv1_bn', 'block5_sepconv2_act', 'block5_sepconv2', 'block5_sepconv2_bn', 'block5_sepconv3_act', 'block5_sepconv3', 'block5_sepconv3_bn', 'add_111', 'block6_sepconv1_act', 'block6_sepconv1', 'block6_sepconv1_bn', 'block6_sepconv2_act', 'block6_sepconv2', 'block6_sepconv2_bn', 'block6_sepconv3_act', 'block6_sepconv3', 'block6_sepconv3_bn', 'add_112', 'block7_sepconv1_act', 'block7_sepconv1', 'block7_sepconv1_bn', 'block7_sepconv2_act', 'block7_sepconv2', 'block7_sepconv2_bn', 'block7_sepconv3_act', 'block7_sepconv3', 'block7_sepconv3_bn', 'add_113', 'block8_sepconv1_act', 'block8_sepconv1', 'block8_sepconv1_bn', 'block8_sepconv2_act', 'block8_sepconv2', 'block8_sepconv2_bn', 'block8_sepconv3_act', 'block8_sepconv3', 'block8_sepconv3_bn', 'add_114', 'block9_sepconv1_act', 'block9_sepconv1', 'block9_sepconv1_bn', 'block9_sepconv2_act', 'block9_sepconv2', 'block9_sepconv2_bn', 'block9_sepconv3_act', 'block9_sepconv3', 'block9_sepconv3_bn', 'add_115', 'block10_sepconv1_act', 'block10_sepconv1', 'block10_sepconv1_bn', 'block10_sepconv2_act', 'block10_sepconv2', 'block10_sepconv2_bn', 'block10_sepconv3_act', 'block10_sepconv3', 'block10_sepconv3_bn', 'add_116', 'block11_sepconv1_act', 'block11_sepconv1', 'block11_sepconv1_bn', 'block11_sepconv2_act', 'block11_sepconv2', 'block11_sepconv2_bn', 'block11_sepconv3_act', 'block11_sepconv3', 'block11_sepconv3_bn', 'add_117', 'block12_sepconv1_act', 'block12_sepconv1', 'block12_sepconv1_bn', 'block12_sepconv2_act', 'block12_sepconv2', 'block12_sepconv2_bn', 'block12_sepconv3_act', 'block12_sepconv3', 'block12_sepconv3_bn', 'add_118', 'block13_sepconv1_act', 'block13_sepconv1', 'block13_sepconv1_bn', 'block13_sepconv2_act', 'block13_sepconv2', 'block13_sepconv2_bn', 'conv2d_39', 'block13_pool', 'batch_normalization_39', 'add_119', 'block14_sepconv1', 'block14_sepconv1_bn', 'block14_sepconv1_act', 'block14_sepconv2', 'block14_sepconv2_bn', 'block14_sepconv2_act', 'avg_pool', 'predictions']