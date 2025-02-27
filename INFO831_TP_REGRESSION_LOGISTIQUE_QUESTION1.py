# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:25:02 2025

@author: atto

IDU_INFO831 TP REGRESSION LOGISTIQUE (LOGIT) - QUESTION 1
"""


import os
import numpy as np
import cv2  # OpenCV pour la lecture des images
import pandas as pd  # Pour sauvegarder en CSV
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt

img_height = 299
img_width = 299
batch_size = 32
classes = 2



#%% Code de vectorisation avancee des images en utilisant xception
def Importer_Et_Vectoriser_Images(folder_path):
    image_vectors = []
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_height, img_width, 3)
    )  
    # Mode inference : pas de training (sert uniquement en tant que vectoriseur avancé)
    base_model.trainable = False
    base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    #assert base_model.layers[-1].activation == tf.keras.activations.softmax
    # base_model.summary()
    
    # Couches additionnelles d'entrees pour la normalisation (requise pour consistence des entrees xception)
    inputs = keras.Input(shape=(img_height, img_width, 3)) # x est une entree de valeurs dans [0, 255] 
    # normalisation suivante selon y = (x - moyenne) / sqrt(variance)
    moyenne = np.array([127.5] * 3)
    var = moyenne ** 2
    norm_layer = keras.layers.Normalization(mean=moyenne, variance=var)
    x = norm_layer(inputs)  # x est maintenant dans  [-1., +1.]
    temp_model = base_model(x, training=False) # rajout de la couche de normalisation
    final_model = keras.Model(inputs, temp_model)  # rajout de la couche des vraies entrees non-normalisees
    #final_model.summary()
    
    # Exemple de calcul des sorties sur une entree donnee x aleatoire
    img = np.random.randint(0,10,(img_height,img_width,3))
    # img.shape # taille classique d'une seule image 
    img = np.expand_dims(img, axis=0) # on rajoute son index dans le batch (xception attend un batch d'images par défaut)
    img.shape
           
    u=final_model.predict(img)  # u est array de floats dont la dimension 1 correspond au batch
    y = u[0].flatten()  # conversion de u en tableau de valeurs numeriques (suppression de l'information de batch)
    y.shape
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            # Charger l'image 
            image = cv2.imread(file_path)
            image = cv2.resize(image, (img_height, img_width)) 
            
            if image is not None:
                # Vectoriser l'image (aplatir en 1D) par une transformation xception
                image = np.expand_dims(image, axis=0) # autre solution de faire un seul ou plusieurs batchs (non ici, par soucis de comprehensibilite)
                u=final_model.predict(image)  # u est array de floats dont la dimension 1 correspond au batch
                image_vector = u[0].flatten()  # conversion de u en tableau de valeurs numeriques (suppression de l'information de batch)
                # image_vector = image.flatten()
                image_vectors.append(image_vector)
    
    # Convertir en matrice NumPy
    if image_vectors:
        image_matrix = np.vstack(image_vectors)
        return image_matrix
    else:
        return None


#%% Chargement et organisation des donnees et definition des labels
#  
#
### MAIN (DEBUT) ### 
# X_train = X1 union X2 : Remplacer par le chemin reel
train_data_dir1 = "CAPTCHA_IDU/1_Train/__I"
"""X1 = Importer_Et_Vectoriser_Images(train_data_dir1)
if X1 is not None:
    print(f"Matrice finale de taille: {X1.shape}")
    N_X1 = X1.shape[0]
    y1 = np.ones((N_X1, 1))  # labels
else:
    print("Aucune image valide trouvée dans le dossier.")"""
train_data_dir2 = "CAPTCHA_IDU/1_Train/_DU"
"""X2 = Importer_Et_Vectoriser_Images(train_data_dir2)
if X2 is not None:
    print(f"Matrice finale de taille: {X2.shape}")
    N_X2 = X2.shape[0]
    y2 = np.zeros((N_X2, 1))  # labels
else:
    print("Aucune image valide trouvée dans le dossier.")
N_train = N_X1 + N_X2
X_train = np.concatenate((X1, X2), axis=0)
y_train = np.concatenate((y1, y2), axis=0)"""

# Remplacer par le chemin reel
test_data_dir1 = "CAPTCHA_IDU/2_Test/__I"
"""X1 = Importer_Et_Vectoriser_Images(test_data_dir1)
if X1 is not None:
    print(f"Matrice finale de taille: {X1.shape}")
    N_X1 = X1.shape[0]
    y1 = np.ones((N_X1, 1))  # labels
else:
    print("Aucune image valide trouvée dans le dossier.")"""
test_data_dir2 = "CAPTCHA_IDU/2_Test/_DU"
"""X2 = Importer_Et_Vectoriser_Images(test_data_dir2)
if X2 is not None:
    print(f"Matrice finale de taille: {X2.shape}")
    N_X2 = X2.shape[0]
    y2 = np.zeros((N_X2, 1))  # labels
else:
    print("Aucune image valide trouvée dans le dossier.")
N_test = N_X1 + N_X2
X_test = np.concatenate((X1, X2), axis=0)
y_test = np.concatenate((y1, y2), axis=0)"""


"""# Transposees
X_train = np.transpose(X_train)
y_train = np.transpose(y_train)
X_test = np.transpose(X_test)
y_test = np.transpose(y_test)"""

"""# Sauvegarde en fichiers CSV
np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")

print("Les données ont été sauvegardées en CSV avec succès.")

# Nombre de caracteristiques extraites par la transformation xception (etape de vectorisation)
nb_features = X_train.shape[0]""" 



#%% Les noms des couches de xception (au cas où vous souhaiteriez changer la procedure de vectorisation)
# X= ['input_layer_13', 'block1_conv1', 'block1_conv1_bn', 'block1_conv1_act', 'block1_conv2', 'block1_conv2_bn', 'block1_conv2_act', 'block2_sepconv1', 'block2_sepconv1_bn', 'block2_sepconv2_act', 'block2_sepconv2', 'block2_sepconv2_bn', 'conv2d_36', 'block2_pool', 'batch_normalization_36', 'add_108', 'block3_sepconv1_act', 'block3_sepconv1', 'block3_sepconv1_bn', 'block3_sepconv2_act', 'block3_sepconv2', 'block3_sepconv2_bn', 'conv2d_37', 'block3_pool', 'batch_normalization_37', 'add_109', 'block4_sepconv1_act', 'block4_sepconv1', 'block4_sepconv1_bn', 'block4_sepconv2_act', 'block4_sepconv2', 'block4_sepconv2_bn', 'conv2d_38', 'block4_pool', 'batch_normalization_38', 'add_110', 'block5_sepconv1_act', 'block5_sepconv1', 'block5_sepconv1_bn', 'block5_sepconv2_act', 'block5_sepconv2', 'block5_sepconv2_bn', 'block5_sepconv3_act', 'block5_sepconv3', 'block5_sepconv3_bn', 'add_111', 'block6_sepconv1_act', 'block6_sepconv1', 'block6_sepconv1_bn', 'block6_sepconv2_act', 'block6_sepconv2', 'block6_sepconv2_bn', 'block6_sepconv3_act', 'block6_sepconv3', 'block6_sepconv3_bn', 'add_112', 'block7_sepconv1_act', 'block7_sepconv1', 'block7_sepconv1_bn', 'block7_sepconv2_act', 'block7_sepconv2', 'block7_sepconv2_bn', 'block7_sepconv3_act', 'block7_sepconv3', 'block7_sepconv3_bn', 'add_113', 'block8_sepconv1_act', 'block8_sepconv1', 'block8_sepconv1_bn', 'block8_sepconv2_act', 'block8_sepconv2', 'block8_sepconv2_bn', 'block8_sepconv3_act', 'block8_sepconv3', 'block8_sepconv3_bn', 'add_114', 'block9_sepconv1_act', 'block9_sepconv1', 'block9_sepconv1_bn', 'block9_sepconv2_act', 'block9_sepconv2', 'block9_sepconv2_bn', 'block9_sepconv3_act', 'block9_sepconv3', 'block9_sepconv3_bn', 'add_115', 'block10_sepconv1_act', 'block10_sepconv1', 'block10_sepconv1_bn', 'block10_sepconv2_act', 'block10_sepconv2', 'block10_sepconv2_bn', 'block10_sepconv3_act', 'block10_sepconv3', 'block10_sepconv3_bn', 'add_116', 'block11_sepconv1_act', 'block11_sepconv1', 'block11_sepconv1_bn', 'block11_sepconv2_act', 'block11_sepconv2', 'block11_sepconv2_bn', 'block11_sepconv3_act', 'block11_sepconv3', 'block11_sepconv3_bn', 'add_117', 'block12_sepconv1_act', 'block12_sepconv1', 'block12_sepconv1_bn', 'block12_sepconv2_act', 'block12_sepconv2', 'block12_sepconv2_bn', 'block12_sepconv3_act', 'block12_sepconv3', 'block12_sepconv3_bn', 'add_118', 'block13_sepconv1_act', 'block13_sepconv1', 'block13_sepconv1_bn', 'block13_sepconv2_act', 'block13_sepconv2', 'block13_sepconv2_bn', 'conv2d_39', 'block13_pool', 'batch_normalization_39', 'add_119', 'block14_sepconv1', 'block14_sepconv1_bn', 'block14_sepconv1_act', 'block14_sepconv2', 'block14_sepconv2_bn', 'block14_sepconv2_act', 'avg_pool', 'predictions']