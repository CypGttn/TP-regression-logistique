# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:25:02 2025

@author: atto

IDU_INFO831 TP REGRESSION LOGISTIQUE (LOGIT) - QUESTION 3
"""
from INFO831_TP_REGRESSION_LOGISTIQUE_QUESTION1 import *
from INFO831_TP_REGRESSION_LOGISTIQUE_QUESTION2 import *

#  

X_train = pd.read_csv("X_train.csv", header=None).values  # Convertir en NumPy
y_train = pd.read_csv("y_train.csv", header=None).values
X_test = pd.read_csv("X_test.csv", header=None).values
y_test = pd.read_csv("y_test.csv", header=None).values
"""
train_data_dir1 = "TP2 Regression logistique/CAPTCHA_IDU/1_Train/__I"
train_data_dir2 = "TP2 Regression logistique/CAPTCHA_IDU/1_Train/_DU"
test_data_dir1 = "TP2 Regression logistique/CAPTCHA_IDU/2_Test/__I"
test_data_dir2 = "TP2 Regression logistique/CAPTCHA_IDU/2_Test/_DU"
img_height = 299
img_width = 299
batch_size = 32
classes = 2
"""

#%%
### MAIN (SUITE) ###

# Premier essai d"Apprentissage
d = Apprentissage_Logit(np.multiply(X_train, X_train), y_train, X_test, y_test, nb_iterations = 1150, taux_descente = 0.19, print_fonction_cout = True)
#
# Lire et verifier la classification d'une des images
index = 10
ListFiles = os.listdir(test_data_dir1)
filename = ListFiles[index][0:]
file_path = os.path.join(test_data_dir1, filename)
image = cv2.imread(file_path)
image = cv2.resize(image, (img_height, img_width)) 
        
classes = y_test     
plt.imshow(image)
print ("y = " + str(y_test[0,index]) + ", le modele predit que c'est \"" + d["Y_prediction_test"][0,index].astype(str))
#
# Plot learning curve (with fonction_couts)
fonction_couts = np.squeeze(d['fonction_couts'])
plt.plot(fonction_couts)
plt.ylabel('fonction_cout')
plt.xlabel('iterations (*100)')
plt.title("Learning rate =" + str(d["taux_descente"]))
plt.show()
#
#
#
#%% Differentes strategies de descentes 
# 
# BEST hyperparameters :
#learning rate is: 0.19
#nb iteration is: 1150
#train accuracy: 79.23333333333333 %
#test accuracy: 70.83333333333333 %

taux_descentes = [0.19]
# nb_iterations = list(range(1000, 2000, 50))
nb_iterations = [1150]
models = {}
for i in taux_descentes:
    for j in nb_iterations:
        print ("learning rate is: " + str(i))
        print ("nb iteration is: " + str(j))
        models[str(i)] = Apprentissage_Logit(X_train, y_train, X_test, y_test, nb_iterations = j, taux_descente = i, print_fonction_cout = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

for i in taux_descentes:
    plt.plot(np.squeeze(models[str(i)]["fonction_couts"]), label= str(models[str(i)]["taux_descente"]))

plt.ylabel('fonction_cout')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
### MAIN (FIN) ### 
#
#%% 
## UTILISATION DU MODELE AU DELA DE LA BASE D"Apprentissage
mypath = "images"   # un dossier quelconque contenant une image non-labelisée
images = Importer_Et_Vectoriser_Images(mypath)
images = images.T
# La premiere image du dossier
image = images[0:]
classe_predite = Predire_Par_Seuil_De_Decision_Sigmoide(d["w"], d["b"], image)
print ("le modele predit que c'est")
print(classe_predite)






#%% Les noms des couches de xception (au cas où vous souhaiteriez changer la procedure de vectorisation)
# X= ['input_layer_13', 'block1_conv1', 'block1_conv1_bn', 'block1_conv1_act', 'block1_conv2', 'block1_conv2_bn', 'block1_conv2_act', 'block2_sepconv1', 'block2_sepconv1_bn', 'block2_sepconv2_act', 'block2_sepconv2', 'block2_sepconv2_bn', 'conv2d_36', 'block2_pool', 'batch_normalization_36', 'add_108', 'block3_sepconv1_act', 'block3_sepconv1', 'block3_sepconv1_bn', 'block3_sepconv2_act', 'block3_sepconv2', 'block3_sepconv2_bn', 'conv2d_37', 'block3_pool', 'batch_normalization_37', 'add_109', 'block4_sepconv1_act', 'block4_sepconv1', 'block4_sepconv1_bn', 'block4_sepconv2_act', 'block4_sepconv2', 'block4_sepconv2_bn', 'conv2d_38', 'block4_pool', 'batch_normalization_38', 'add_110', 'block5_sepconv1_act', 'block5_sepconv1', 'block5_sepconv1_bn', 'block5_sepconv2_act', 'block5_sepconv2', 'block5_sepconv2_bn', 'block5_sepconv3_act', 'block5_sepconv3', 'block5_sepconv3_bn', 'add_111', 'block6_sepconv1_act', 'block6_sepconv1', 'block6_sepconv1_bn', 'block6_sepconv2_act', 'block6_sepconv2', 'block6_sepconv2_bn', 'block6_sepconv3_act', 'block6_sepconv3', 'block6_sepconv3_bn', 'add_112', 'block7_sepconv1_act', 'block7_sepconv1', 'block7_sepconv1_bn', 'block7_sepconv2_act', 'block7_sepconv2', 'block7_sepconv2_bn', 'block7_sepconv3_act', 'block7_sepconv3', 'block7_sepconv3_bn', 'add_113', 'block8_sepconv1_act', 'block8_sepconv1', 'block8_sepconv1_bn', 'block8_sepconv2_act', 'block8_sepconv2', 'block8_sepconv2_bn', 'block8_sepconv3_act', 'block8_sepconv3', 'block8_sepconv3_bn', 'add_114', 'block9_sepconv1_act', 'block9_sepconv1', 'block9_sepconv1_bn', 'block9_sepconv2_act', 'block9_sepconv2', 'block9_sepconv2_bn', 'block9_sepconv3_act', 'block9_sepconv3', 'block9_sepconv3_bn', 'add_115', 'block10_sepconv1_act', 'block10_sepconv1', 'block10_sepconv1_bn', 'block10_sepconv2_act', 'block10_sepconv2', 'block10_sepconv2_bn', 'block10_sepconv3_act', 'block10_sepconv3', 'block10_sepconv3_bn', 'add_116', 'block11_sepconv1_act', 'block11_sepconv1', 'block11_sepconv1_bn', 'block11_sepconv2_act', 'block11_sepconv2', 'block11_sepconv2_bn', 'block11_sepconv3_act', 'block11_sepconv3', 'block11_sepconv3_bn', 'add_117', 'block12_sepconv1_act', 'block12_sepconv1', 'block12_sepconv1_bn', 'block12_sepconv2_act', 'block12_sepconv2', 'block12_sepconv2_bn', 'block12_sepconv3_act', 'block12_sepconv3', 'block12_sepconv3_bn', 'add_118', 'block13_sepconv1_act', 'block13_sepconv1', 'block13_sepconv1_bn', 'block13_sepconv2_act', 'block13_sepconv2', 'block13_sepconv2_bn', 'conv2d_39', 'block13_pool', 'batch_normalization_39', 'add_119', 'block14_sepconv1', 'block14_sepconv1_bn', 'block14_sepconv1_act', 'block14_sepconv2', 'block14_sepconv2_bn', 'block14_sepconv2_act', 'avg_pool', 'predictions']