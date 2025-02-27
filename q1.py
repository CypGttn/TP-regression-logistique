import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Charger les caractéristiques extraites
X_train = np.loadtxt("X_train.csv", delimiter=",")
y_train = np.loadtxt("y_train.csv", delimiter=",")

# Réduction de dimension avec PCA pour visualisation 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

print(X_train_pca.shape)  # Doit être (N, 2) après PCA
print(y_train.shape)  # Doit être (N, 1) ou (N,)

y_train = y_train[:X_train_pca.shape[0]]  # Tronque pour correspondre à X_train_pca

# S'assurer qu'on a le même nombre d'échantillons
if X_train_pca.shape[0] != y_train.shape[0]:
    raise ValueError(f"Incohérence dans les dimensions : {X_train_pca.shape[0]} != {y_train.shape[0]}")


y_train = y_train.flatten()  # Transforme en un vecteur 1D de taille (N,)

# Affichage des caractéristiques dans un graphe 2D
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=y_train.flatten(), palette="coolwarm")
plt.title("Projection PCA des vecteurs de caractéristiques Xception")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(["Classe 0", "Classe 1"])
plt.show()

# Calcul de la similarité entre les vecteurs de caractéristiques
similarity_matrix = cosine_similarity(X_train)

# Affichage sous forme de heatmap
plt.figure(figsize=(10,8))
sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False)
plt.title("Matrice de similarité cosinus entre les images")
plt.show()
