# 🐞 Insect Classifier - Bee or Not to Bee?

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow)

Ce projet est une application web de **Machine Learning** capable de classifier des espèces d'insectes (Abeilles, Bourdons, Autres) en analysant leurs caractéristiques morphologiques, colorimétriques et texturales.

L'application a été conçue pour être déployée sur **Hugging Face Spaces** et utilise une **approche hybride** pour l'extraction de données : elle privilégie les masques de segmentation manuels (fichiers `.tif`) lorsqu'ils existent, et bascule sur une segmentation automatique pour les nouvelles images.

## ✨ Fonctionnalités Clés

* **Interface Intuitive :** Interface utilisateur basée sur **Gradio** permettant de visualiser les résultats et les masques utilisés.
* **Système Hybride de Masquage :**
    * ✅ **Mode Dataset :** Pour les images d'exemple (dossier `img/`), le script charge automatiquement le masque binaire correspondant depuis le dossier `masks/` pour garantir une précision maximale des features.
    * ⚙️ **Mode Upload :** Pour les images importées par l'utilisateur, un algorithme de vision par ordinateur (Otsu Thresholding + Morphologie mathématique) génère un masque en temps réel.
* **Analyse Complète (26 Features) :**
    * **Forme :** Circularité (`roundness`), Symétrie (Verticale/Horizontale), Ratio d'aire, Fit Ellipse.
    * **Couleur :** Statistiques RGB (Min, Moyenne, Écart-type, Médiane).
    * **Texture :** Local Binary Patterns (LBP) pour analyser la surface de l'insecte.
    * **Bords :** Densité des contours (Sobel).

## 📂 Structure du Projet

Voici l'organisation des fichiers nécessaire au bon fonctionnement :

```text
insect-classifier/
│
├── app.py                        # Code principal (FastAPI + Gradio + Extraction Features)
├── logistic_regression_model.pkl # Modèle ML entraîné (Pipeline Scikit-learn)
├── requirements.txt              # Liste des dépendances Python
├── README.md                     # Documentation du projet
│
├── img/                          # Images d'exemple (.jpg/.png) affichées dans l'interface
│   ├── 10.jpg
│   ├── 12.jpg
│   └── ...
│
└── masks/                        # Masques binaires correspondants (.tif) pour la vérité terrain
    ├── binary_10.tif
    ├── binary_12.tif
    └── ...
```
🚀 Installation et Lancement Local
1. Cloner le projet
Bash

git clone [https://github.com/votre-pseudo/insect-classifier.git](https://github.com/votre-pseudo/insect-classifier.git)
cd insect-classifier
2. Créer un environnement virtuel
Bash

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Installer les dépendances
Bash

pip install -r requirements.txt
4. Lancer l'application
Bash

python app.py
L'application sera accessible dans votre navigateur à l'adresse : http://localhost:7860

⚙️ Fonctionnement Technique
Le modèle (Régression Logistique) ne "voit" pas l'image. Il prend en entrée un vecteur mathématique de 26 colonnes.

Chargement de l'image : L'utilisateur sélectionne ou uploade une image.

Recherche de Masque :

Le script regarde le nom du fichier. Si c'est 10.jpg, il cherche masks/binary_10.tif.

Si le fichier TIF existe, il est utilisé (Précision : ⭐⭐⭐⭐⭐).

Si le fichier n'existe pas, un masque est généré via OpenCV (Précision : ⭐⭐⭐).

Extraction : Les bibliothèques opencv et scikit-image calculent les métriques sur les pixels isolés par le masque.

Prédiction : Les données sont envoyées au modèle .pkl pour classification.