# PLDAC

Ce dossier contient tout le nécessaire à la mise en place d'un environnement permettant d'effectuer un apprentissage ou une évaluation d'un modèle pour la détection des voies biliaires. 

  
###################################### A faire ######################################  

1) Tout d'abord, cloner le projet et l'importer sur un drive.
2) Puis, executer le notebook 'Installation.ipynb' afin de télécharger le dossier darknet nécessaire au fonctionnement de YOLO et la mise en place d'un environnement tenant compte des modifications que nous avons apportées. 
Le dossier permet alors d'effectuer des prédictions avec le modèle que nous avons selectionné ou d'entraîner un modèle avec de nouvelles images ou avec d'autres paramètres.
Le notebook 'Test.ipynb' permet de donner un exemple de prédiction et d'évaluation pouvant être effectué.
Le notebook 'Train.ipynb' montre comment l'entraînement d'un nouveau modèle peut être effectué. 

  
##################################### Description ######################################  
  

- Ajouts : contient différents fichiers que nous avons créés afin de répondre à nos problématiques (split, ajout de data_augmentation, évaluation des résultats).
- labelling : contient le nécessaire à la labellisations des données. 
- Modifications : contient des fichiers modifiés de darknet pour répondre aux contraintes de notre problématique.
- negative_images : contient les images négatives labellisées que nous avons utilisées pour entraîner et tester notre modèle.
- original_images : contient les images originales labellisées utilisées pour entraîner et tester notre modèle.
- piford : contient la totalité des images labellisées (y compris les images obtenues par data augmentation).
- resultats : contient les résultats que nous avons obtenus lors des différents train, un fichier décrit les conditions d'entraînement de chaque modèle.
- visualisation_and_data_augmentation : contient trois notebooks permettant de visualiser et décrire les caractéristiques des images à notre disposition, de visualiser les différentes data augmentations possibles et de les effectuer. 
- weights_cfg : contient les fichiers cfg utilisés lors de l'entraînement, il contiendra également par la suire les weights des différents modèles que nous avons testé (ceux-ci sont sur un drive distant et sont téléchargés lors de l'exécution de 'Installation.ipynb').

