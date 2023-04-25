import os
import re
import numpy as np
import json
import cv2
from matplotlib import pyplot as plt
from mapcalc import calculate_map
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc

import itertools
import shutil
import math


def get_images_test(filename_test):
    """
    Retourne la liste des images présentent dans le fichier filename_test.
    filename_test : nom du fichier (str) 
    return : la liste des images réservées pour le test (list str)
    """
    list_images_test = []
    with open(filename_test, 'r') as f:
        for line in f:
            list_images_test.append(line.strip())
    return list_images_test

def predict_custom(image_list,dossier_resultats, dossier_weights_cfg,weight):
    """
    Effectue les prédictions pour les images tests dans le cas où le train était effectué à partir d'un réseau pré-entraîné. 
    Les prédictions sont stockées dans des fichiers .txt de même nom que l'image dans le dossier dossier_resultats.
    Le temps pour effectuer chaque prédiction ainsi que le moyenne de temps de prédiction sont stockées dans un fichier prediction_times.txt. 
    image_list : liste des images pour le test (list str)
    dossier_resultats : chemin vers le dossier dans lequel sera stocké le résultat (str)
    dossier_weights_cfg : chemin vers le dossier dans lequel était stocké le fichier cfg et le fichier contenant les poids (str)
    weigh : numéro du poids dans le cas où plusieurs versions existent (int ou chaîne vide)
    return : None 
    """
    total_time = 0.0
    nb_images = len(image_list)

    with open(f"{dossier_resultats}prediction_stats/prediction_times.txt", "w") as f:
        for image in image_list:
            output = os.popen(f"./darknet detector test data/piford.data {dossier_weights_cfg}cfg/yolov4-custom.cfg {dossier_weights_cfg}backup/yolov4-custom_best{weight}.weights {image} -thresh 0.05 -dont_show -map -ext_output -out {dossier_resultats}prediction/{image[12:-4]}.txt").read()

            match = re.search(r"prédite en (\d+\.\d+) ms.", output)
            if match==None :
                match = re.search(r"Predicted in (\d+\.\d+) milli-seconds.", output)

            time_prediction = float(match.group(1))
            total_time += time_prediction

            f.write(f"{image}: {time_prediction} ms\n")

            print(f"Image {image} prédite en {time_prediction} ms")
    f.close()

    avg_time = total_time / nb_images
    fps = nb_images / (total_time)*1000
    with open(f"{dossier_resultats}prediction_stats/prediction_times.txt", "a") as f:
        f.write(f"\nAverage prediction time for {nb_images} images: {avg_time:.3f} ms\n")
        f.write(f"\nFrame per second calculated on {nb_images} images: {fps:.3f}\n")
    f.close()

def predict(image_list, dossier_resultats, dossier_weights_cfg, weight):
    """
    Effectue les prédictions pour les images tests dans le cas où le train était effectué à partir d'un réseau non pré-entraîné. 
    Les prédictions sont stockées dans des fichiers .txt de même nom que l'image dans le dossier dossier_resultats.
    Le temps pour effectuer chaque prédiction ainsi que le moyenne de temps de prédiction sont stockées dans un fichier prediction_times.txt. 
    image_list : liste des images pour le test (list str)
    dossier_resultats : chemin vers le dossier dans lequel sera stocké le résultat (str)
    dossier_weights_cfg : chemin vers le dossier dans lequel était stocké le fichier cfg et le fichier contenant les poids (str)
    weigh : numéro du poids dans le cas où plusieurs versions existent (int ou chaîne vide)
    return : None 
    """
    total_time = 0.0
    nb_images = len(image_list)

    with open(f"{dossier_resultats}prediction_stats/prediction_times.txt", "w") as f:
        for image in image_list:
            output = os.popen(f"./darknet detector test data/piford.data {dossier_weights_cfg}cfg/yolov4.cfg {dossier_weights_cfg}backup/yolov4_best{weight}.weights {image} -thresh 0.05 -dont_show -map -ext_output -out {dossier_resultats}prediction/{image[12:-4]}.txt").read()
    
            match = re.search(r"prédite en (\d+\.\d+) ms.", output)
            if match==None :
                match = re.search(r"Predicted in (\d+\.\d+) milli-seconds.", output)

            time_prediction = float(match.group(1))
            total_time += time_prediction

            f.write(f"{image}: {time_prediction} ms\n")

            print(f"Image {image} prédite en {time_prediction} ms")
    f.close()

    avg_time = total_time / nb_images
    fps = nb_images / (total_time)*1000
    with open(f"{dossier_resultats}prediction_stats/prediction_times.txt", "a") as f:
        f.write(f"\nAverage prediction time for {nb_images} images: {avg_time:.3f} ms\n")
        f.write(f"\nFrame per second calculated on {nb_images} images: {fps:.3f}\n")
    f.close()

def get_coordinates(image_list,dossier_resultats):
    """
    Retourne les coordonnées des bbox prédites pour chaque image.
    image_list : liste des images pour le test (list str)
    dossier_resultats : chemin vers le dossier dans lequel sera stocké le résultat (str)
    return : dictionnaire (str : list str)
    """
    bbox_dict = dict()

    for image in image_list :
        image = f'{dossier_resultats}prediction/{image[12:-4]}.txt'
        with open(image) as f:
            result_dict = json.load(f)
            objects = result_dict[0]['objects']
        
        l = len(dossier_resultats)

        bbox_dict[image[11+l:-4]] = []
        for obj in objects:
            bbox = obj['relative_coordinates']
            bbox_dict[image[11+l:-4]].append([bbox['center_x'], bbox['center_y'], bbox['width'], bbox['height']])

        if bbox_dict[image[11+l:-4]] == [] :
            bbox_dict[image[11+l:-4]] = [[0,0,0,0]]

    return bbox_dict

def get_confidence(image_list,dossier_resultats):
    """
    Retourne un dictionnaire associant un identifiant d'image à la liste de ses scores de confiance.
    image_list : liste des images pour le test (list str)
    dossier_resultats : chemin vers le dossier dans lequel sera stocké le résultat (str)
    return : dictionnaire (str : list str)
    """
    confidence_dict = dict()

    for image in image_list :
        image = f'{dossier_resultats}prediction/{image[12:-4]}.txt'
        with open(image) as f:
            result_dict = json.load(f)
            objects = result_dict[0]['objects']

        l = len(dossier_resultats)

        confidence_dict[image[11+l:-4]] = []
        for obj in objects:
            confidence = obj['confidence']
            confidence_dict[image[11+l:-4]].append(confidence)

        if confidence_dict[image[11+l:-4]] == []:
            confidence_dict[image[11+l:-4]] = [0]

    return confidence_dict

def keep_best_predict(dict_coord, dict_coord_true, dict_pred):
    """
    Permet de ne conserver que la bbox associée au meilleur score de confiance pour chaque image. 
    dict_coord : dictionnaire associant chaque image à ses coordonéees de bbox 
    dict_coord_true : dictionnaire associant chaque image à ses réelles coordonnées 
    dict_pred : dictionnaire associant chaque image aux scores de confiance de ses bbox
    return : trois dictionnaires (dict_coord, dict_coord_true et dict_pred) avec uniquement les informations de la bbox avec la meilleure confiance
    """
    for img, preds in dict_pred.items():
        m = preds.index(max(preds))
        dict_coord[img] = dict_coord[img][m]
        dict_coord_true[img] = dict_coord_true[img][m]
        dict_pred[img] = dict_pred[img][m]
    return dict_coord, dict_coord_true, dict_pred

def intersection_over_union(label_true, label_pred):
    """
    Calcule le score IoU à partir des coordonnées réelles et prédites de bbox.
    label_true : coordonnées réelles de la bbox de l'image 
    label_pred : coordonnées prédites de la bbox l'image
    return : score IoU
    """
    #Coordonnées
    x_center_true = float(label_true[0])
    y_center_true = float(label_true[1])
    width_true = float(label_true[2])
    height_true = float(label_true[3])

    x_center_pred = float(label_pred[0])
    y_center_pred = float(label_pred[1])
    width_pred = float(label_pred[2])
    height_pred = float(label_pred[3])

    #Calcule les coins supérieur gauche et inférieur droite
    x1_true = float(x_center_true - 0.5 * width_true)
    y1_true = float(y_center_true - 0.5 * height_true)
    x2_true = float(x_center_true + 0.5 * width_true)
    y2_true = float(y_center_true + 0.5 * height_true)

    x1_pred = float(x_center_pred - 0.5 * width_pred)
    y1_pred = float(y_center_pred - 0.5 * height_pred)
    x2_pred = float(x_center_pred + 0.5 * width_pred)
    y2_pred = float(y_center_pred + 0.5 * height_pred)

    #Calcule les coordonnées de l'intersection
    x1_inters = max(x1_true, x1_pred)
    y1_inters = max(y1_true, y1_pred)
    x2_inters = min(x2_true, x2_pred)
    y2_inters = min(y2_true, y2_pred)

    #Calcule surface intersection : vaut 0 si x2_inters plus petit que x1_inters (idem avec y)
    surf_inters = max(0, x2_inters - x1_inters) * max(0, y2_inters - y1_inters)

    #Calule surface union
    surf_union = width_true*height_true + width_pred*height_pred - surf_inters

    return surf_inters/surf_union

def liste_IoU(dict_true, dict_coord):
    """
    Permet de calculer l'IoU pour chaque image (vaut 0 en cas de faux positif)
    dict_true : dictionnaire associant l'identifiant de l'image aux coordonnées réelles de la bbox de l'image (dict( str : list str))
    dict_coord : dictionnaire associant l'identifiant de l'image aux coordonnées prédites de la bbox de l'image (dict (str : list int))
    return : dictionnaire des scores iou pour chaque image (dict (str : float))
    """
    scores_iou = dict()
    for name, label_true in dict_true.items():
        label_predict = dict_coord[name]
        if label_true != ['0', '0', '0', '0\n'] :
            scores_iou[name] = intersection_over_union(label_true, label_predict)
        else :
            scores_iou[name] = 0
    return scores_iou

def get_true_coordinates(image_list):
    """
    Permet de récupérer les coordonnées réelles pour chaque image de la liste.
    image_list : liste des identifiants des images (list str)
    return : dictionnaire associant chaque identifiant d'image à la bbox réeelle (dict (str : str))
    """
    labels_true = dict()
    liste_img_neg = []
    for image in image_list:
        image = f'{image[:-4]}.txt'
        with open(image,'r') as fp:
            ligne = fp.readline()
            liste = ligne.split(' ')
            if liste == ['\n'] :
                liste = ['-1', '0','0','0','0\n']
                liste_img_neg.append(image)
            labels_true[image[12:-4]] = [liste[1:]]

    return labels_true, liste_img_neg 

def load_images(path, gray_scale = False):
    """
    Permet de télécharger toutes les images se trouvant dans un répertoire.
    path : chemin vers lequel se trouve les images (str)
    gray_scale : images en noir et blanc ou en couleur (boolean)
    return : dictionnaire associant chaque identifiant d'image à l'image
    """
    path = "data/piford"
    images = dict()
    names = os.listdir(path)
    for f in os.listdir(path):
      img = None
      if gray_scale :
          img = cv2.imread(os.path.join(path, f), gray_scale)
      else :
          img = cv2.imread(os.path.join(path, f))
      if img is not None:
          img = cv2.resize(img, (1024,1024))
          images[f[:-4]] = img
    return images

def draw_rectangle(name, img, coord, coord_true, pred, iou, dossier_resultats):
    """
    Permet de dessiner les bbox réelles et correspondantes à la prédiction ainsi que d'afficher le score de confiance et l'IoU.
    name : nom de l'image (str)
    img : image (list list int)
    coord : coordonnées prédites de la bbox (list str)
    coord_true : coordonnées réelles de la bbox (list str)
    pred : score de confiance(float)
    iou : score IoU (float)
    dossier_resultats : chemin vers le dossier où sera enregistré l'image modifiée
    return : l'image modifiée
    """
    n , m , c = img.shape

    # coordonnées de la prédiction
    x_center = int(float(coord[0]) * m)
    y_center = int(float(coord[1]) * n)
    width = int(float(coord[2]) * m)
    height = int(float(coord[3]) * n)
    x1 = int(x_center - 0.5 * width)
    y1 = int(y_center - 0.5 * height)
    x2 = int(x_center + 0.5 * width)
    y2 = int(y_center + 0.5 * height)

    # coordonnées réelles
    x_center_true = int(float(coord_true[0]) * m)
    y_center_true = int(float(coord_true[1]) * n)
    width_true = int(float(coord_true[2]) * m)
    height_true = int(float(coord_true[3]) * n)
    x1_true = int(x_center_true - 0.5 * width_true)
    y1_true = int(y_center_true - 0.5 * height_true)
    x2_true = int(x_center_true + 0.5 * width_true)
    y2_true = int(y_center_true + 0.5 * height_true)

    # draw rectangles on image
    img_modif = img.copy()
    if (x_center,y_center,width,height)!=(0,0,0,0):
        img_modif = cv2.rectangle(img_modif, (x1, y1), (x2, y2), (0, 0, 255),thickness = 2)
        img_modif = cv2.putText(img_modif, f'{pred:.3f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    if (x_center_true,y_center_true,width_true,height_true)!=(0,0,0,0):
        img_modif = cv2.rectangle(img_modif, (x1_true , y1_true ), (x2_true , y2_true ), (255, 0, 0),thickness = 2)
        img_modif = cv2.putText(img_modif, 'true', (x2_true-55, y1_true-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    fig = plt.figure(figsize=(20, 10))
    plt.imshow(img_modif)
    plt.title(f'{name} (IoU : {iou})')
    plt.xticks([]), plt.yticks([])
    plt.savefig(f'{dossier_resultats}prediction/{name}.png')
    return img_modif

def images_rectangles(images, dict_coord, dict_coord_true, dict_pred, scores_iou, dossier_resultats):
    """
    Permet de dessiner les bbox réelles et prédites pour toutes les images.
    images : liste des images (list str)
    dict_coord : dictionnaire associant les identifiants des images aux coordonnées prédites (dict (str : list str))
    dict_coord_true : dictionnaire associant les identifiants des images aux coordonnées réelles (dict (str : list str))
    dict_pred : dictionnaire associant les identifiants des images aux scores de confiance (dict (str : float))
    """
    for img, coord in dict_coord.items():
        draw_rectangle(img, images[img], dict_coord[img], dict_coord_true[img], dict_pred[img], scores_iou[img], dossier_resultats)
    return

def prepare_map(images, dict_true, dict_pred, dico_scores_iou):
    """
    Permet de formater les listes de façon à pour calculer la MAP. 
    images : liste des images (list str)
    dict_true : dictionnaire associant l'identifiant d'une image aux coordonnées de sa bbox (dict (str : list str))
    dict_pred : dictionnaire associant l'identifiant d'une image à son score de confiance (dict (str : float))
    dict_scores_iou : dictionnaire associant l'identifiant d'une image à son score iou (dict (str : float))
    return y_true : liste où chaque élément correspond à une image et vaut -1 si l'image est négative et 0 sinon
    return pred_scores : liste où chaque élément correspond à une image et correspond à son score de confiance
    return scores_iou : liste où chaque élément correspond à une image et correspond à son score IoU
    """
    y_true = []
    pred_scores = []
    scores_iou = []
    for name_l in images:
        img = name_l[12:-4]

        if dict_true[img] == ['0', '0', '0', '0\n']:
            y_true.append(-1)
        else :
            y_true.append(0)

        #score de prediction
        pred_scores.append(dict_pred[img])
        scores_iou.append(dico_scores_iou[img])

    return y_true, pred_scores, scores_iou

def liste_CIoU(dict_true, dict_coord, scores_iou):
    """
    Permet de calculer le score CIoU pour chaque image.
    dict_true : dictionnaire associant chaque image aux coordonnées réelles de la bbox (dict (str : list str))
    dict_coord : dictionnaire associant chaque image aux coordonnées prédites (dict (str : list str))
    scores_iou : dictionnaire associant chaque image à son score IoU (dict (str : float))
    return : dictionnaire associant chaque image à son score CIoU (dict (str : float))
    """
    scores_c_iou = dict()
    for name, label_true in dict_true.items():
        label_predict = dict_coord[name]
        if label_true != ['0', '0', '0', '0\n'] :
            scores_c_iou[name] = c_iou(label_true, label_predict, scores_iou[name])
        else :
            scores_c_iou[name] = 0

    return scores_c_iou

def write_stats_prediction(dict_coord, dict_coord_true, dict_pred, scores_iou, score_map, best_threshold,dossier_resultats):
    """
    Permet d'écrire dans le dossier predicition_stats les coordonnées prédites pour chaque image, les coordonnées réelles, les scores de confiance, 
    les scores IoU, les scores CIoU, le meilleur score MAP ainsi que le threshold associé.
    dict_coord : dictionnaire associant chaque image aux coordonnées prédites (dict (str : list str))
    dict_coord_true : dictionnaire associant chaque image aux coordonnées réelles (dict (str : list str))
    scores_iou : dictionnaire associant chaque image à son score IoU (dict (str : float))
    score_map : meilleur score MAP (float)
    best_threshold : threshold associé au meilleur score MAP (float)
    dossier_resultat : chemin vers le dossier où seront stockés les résultats (str)
    """
    with open(f"{dossier_resultats}prediction_stats/coord_pred.txt", "w") as f:
        f.write(json.dumps(dict_coord))
    f.close()

    with open(f"{dossier_resultats}prediction_stats/coord_true.txt", "w") as f:
        f.write(json.dumps(dict_coord_true))
    f.close()

    with open(f"{dossier_resultats}prediction_stats/confidences.txt", "w") as f:
        f.write(json.dumps(dict_pred))
    f.close()

    with open(f"{dossier_resultats}prediction_stats/scores_iou.txt", "w") as f:
        f.write(json.dumps(scores_iou))
    f.close()

    with open(f"{dossier_resultats}prediction_stats/best_score_map.txt", "w") as f:
        f.write(str(score_map))
    f.close()

    with open(f"{dossier_resultats}prediction_stats/best_threshold.txt", "w") as f:
        f.write(str(best_threshold))
    f.close()

    liste_cIoU = liste_CIoU(dict_coord_true, dict_coord, scores_iou)
    with open(f"{dossier_resultats}prediction_stats/scores_ciou.txt", "w") as f:
        f.write(json.dumps(liste_cIoU))
    f.close()

def map_range(y_true, pred_scores, scores_iou, thresholds, dossier_resultats):
    """
    Permet de calculer le score map en fonction du thresholds et de tracer la courbe MAP vs threshold et l'enregistrer dans dossier_resultats, 
    ainsi que de trouver le meilleur score MAP et le threshold associé.
    y_true : liste où chaque élément correspond à une image et vaut -1 si l'image est négative et 0 sinon (list int)
    pred_scores : liste où chaque élément correspond à une image et correspond à son score de confiance (list float)
    scores_iou : liste où chaque élément correspond à une image et correspond à son score IoU (list float)
    thresholds : liste de thresholds à tester (list float)
    dossier_resultats : chemin vers le dossier où seront stockés les résultats (str)
    return best_map : meilleur score map obtenu (float)
    return best_threshold : meilleur threshold correspondant (float)
    """
    liste_AP = []
    for threshold in thresholds:
        pred_labels = np.where( pred_scores >= threshold , 1, 0)
        AP = average_precision_score(y_true, pred_labels, pos_label=0)
        liste_AP.append(AP)

    best_map = np.max(liste_AP)
    best_index = np.where(np.array(liste_AP)==best_map)[0]
    if best_map >= 0.5 :
        best_threshold = thresholds[best_index[0]]
    else :
        best_threshold = thresholds[best_index[-1]]

    plt.figure()
    plt.plot(thresholds, liste_AP, linewidth=4, color="red", zorder=0)
    plt.xlabel("Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("MAP", fontsize=12, fontweight='bold')
    plt.title("MAP vs Threshold", fontsize=15, fontweight="bold")
    plt.savefig(f"{dossier_resultats}prediction_stats/MAP_vs_threshold.jpg")

    return best_map, best_threshold

def precision_recall_curve(y_true, pred_scores, thresholds, dossier_resultats):
    """
    Permet de tracer les courbes de precisions et de rappel en fonction des thresholds, ainsi que le precision-recall curve.
    y_true : liste où chaque élément correspond à une image et vaut -1 si l'image est négative et 0 sinon (list int)
    pred_scores : liste où chaque élément correspond à une image et correspond à son score de confiance (list float)
    thresholds : liste de thresholds à tester (list float)
    dossier_resultats : chemin vers le dossier où seront stockés les résultats (str)
    return precisions : liste des precisions en fonction des thresholds (list float)
    return recalls : liste des recalls en fonction des thresholds (list float)
    """
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = [0 if score >= threshold else -1 for score in pred_scores]
        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label=0, average='binary', zero_division=0)
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label=0, average='binary', zero_division=0)
        precisions.append(precision)
        recalls.append(recall)

    plt.figure()
    plt.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.savefig(f"{dossier_resultats}prediction_stats/Precisions_recall_curve.jpg")

    plt.figure()
    plt.plot(thresholds,precisions, linewidth=4, color="red", zorder=0)
    plt.xlabel("thresholds", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision Curve", fontsize=15, fontweight="bold")
    plt.savefig(f"{dossier_resultats}prediction_stats/Precisions_curve.jpg")

    plt.figure()
    plt.plot(thresholds, recalls, linewidth=4, color="red", zorder=0)
    plt.xlabel("thresholds", fontsize=12, fontweight='bold')
    plt.ylabel("Recall", fontsize=12, fontweight='bold')
    plt.title("Recall Curve", fontsize=15, fontweight="bold")
    plt.savefig(f"{dossier_resultats}prediction_stats/Recalls_curve.jpg")

    return precisions, recalls

def matrice_confusion(y_true, pred_scores, best_threshold, dossier_resultats):
    """
    Permet de tracer la matrice de confusion obtenue pour le best_threshold et de l'enregistrer dans le dossier_resultat.
    y_true : liste où chaque élément correspond à une image et vaut -1 si l'image est négative et 0 sinon (list int)
    pred_scores : liste où chaque élément correspond à une image et correspond à son score de confiance (list float)
    best_threshold : threhsold permettant de maximiser la MAP (float)
    dossier_resultats : chemin vers le dossier où seront stockés les résultats (str)
    """
    pred = np.where(np.array(pred_scores) < best_threshold, -1, 0)

    classes = [-1, 0]
    cm = confusion_matrix(y_true, pred, labels=classes)

    # Plot de la matrice de confusion
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(f"{dossier_resultats}prediction_stats/confusion_matrix.jpg")

def d_iou(label_true, label_pred, scores_iou):
    """
    Calcule le score DIoU.
    label_true : coordonnées réelles de la bbox (list str)
    label_pred : coordonnées prédites de la bbox (list str)
    scores_iou : score iou (float)
    return : score DIoU (float)
    """
    x_center_true = float(label_true[0])
    y_center_true = float(label_true[1])
    width_true = float(label_true[2])
    height_true = float(label_true[3])

    x_center_pred = float(label_pred[0])
    y_center_pred = float(label_pred[1])
    width_pred = float(label_pred[2])
    height_pred = float(label_pred[3])

    x1_true = float(x_center_true - 0.5 * width_true)
    y1_true = float(y_center_true - 0.5 * height_true)
    x2_true = float(x_center_true + 0.5 * width_true)
    y2_true = float(y_center_true + 0.5 * height_true)

    x1_pred = float(x_center_pred - 0.5 * width_pred)
    y1_pred = float(y_center_pred - 0.5 * height_pred)
    x2_pred = float(x_center_pred + 0.5 * width_pred)
    y2_pred = float(y_center_pred + 0.5 * height_pred)

    d2 = (y_center_true - y_center_pred) ** 2 + (x_center_true - x_center_pred) ** 2
    min_xL = min(x1_true, x1_pred)
    min_yL = min(y1_true, y1_pred)
    max_xR = max(x2_true, x2_pred)
    max_yR = max(y2_true, y2_pred)
    c2 = ((min_yL - max_yR) ** 2 + (min_xL - max_xR) ** 2) + 1e-10

    diou = scores_iou - (d2 / c2)
    return diou

def c_iou(label_true, label_pred, score_iou):
    """
    Calcule le score CIoU.
    label_true : coordonnées réelles de la bbox (list str)
    label_pred : coordonnées prédites de la bbox (list str)
    scores_iou : score iou (float)
    return : score CIoU (float)
    """
    x_center_true = float(label_true[0])
    y_center_true = float(label_true[1])
    width_true = float(label_true[2])
    height_true = float(label_true[3])

    x_center_pred = float(label_pred[0])
    y_center_pred = float(label_pred[1])
    width_pred = float(label_pred[2])
    height_pred = float(label_pred[3])

    x1_true = float(x_center_true - 0.5 * width_true)
    y1_true = float(y_center_true - 0.5 * height_true)
    x2_true = float(x_center_true + 0.5 * width_true)
    y2_true = float(y_center_true + 0.5 * height_true)

    x1_pred = float(x_center_pred - 0.5 * width_pred)
    y1_pred = float(y_center_pred - 0.5 * height_pred)
    x2_pred = float(x_center_pred + 0.5 * width_pred)
    y2_pred = float(y_center_pred + 0.5 * height_pred)

    d2 = (y_center_true - y_center_pred) ** 2 + (x_center_true - x_center_pred) ** 2
    min_xL = min(x1_true, x1_pred)
    min_yL = min(y1_true, y1_pred)
    max_xR = max(x2_true, x2_pred)
    max_yR = max(y2_true, y2_pred)
    c2 = ((min_yL - max_yR) ** 2 + (min_xL - max_xR) ** 2) + 1e-10

    v = (4 / (math.pi**2)) * (
        math.atan(width_true / height_true) - math.atan(width_pred / height_pred)
    ) ** 2
    al = v / ((1 - score_iou) + v)
    ciou = score_iou - (d2 / c2) - al * v
    return ciou

def copie_chart():
    """
    Permet de copier l'image chart de l'apprentissage.
    """
    current_dir = os.getcwd()
    dest = os.path.join(current_dir, "prediction_stats")
    src = os.path.join(current_dir, "chart.png")
    try:
        shutil.copy2(src, os.path.abspath(dest))
    except OSError as error:
        print("Une erreur s'est produite lors de la copie de chart.png.")
        print(error)

def copie_backup():
    """
    Permet de copier le meilleur weight obtenu en apprentissage.
    """
    current_dir = os.getcwd()
    dest = os.path.join(current_dir, "prediction_stats")
    src = os.path.join(current_dir, "backup/yolov4-custom_best.weights")
    try:
        shutil.copy2(src, os.path.abspath(dest))
    except OSError as error:
        print("Une erreur s'est produite lors de la copie de yolov4_custom_best.weights.")
        print(error)

def copie_train():
    """
    Permet de copier le fichier train.txt pour savoir quelles images ont été utilisées en train. 
    """
    current_dir = os.getcwd()
    dest = os.path.join(current_dir, "prediction_stats")
    src = os.path.join(current_dir, "data/train.txt")
    try:
        shutil.copy2(src, os.path.abspath(dest))
    except OSError as error:
        print("Une erreur s'est produite lors de la copie de train.txt.")
        print(error)

def copie_test():
    """
    Permet de copier le fichier test.txt pour savoir quelles images ont été utilisées en test.
    """
    current_dir = os.getcwd()
    dest = os.path.join(current_dir, "prediction_stats")
    src = os.path.join(current_dir, "data/test.txt")
    try:
        shutil.copy2(src, os.path.abspath(dest))
    except OSError as error:
        print("Une erreur s'est produite lors de la copie de test.txt.")
        print(error)

def avg_iou(scores_iou,liste_img_neg, dossier_resultats):
    """
    Permet de calculer le score IoU moyen obtenu sur les images tests et de l'enregistrer dans un fichier.
    scores_iou : dictionnaire associant à chaque image un score iou (dict (str: float))
    liste_img_neg : liste des images négatives (list str)
    dossier_resultats : chemin vers le dossier où sera enregistré la moyenen des scores iou (str)
    return : moyenne des scores IoU des images positives (float)
    """
    liste_iou = [iou for img,iou in scores_iou.items() if 'data/piford/'+img+'.txt' not in liste_img_neg]
    avg = np.mean(liste_iou)
    with open(f"{dossier_resultats}prediction_stats/avg_iou.txt", "w") as f:
        f.write(str(avg))
    f.close()
    return avg

def run(file, dossier_resultats, dossier_weights_cfg, weight, custom = True) :
    """
    Permet de lancer une évaluation à partir d'un poids et d'un cfg. 
    file : chemin vers le fichier test.txt (str)
    dossier_resultats : chemin vers le dossier où seront enregistrés les résultats (str)
    dossier_weight_cfg : chemin vers le dossier contenant les dossiers backup (contenant le poids obtenu en train) et cfg (str)
    weight : numéro du weight à utiliser (int ou chaîne vide)
    custom : permet de préciser si le réseau était pré-entraîné ou non.
    """
    print("start :")
    image_list = get_images_test(file)
    
    if custom : 
        predict_custom(image_list, dossier_resultats, dossier_weights_cfg, weight)
    else : 
        predict(image_list, dossier_resultats, dossier_weights_cfg, weight)

    dict_coord = get_coordinates(image_list,dossier_resultats)
    dict_coord_true,liste_img_neg = get_true_coordinates(image_list)
    dict_pred = get_confidence(image_list,dossier_resultats)
    #print("liste images : ", image_list)
    #print("coordonnées : \n", dict_coord)
    #print("coordonnées réelles: \n", dict_coord_true)
    #print("predictions : \n", dict_pred)

    #print("liste images négatives :", liste_img_neg)


    dict_coord, dict_coord_true, dict_pred = keep_best_predict(dict_coord, dict_coord_true, dict_pred)
    scores_iou = liste_IoU(dict_coord_true,dict_coord)
    #print("\ncoordonnées : \n", dict_coord)
    #print("coordonnées réelles: \n", dict_coord_true)
    #print("predictions : \n", dict_pred)
    #print("scores iou : \n", scores_iou)

    images = load_images('data/piford')
    images_rectangles(images, dict_coord, dict_coord_true, dict_pred, scores_iou, dossier_resultats)

    y_true, pred_scores, liste_scores_iou = prepare_map(image_list, dict_coord_true, dict_pred, scores_iou)
    #print(ground_truth)
    #print(result_dict)
    thresholds = np.arange(0.05, 1, 0.01)
    precision_recall_curve(y_true, pred_scores, thresholds, dossier_resultats)
    best_map, best_threshold = map_range(y_true, pred_scores, liste_scores_iou, thresholds,dossier_resultats)
    print("best map : ", best_map)
    print("threshold : ", best_threshold)

    write_stats_prediction(dict_coord, dict_coord_true, dict_pred, scores_iou, best_map, best_threshold,dossier_resultats)
    matrice_confusion(y_true, pred_scores, best_threshold,dossier_resultats)
    
    #copie_chart()
    #copie_backup()
    #copie_test()
    #copie_train()

    avg = avg_iou(scores_iou,liste_img_neg,dossier_resultats)
    print("average IoU : ", avg)

    print("done")

def run_all(file) :
    """
    Permet de lancer l'évaluation à partir d'un ensemble de poids obtenus lors de différents apprentissages.
    La premier à partir d'un réseau non pré-entraîné et les six suivants à partir d'un réseau pré-entraîné.
    Les poids et cfg dont dans le dossier weights_cfg.
    file : chemin vers le fichier test.txt (str)
    """
    print("start all: ")
    dossier_weights_cfg =  "weights_cfg/"

    #premiere version (non custom) 
    dossier_resultats = "resultats/V1/" 
    run(file, dossier_resultats, dossier_weights_cfg, 1, False)

    #autres versions (custom)
    nb_versions = 7
    for i in range(2, nb_versions+1) : 
        dossier_resultats = f"resultats/V{i}/" 
        run(file, dossier_resultats, dossier_weights_cfg, i, True)
    print("done")



run('data/test.txt', '', '', '', True)
#run_all('data/test.txt')

