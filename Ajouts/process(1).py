import glob
import os
import numpy as np
import sys

# fichier avec la liste des chemins d'images pour le train et le test au format yolo
file_train = open("data/train.txt", "w")
file_val = open("data/test.txt", "w")

# si true, il est possible d'ajouter des images négatives
img_neg = True

# réalisation d'un split train test vers 85% avec marge d'erreur en fonction du nb d'image appartenant à un meme patient aafin de ne pas l'inclure en train et en test

# traitement des images positives
current_dir = "data/original_images"
final_dir = "data/piford"
pourcent = 85
counter = 1
tot = 0
nb_other = 0
dic = {}
patients = set()
others = set()
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.[bp][mn][pg]")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if(title[0]=='T'):
        file_val.write(final_dir + "/" + title + ".bmp" + "\n")
        nb_other+=1
        continue
    else : 
        tot += 1
        patient = title[:4]
        patients.add(patient)
        if patient in dic:
            dic[patient].append(title)
        else:
            dic[patient] = [title]

split = round(pourcent * tot / 100)
max_split = round((pourcent + 5) * tot / 100)
min_split = round((pourcent - 5) * tot / 100)

liste_test = []
liste_train = []

load = 0
for patient in patients:
    if load <= split:
        for title in dic[patient]:
            if title[0] == 'T':
                tile_val.write(final_dir + "/" + title + ".bmp" + "\n")
                liste_test.append(patient)
            else:
                file_train.write(final_dir + "/" + title + ".bmp" + "\n")
                load += 1
                liste_train.append(patient)
    else:
        for title in dic[patient]:
            file_val.write(final_dir + "/" + title + ".bmp" + "\n")
            liste_test.append(patient)
            
print(f"nombre d'images positives: {tot}\nnombre dans train: {load}\nnombre dans test: {tot-load}")
print("others :", nb_other)
if load > max_split:
    print(
        "ATTENTION !!! \n Le split a été réalisé cependant le train excede de plus de 5% la valeur souhaité"
    )
if load < min_split:
    print(
        "ATTENTION !!! \n Le split a été réalisé cependant le train est inférieur de plus de 5% la valeur souhaité"
    )

# traitement des images positives

if img_neg:           
    current_dir = "data/negative_images"
    final_dir = "data/piford"
    pourcent = 85
    counter = 1
    tot = 0
    nb_other = 0
    dic = {}
    patients = set()
    others = set()
    for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.[bp][mn][pg]")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if(title[0]=='T'):
            file_val.write(final_dir + "/" + title + ".bmp" + "\n")
            nb_other+=1
            continue
        else : 
            tot += 1
            patient = title[:4]
            patients.add(patient)
            if patient in dic:
                dic[patient].append(title)
            else:
                dic[patient] = [title]

    split = round(pourcent * tot / 100)
    max_split = round((pourcent + 5) * tot / 100)
    min_split = round((pourcent - 5) * tot / 100)
    
    load = 0
    for patient in patients:
        if str(patient) in liste_test:
            for title in dic[patient]:
                    file_val.write(final_dir + "/" + title + ".bmp" + "\n")
        elif str(patient) in liste_train: 
            for title in dic[patient]:
                    file_train.write(final_dir + "/" + title + ".bmp" + "\n")
                    load +=1
        elif load <= split:
            for title in dic[patient]:
                if title[0] == 'T':
                    tile_val.write(final_dir + "/" + title + ".bmp" + "\n")
                else:
                    file_train.write(final_dir + "/" + title + ".bmp" + "\n")
                    load += 1
        else:
            for title in dic[patient]:
                file_val.write(final_dir + "/" + title + ".bmp" + "\n")

file_train.close()
file_val.close()
print(f"nombre d'images negatives: {tot}\nnombre dans train: {load}\nnombre dans test: {tot-load}")
print("others :", nb_other)
if load > max_split:
    print(
        "ATTENTION !!! \n Le split a été réalisé cependant le train excede de plus de 5% la valeur souhaité"
    )
if load < min_split:
    print(
        "ATTENTION !!! \n Le split a été réalisé cependant le train est inférieur de plus de 5% la valeur souhaité"
    )
