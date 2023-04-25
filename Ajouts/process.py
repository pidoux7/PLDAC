import glob
import os
import numpy as np
import sys

current_dir = "data/piford"
pourcent = 85
file_train = open("data/train.txt", "w")
file_val = open("data/test.txt", "w")
counter = 1
tot = 0
nb_other = 0
dic = {}
patients = set()
others = set()
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.[bp][mn][pg]")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if(title[0]=='T'):
        file_val.write(current_dir + "/" + title + ".bmp" + "\n")
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

load = 0
for patient in patients:
    if load <= split:
        for title in dic[patient]:
            if title[0] == 'T':
                tile_val.write(current_dir + "/" + title + ".bmp" + "\n")
            else:
                file_train.write(current_dir + "/" + title + ".bmp" + "\n")
                load += 1
    else:
        for title in dic[patient]:
            file_val.write(current_dir + "/" + title + ".bmp" + "\n")
file_train.close()
file_val.close()
print(f"nombre d'image: {tot}\nnombre dans train: {load}\nnombre dans test: {tot-load}")
print("others :", nb_other)
if load > max_split:
    print(
        "ATTENTION !!! \n Le split a été réalisé cependant le train excede de plus de 5% la valeur souhaité"
    )
