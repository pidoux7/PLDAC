import os


def load_labels(path,modif):
    """
    On charge les labels 
    """
    labels = []
    names = os.listdir(path)
    for f in os.listdir(path):
        with open(path+f,'r') as fp:
            ligne = fp.readline()
            liste = ligne.split(' ')
            for i in range(len(liste)):
                liste[i] = int(int(liste[i])/modif*1024)
            labels.append(liste)
    return labels, names

def output(labels, names, o_path):
    for lbl,nm in zip(labels, names):
        with open(o_path+nm,'w') as fp:
            ligne = f'{lbl[0]} {lbl[1]} {lbl[2]} {lbl[3]}\n'
            fp.write(ligne)

path = './Labels/Biliary_ducts/'
output_path = './Labels/resize_Biliary_ducts/'

if __name__ == '__main__':
    modification = 700
    labels, names = load_labels(path,modification)
    output(labels, names, output_path)


        
