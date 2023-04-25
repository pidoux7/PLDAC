import os

# choisissez parmis la liste en commentaire le type de data_augmentation que vous souhaitez
# ceux ci seront ajouté à train.txt 
argument = ['eq','stretch']
#argument = ['eq','stretch','r90','r180','r270','flip','blur','gauss','motion']

def split_aug(arg):
    with open('data/train.txt', 'r') as f:
        lines = f.readlines()
    
    with open('data/train.txt', 'w') as f:
        for i in range(len(lines)):
            f.write(lines[i])
            if 'eq' in arg:
                print('hello')
                f.write(lines[i][:-5] + '_eq' + lines[i][-5:])
            if 'stretch' in arg:  
                f.write(lines[i][:-5] + '_stretch' + lines[i][-5:])
            if 'r90' in arg:
                f.write(lines[i][:-5] + '_r90' + lines[i][-5:])
            if 'r180' in arg:
                f.write(lines[i][:-5] + '_r180' + lines[i][-5:])
            if 'r270' in arg:
                f.write(lines[i][:-5] + '_r270' + lines[i][-5:])
            if 'flip' in arg:
                f.write(lines[i][:-5] + '_flip' + lines[i][-5:])
            if 'blur' in arg:
                f.write(lines[i][:-5] + '_blur' + lines[i][-5:])    
            if 'gauss' in arg:
                f.write(lines[i][:-5] + '_gauss' + lines[i][-5:])
            if 'motion' in arg:
                f.write(lines[i][:-5] + '_motion' + lines[i][-5:])
                
                
    # pour ajouter sur test.txt vous pouvez décommenter les lignes ci dessous
    """
    with open('test_split.txt', 'r') as f:
        lines = f.readlines()
    
    with open('test.txt', 'w') as f:
        for i in range(len(lines)):
            f.write(lines[i])
            if 'eq' in arg:
                f.write(lines[i][:-5] + '_eq' + lines[i][-5:])
            if 'stretch' in arg:  
                f.write(lines[i][:-5] + '_stretch' + lines[i][-5:])
            if 'r90' in arg:
                f.write(lines[i][:-5] + '_r90' + lines[i][-5:])
            if 'r180' in arg:
                f.write(lines[i][:-5] + '_r180' + lines[i][-5:])
            if 'r270' in arg:
                f.write(lines[i][:-5] + '_r270' + lines[i][-5:])
            if 'flip' in arg:
                f.write(lines[i][:-5] + '_flip' + lines[i][-5:])
            if 'blur' in arg:
                f.write(lines[i][:-5] + '_blur' + lines[i][-5:])    
            if 'gauss' in arg:
                f.write(lines[i][:-5] + '_gauss' + lines[i][-5:])
            if 'motion' in arg:
                f.write(lines[i][:-5] + '_motion' + lines[i][-5:])
    """ 
split_aug(argument)
