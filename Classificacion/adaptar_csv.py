import pandas as pd
from sklearn.model_selection import train_test_split
import os

def save_labels_text(data, label_file, text_file):
    
    # Dividir los datos en etiquetas y texto
    etiquetas = data['class']
    textos = data['text']

    # Convertir las etiquetas a valores numéricos (1 para 'suicide' y 0 para 'non-suicide')
    etiquetas_numericas = etiquetas.apply(lambda x: 1 if x == 'suicide' else 0)

    # Guardar etiquetas en un archivo
    with open(label_file, 'w', encoding='utf-8') as file:
        for etiqueta in etiquetas_numericas:
            file.write(f'{etiqueta}\n')

    # Guardar textos en otro archivo
    with open(text_file, 'w', encoding='utf-8') as file:
        for texto in textos:
            file.write(f'{texto}\n')



if __name__ == '__main__':

    # Leer el fichero csv 
    data = pd.read_csv('datasets/Suicide_Detection10000.csv')

    # Train: 75%; Dev: 15%; Test: 10%
    train, test_val = train_test_split(data, test_size=0.25, random_state=42)
    val, test = train_test_split(test_val, test_size=0.4, random_state=42)

    if not os.path.exists('formated'): os.makedirs('formated')

    train.to_csv('formated/train.csv', index=False)
    val.to_csv('formated/val.csv', index=False)
    test.to_csv('formated/test.csv', index=False)


    train_data = pd.read_csv('formated/train.csv')
    val_data = pd.read_csv('formated/val.csv')
    test_data = pd.read_csv('formated/test.csv')
    

    save_labels_text(train_data, 'formated/train_labels.txt', 'formated/train_text.txt')
    save_labels_text(val_data, 'formated/val_labels.txt', 'formated/val_text.txt')
    save_labels_text(test_data, 'formated/test_labels.txt', 'formated/test_text.txt')



