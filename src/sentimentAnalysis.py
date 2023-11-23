import json
from tqdm import tqdm

from transformers import pipeline
from transformers import AutoTokenizer

"""
##########################################################################################
############################## OUTPUT DEL TRANSFORMER ####################################
##########################################################################################
LINK: https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion
"""
"""
output:
[[
{'label': 'sadness', 'score': 0.0005138228880241513}, 
{'label': 'joy', 'score': 0.9972520470619202}, 
{'label': 'love', 'score': 0.0007443308713845909}, 
{'label': 'anger', 'score': 0.0007404946954920888}, 
{'label': 'fear', 'score': 0.00032938539516180754}, 
{'label': 'surprise', 'score': 0.0004197491507511586}
]]
"""


def truncate_text(text, max_length=512):
    """
    Trunca el texto si excede la longitud máxima permitida.
    """
    return text[:max_length]

"""
##########################################################################################
########################## SOLO SENTIMIENTO PREDOMINANTE #################################
##########################################################################################
"""
def getArgmaxSentimentAndClass(classData, nInstances):
    """
    Recoje el sentimiento predominante y la clase de cada texto
    """
    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open(f'../out/emociones/emocionesEnumeradas_{nInstances}.json', 'r', encoding='utf-8') as json_file:
            emociones_array = json.load(json_file)
        emocionesDominantes = []
        kontEmo = {'sadness': 0, 'joy': 0, 'love': 0, 'anger': 0, 'fear': 0, 'surprise': 0}
        print(f'Textos de emociones: {len(emociones_array)}')
        print(f'Textos de clase: {len(classData)}')
        for i, text in enumerate(emociones_array):
            nuevaEmocionDom = {}
            nuevaEmocionDom['instancia'] = text['instancia']
            nuevaEmocionDom['emocion'] = max(text['emociones'], key=text['emociones'].get)
            kontEmo[nuevaEmocionDom['emocion']]+=1
            nuevaEmocionDom['clase'] = classData[i][1]
            emocionesDominantes.append(nuevaEmocionDom)

        with open(f'../out/emociones/emocionesDominantesConClase_{len(emocionesDominantes)}.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)
        print(kontEmo)
        return 0

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return -1
def getArgmaxSentiment(nInstances):
    """
    Recoje el sentimiento predominante de cada texto
    """
    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open(f'..\out\emociones\emocionesEnumeradas_{nInstances}.json', 'r', encoding='utf-8') as json_file:
            emociones_array = json.load(json_file)
        emocionesDominantes = []
        res = []
        for i, text in enumerate(emociones_array):
            nuevaEmocionDom = {}
            nuevaEmocionDom['instancia'] = text['instancia']
            nuevaEmocionDom['emocion'] = max(text['emociones'], key=text['emociones'].get)
            res.append(nuevaEmocionDom['emocion'])
            emocionesDominantes.append(nuevaEmocionDom)

        with open(f'..\out\emociones\emocionesDominantes_{len(emocionesDominantes)}.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return res

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return []
"""
##########################################################################################
############################## TODOS LOS SENTIMIENTOS ####################################
##########################################################################################
"""
def getSentiment(rawData):
    """
    Recoje el porcentajde de cada sentimiento de cada texto
    """
    classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
                          return_all_scores=True)
    predictions = []

    for i, instance in enumerate(rawData):
        # Truncar el texto si es demasiado largo
        truncated_instance = truncate_text(instance)

        # Tokenizar el texto
        prediction = classifier(truncated_instance, )[0]
        pjson = {}
        pjson['instancia'] = i
        pjson['emociones'] = {}
        for emotion in prediction:
            pjson['emociones'][emotion['label']] = emotion['score']

        predictions.append(pjson)

    with open(f'..\out\emociones\emocionesEnumeradas_{len(predictions)}.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    with open(f'emocionesEnumeradas_{len(predictions)}.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    return predictions
def getSentimentAndClass(classData):
    """
    Recoje el porcentaje de cada sentimiento y la clase de cada texto
    """
    classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
                          return_all_scores=True)
    predictions = []

    for i, instance in enumerate(classData):
        # Truncar el texto si es demasiado largo
        truncated_instance = truncate_text(instance[0])

        # Tokenizar el texto
        prediction = classifier(truncated_instance, )[0]
        pjson = {}
        pjson['instancia'] = i
        pjson['clase'] = instance[1]
        pjson['emociones'] = {}
        for emotion in prediction:
            pjson['emociones'][emotion['label']] = emotion['score']

        predictions.append(pjson)

    with open(f'..\out\emociones\emocionesEnumeradasConClase_{len(predictions)}.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    return predictions





"""
##########################################################################################
########################## SENTIMIENTOS CON FIABILIDAD X #################################
##########################################################################################
"""
def getMultiArgmaxSentiment(nInst, fiabilidad=0.8):
    """
    Recoje el los sentimientos por encima del threshold  de cada texto
    """
    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open(f'..\out\emociones\emocionesEnumeradas_{nInst}.json', 'r', encoding='utf-8') as json_file:
            emociones_array = json.load(json_file)
        emocionesDominantes = []
        res = []
        for i, text in enumerate(emociones_array):
            nuevaEmocionDom = {}
            nuevaEmocionDom['instancia'] = text['instancia']
            emociones = []
            for emocion, valor in text['emociones'].items():
                if valor > fiabilidad:
                    emociones.append(emocion)
            nuevaEmocionDom['emocion'] = emociones
            res.append(emociones)
            emocionesDominantes.append(nuevaEmocionDom)

        with open(f'..\out\emociones\emocionesDominantes_{len(emocionesDominantes)}.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return res

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return []

"""
##########################################################################################
####################### TODOS LOS SENTIMIENTOS OTRO ROBERTA ##############################
##########################################################################################
"""

def getSentimentRoberta(rawData):
    """
    Recoje el porcentajde de cada sentimiento de cada texto
    """
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    predictions = []

    for i, instance in enumerate(rawData):
        # Truncar el texto si es demasiado largo
        truncated_instance = truncate_text(instance)

        # Tokenizar el texto
        prediction = classifier(truncated_instance, )[0]
        pjson = {}
        pjson['instancia'] = i
        pjson['emociones'] = {}
        for emotion in prediction:
            pjson['emociones'][emotion['label']] = emotion['score']

        predictions.append(pjson)

    with open(f'..\out\emociones\emocionesEnumeradas_{len(predictions)}.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    with open(f'emocionesEnumeradas_{len(predictions)}.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    return predictions

def getSentimentAndClassRoberta(classData):
    """
    Recoje el porcentaje de cada sentimiento y la clase de cada texto
    """
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    predictions = []
    with open(f'..\out\emo\emocionesEnumeradasConClase_{len(predictions)}.json', 'a',
              encoding='utf-8') as json_file:

        for i, instance in enumerate(classData):
            # Truncar el texto si es demasiado largo
            truncated_instance = truncate_text(instance[0])

            # Tokenizar el texto
            prediction = classifier(truncated_instance, )[0]
            pjson = {}
            pjson['instancia'] = i
            pjson['clase'] = instance[1]
            pjson['emociones'] = {}
            for emotion in prediction:
                pjson['emociones'][emotion['label']] = emotion['score']

            predictions.append(pjson)

            json.dump(pjson, json_file, indent=2, ensure_ascii=False)

    return predictions



"""
##########################################################################################
############################## PARA LAS PREDICCIONES ####################################
##########################################################################################
"""
def getSentimentForTest(rawDataClass):
    """
    Recoje el porcentajde de cada sentimiento de cada texto
    """
    classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
                          return_all_scores=True)
    predictions = []

    for i, instance in enumerate(rawDataClass):
        # Truncar el texto si es demasiado largo
        truncated_instance = truncate_text(instance[0])

        # Tokenizar el texto
        prediction = classifier(truncated_instance, )[0]
        pjson = {}
        emociones = {}
        for emotion in prediction:
            emociones[emotion['label']] = emotion['score']
        pjson[i] = {'emocion': max(emociones, key=emociones.get), 'claseReal': instance[1]}

        predictions.append(pjson)

    return predictions

