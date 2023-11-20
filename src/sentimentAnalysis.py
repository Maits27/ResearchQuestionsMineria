import json
from tqdm import tqdm

from transformers import pipeline
from transformers import AutoTokenizer


def truncate_text(text, max_length=512):
    """
    Trunca el texto si excede la longitud máxima permitida.
    """
    return text[:max_length]
def getArgmaxSentimentAndClass(classData, nInstances):
    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open(f'../out/emociones/emocionesEnumeradas_{nInstances}.json', 'r', encoding='utf-8') as json_file:
            emociones_array = json.load(json_file)
        emocionesDominantes = []
        for i, text in enumerate(emociones_array):
            nuevaEmocionDom = {}
            nuevaEmocionDom['instancia'] = text['instancia']
            nuevaEmocionDom['emocion'] = max(text['emociones'], key=text['emociones'].get)
            nuevaEmocionDom['clase'] = classData[i][1]
            emocionesDominantes.append(nuevaEmocionDom)

        with open(f'../out/emociones/emocionesDominantesConClase_{nInstances}.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return 0

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return -1
def getArgmaxSentiment(nInstances):
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

        with open(f'..\out\emociones\emocionesDominantes_{nInstances}.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return res

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return []


def getMultiArgmaxSentiment(fiabilidad=0.8):

    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open('..\out\emociones\emocionesEnumeradas.json', 'r', encoding='utf-8') as json_file:
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

        with open('..\out\emociones\emocionesDominantes.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return res

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return []
# def getMultiArgmaxSentiment1Instance(instancia, fiabilidad=0.8):
#     classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
#                           return_all_scores=True)
#     truncated_instance = truncate_text(instancia)
#     prediction = classifier(truncated_instance, )[0]
#     res = []
#     pjson = {}
#     for emotion in prediction:
#         pjson[emotion['label']] = emotion['score']
#
#     for emocion, valor in pjson.items():
#         if valor > fiabilidad:
#             res.append(emocion)
#
#     return res




def getSentiment(rawData):
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

    with open('..\out\emociones\emocionesEnumeradas.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    getArgmaxSentiment()
    return predictions
def getSentimentAndClass(classData):
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

    with open('..\out\emociones\emocionesEnumeradasConClase.json', 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    return predictions
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
