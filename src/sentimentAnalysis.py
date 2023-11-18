import json
from tqdm import tqdm

from transformers import pipeline
from transformers import AutoTokenizer


def truncate_text(text, max_length=512):
    """
    Trunca el texto si excede la longitud máxima permitida.
    """
    return text[:max_length]
def getArgmaxSentimentAndClass(classData):
    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open('emocionesEnumeradas.json', 'r', encoding='utf-8') as json_file:
            emociones_array = json.load(json_file)
        emocionesDominantes = []
        for i, text in enumerate(emociones_array):
            nuevaEmocionDom = {}
            nuevaEmocionDom['instancia'] = text['instancia']
            nuevaEmocionDom['emocion'] = max(text['emociones'], key=text['emociones'].get)
            nuevaEmocionDom['clase'] = classData[i][1]
            emocionesDominantes.append(nuevaEmocionDom)

        with open('emocionesDominantesConClase.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return 0

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return -1
def getArgmaxSentiment():
    try:
        # Lee el array de JSONs desde el archivo con la codificación UTF-8
        with open('emocionesEnumeradas.json', 'r', encoding='utf-8') as json_file:
            emociones_array = json.load(json_file)
        emocionesDominantes = []
        for i, text in enumerate(emociones_array):
            nuevaEmocionDom = {}
            nuevaEmocionDom['instancia'] = text['instancia']
            nuevaEmocionDom['emocion'] = max(text['emociones'], key=text['emociones'].get)
            emocionesDominantes.append(nuevaEmocionDom)

        with open('emocionesDominantes.json', 'w', encoding='utf-8') as json_file:
            json.dump(emocionesDominantes, json_file, indent=2, ensure_ascii=False)

        return 0

    except FileNotFoundError:
        print(f"El archivo  no fue encontrado.")
        return -1


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

    with open('emocionesEnumeradas.json', 'w', encoding='utf-8') as json_file:
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

    with open('emocionesEnumeradasConClase.json', 'w', encoding='utf-8') as json_file:
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
