import csv
from pathlib import Path

import pandas as pd
from transformers import pipeline
from loadSaveData import loadRAW

def truncate_text(text, max_length=512):
    """
    Trunca el texto si excede la longitud máxima permitida.
    """
    return text[:max_length]


def isPossibleToSplit(totalInstances, numTrainInstances, numTestInstances):
	return (numTrainInstances + numTestInstances) <= totalInstances

'''
##############################################################################		
############################ REDUCCION TRAIN #################################		
##############################################################################		
'''

def reduceDatasetTrain(path, numTrainInstances):
	data = loadRAW(path)

	trainDataset = data.head(n=numTrainInstances)

	trainDataset.to_csv(f'..\Datasets\Suicide_Detection_{numTrainInstances}.csv', index=False)

	return trainDataset

def reduceDatasetTestTrain(path, numTrainInstances, numTestInstances):
	data = loadRAW(path)

	if isPossibleToSplit(len(data), numTrainInstances, numTestInstances):
		trainDataset = data.head(n=numTrainInstances)
		testDataset = data.tail(n=numTestInstances)

		trainDataset.to_csv(f'..\Datasets\Suicide_Detection_train{numTrainInstances}(test{numTestInstances}).csv', index=False)
		testDataset.to_csv(f'..\Datasets\Suicide_Detection_test{numTestInstances}(train{numTrainInstances}).csv', index=False)

		return trainDataset, testDataset

'''
##############################################################################		
############################ REDUCCION TEST ##################################		
##############################################################################		
'''
def crearMiniTests(path, nTest):
	df = pd.read_csv(path)
	tamano_parte = len(df) // nTest
	partes = [df.iloc[i:i + tamano_parte] for i in range(0, len(df), tamano_parte)]
	for i, parte in enumerate(partes):
		parte.to_csv(f'..\Datasets\Suicide_Detection_test{i}_{tamano_parte}.csv', index=False)


def takeTestAfterFirst10000(path, numTrainInstances):
	data = loadRAW(path)

	trainDataset = data.tail(n=100000)
	testDataset = trainDataset.head(n=numTrainInstances)

	testDataset.to_csv(f'..\Datasets\Suicide_Detection_Test_{numTrainInstances}.csv', index=False)

	return testDataset

'''
##############################################################################		
##################### REDUCCION CONTANDO SENTIMIENTOS ########################		
##############################################################################		
'''

def selectDatasetSinXEmociones(path, emotionsToQuit, numInstances):
	archivo_salida = f'..\Datasets\Suicide_Detection_{numInstances}_Sin_{emotionsToQuit}.csv'

	classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
						  return_all_scores=True)

	if not Path(archivo_salida).is_file():
		with open(archivo_salida, mode='w', encoding='utf-8') as file:
			writer = csv.writer(file)
			writer.writerow(['id', 'text', 'class'])  # Agrega los encabezados según tu estructura

	data = loadRAW(path)

	k = 0
	kontTotal = 0
	kontSuicide = 0
	kontNonSuicide = 0

	while kontTotal < numInstances:
		instancia = data.iloc[k]

		id_value = instancia['id']
		text_value = instancia['text']
		class_value = instancia['class']
		print(f'INSTANCIA: {instancia}')

		truncated_instance = truncate_text(text_value)
		prediction = classifier(truncated_instance, )[0]
		pjson = {}
		for emotion in prediction:
			pjson[emotion['label']] = emotion['score']

		emocionDominante = max(pjson, key=pjson.get)
		print(f'EMOCION: {emocionDominante}')

		if emocionDominante not in emotionsToQuit:
			if class_value == 'suicide' and kontSuicide < int(numInstances/2):
				kontSuicide += 1
				kontTotal += 1
				with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
					writer = csv.writer(file)
					writer.writerow([id_value, text_value, class_value])
				print(f'Se añade\n')

			elif class_value == 'non-suicide' and kontNonSuicide < int(numInstances/2):
				kontNonSuicide += 1
				kontTotal += 1
				with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
					writer = csv.writer(file)
					writer.writerow([id_value, text_value, class_value])
				print(f'Se añade\n')

		k += 1

def selectDatasetEmotionBalanced(path, numInstances):
	archivo_salida = f'..\Datasets\Suicide_Detection_{numInstances}_Balanceado.csv'

	classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
						  return_all_scores=True)

	if not Path(archivo_salida).is_file():
		with open(archivo_salida, mode='a', encoding='utf-8') as file:
			writer = csv.writer(file)
			writer.writerow(['id', 'text', 'class'])  # Agrega los encabezados según tu estructura

			data = loadRAW(path)

			k = 0
			kontTotal = 0
			kontSuicide = 0
			kontNonSuicide = 0
			kontEmociones = {'sadness': 0, 'joy': 0, 'love': 0, 'anger': 0, 'fear': 0, 'surprise': 0}

			while kontTotal < numInstances:
				instancia = data.iloc[k]

				id_value = instancia['id']
				text_value = instancia['text']
				class_value = instancia['class']

				truncated_instance = truncate_text(text_value)
				prediction = classifier(truncated_instance, )[0]
				pjson = {}
				for emotion in prediction:
					pjson[emotion['label']] = emotion['score']

				emocionDominante = max(pjson, key=pjson.get)

				if emocionDominante.__eq__('surprise') or emocionDominante.__eq__('love'):
					if class_value == 'suicide':
						kontSuicide += 1
						kontTotal += 1
						with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
							writer = csv.writer(file)
							writer.writerow([id_value, text_value, class_value])

					else:
						kontNonSuicide += 1
						kontTotal += 1
						with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
							writer = csv.writer(file)
							writer.writerow([id_value, text_value, class_value])
					kontEmociones[emocionDominante] += 1

				elif kontEmociones[emocionDominante] < int(numInstances*0.22):
					if class_value == 'suicide' and kontSuicide < int(numInstances/2):
						kontSuicide += 1
						kontEmociones[emocionDominante] += 1
						kontTotal += 1
						with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
							writer = csv.writer(file)
							writer.writerow([id_value, text_value, class_value])

					elif class_value == 'non-suicide' and kontNonSuicide < int(numInstances/2):
						kontNonSuicide += 1
						kontEmociones[emocionDominante] += 1
						kontTotal += 1
						with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
							writer = csv.writer(file)
							writer.writerow([id_value, text_value, class_value])
				k += 1
				if kontTotal % 100 == 0:
					print(kontTotal)



def selectDatasetSoloEmociones(path, numInstances):
	archivo_salida = f'..\Datasets\Suicide_Detection_{numInstances}_Balanceado_SinContarClases.csv'

	classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
						  return_all_scores=True)

	if not Path(archivo_salida).is_file():
		with open(archivo_salida, mode='a', encoding='utf-8') as file:
			writer = csv.writer(file)
			writer.writerow(['id', 'text', 'class'])  # Agrega los encabezados según tu estructura

	data = loadRAW(path)

	k = 0
	kontTotal = 0
	kontEmociones = {'sadness': 0, 'joy': 0, 'love': 0, 'anger': 0, 'fear': 0, 'surprise': 0}

	while kontTotal < numInstances:
		instancia = data.iloc[k]

		id_value = instancia['id']
		text_value = instancia['text']
		class_value = instancia['class']

		truncated_instance = truncate_text(text_value)
		prediction = classifier(truncated_instance, )[0]
		pjson = {}
		for emotion in prediction:
			pjson[emotion['label']] = emotion['score']

		emocionDominante = max(pjson, key=pjson.get)

		if kontEmociones[emocionDominante] < int(numInstances * 0.22):
			kontEmociones[emocionDominante] += 1
			kontTotal += 1
			with open(archivo_salida, mode='a', newline='', encoding='utf-8') as file:
				writer = csv.writer(file)
				writer.writerow([id_value, text_value, class_value])

		k += 1
		if kontTotal % 100 == 0:
			print(kontTotal)





if __name__ == '__main__':
	# crearMiniTests('../Datasets/Suicide_Detection_2000_Balanceado.csv', 5)
	# selectDatasetEmotionBalanced('../Datasets/Suicide_Detection.csv', 10000)
	# takeTestAfterFirst10000('../Datasets/Suicide_Detection.csv', 100)
	# selectDatasetSoloEmociones('../Datasets/Suicide_Detection.csv', 10000)
	pass
