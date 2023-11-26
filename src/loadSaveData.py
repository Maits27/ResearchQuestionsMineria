import pandas as pd

def loadRAWwithClass(path):
    """
    Devuelve un dataset con texto y clase
    """
    data = pd.read_csv(path)
    return [[instancia[1], instancia[2]] for instancia in data.values]

def loadRAW(path):

    return pd.read_csv(path)
def loadTextRaw(path):
    """
    Devuelve un dataset con solo los textos
    """
    data = pd.read_csv(path)
    return [instancia[1] for instancia in data.values]
def loadClassTextList(path):
    data = pd.read_csv(path)
    return [[instancia[1], instancia[2]] for instancia in data.values]

