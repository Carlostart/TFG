import os
import pickle
import arff

import data_processing as dp

from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_arff, to_nominal_labels
from sklweka.preprocessing import MakeNominal
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def add_to_dataset(data_to_add):
    # Necesario abrir maquina virtual de java
    # para utilizar los clasificadores de Weka
    # jvm.start()

    classes = []
    all_data = []
    # Si ya existe un modelo lo leemos para añadir todos los datos
    if os.path.exists(dp.FILE_ARFF):
        with open(dp.FILE_ARFF, "r", encoding="utf8") as f:
            file_ = f.read()

        ds_dict = arff.loads(file_)

        classes = [d[-1] for d in ds_dict["data"]]
        all_data = ds_dict["data"]

    classes += data_to_add["CLASS"].tolist()

    attributes = [(j, "NUMERIC") for j in data_to_add.columns[:-1]]
    attributes += [(data_to_add.columns[-1], list(set(classes)))]

    all_data += data_to_add.values.tolist()

    arff_dic = {
        "attributes": attributes,
        "data": all_data,
        "relation": "Coin Clasificator",
        "description": "",
    }

    with open(dp.FILE_ARFF, "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)

    X, y, _ = load_arff(dp.FILE_ARFF, class_index="last")
    y = to_nominal_labels(y)

    simple_logistic = WekaEstimator(
        classname="weka.classifiers.functions.SimpleLogistic",
        options=["-I", "0", "-P", "-M", "500", "-H", "50", "-W", "0.0"],
    )

    print(simple_logistic.to_commandline())
    simple_logistic.fit(X, y)

    with open(dp.FILE_MODEL, "wb") as f:
        pickle.dump(simple_logistic, f)

    # jvm.stop()


def find_from_dataset(atributes):
    # jvm.start()

    with open(dp.FILE_MODEL, "rb") as f:
        simple_logistic = pickle.load(f)
        # print(simple_logistic)

    # print(atributes.values.tolist())
    scores = simple_logistic.predict(atributes.values.tolist())
    probas = simple_logistic.predict_proba(atributes.values.tolist())

    # print(scores[0])

    # jvm.stop()

    return scores[0]
