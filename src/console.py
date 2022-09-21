from im_processing import *
from config_params import *
from im_classifier import extractData
from machine_learning import add_to_dataset, find_from_dataset
import data_processing as dp

import traceback
import sys
import os
import time
import pandas as pd
import json

import cv2
import sklweka.jvm as jvm


def readPaths():

    try:
        # Leemos el fichero con los paths
        with open(dp.FILE_PATHS, "r") as f:
            lines = f.readlines()
            # Lista de paths de imagenes
            coin_imgs = [line[:-1] for line in lines]
            return coin_imgs
    # Si no esta creado el archivo con los paths devuelve una lista vacía
    except FileNotFoundError:
        return []


def writePaths(paths: list[str], mode="w"):
    # Escribe en el fichero todos los paths de archivos o carpetas
    with open(dp.FILE_PATHS, mode) as f:
        for line in paths:
            f.write(f"{line}\n")


def setImg(args):

    if args == []:  # Si no hay argumentos
        coin_imgs = readPaths()  # Lee file_paths
        if coin_imgs == []:  # Si no hay paths especificados
            print(
                "Escribe img PATH para seleccionar la imagen de la moneda: "
                + "find PATH_1 PATH_2 ... PATH_N\n"
                + "OPCIONES: --a (Añadir a la lista), --w (Escribir encima de la lista)\n"
                + "Se pueden seleccionar carpetas"
            )

        else:
            # Muestra los paths especificados anteriormente
            print("Paths seleccionados: " + ", ".join(coin_imgs))
    else:
        # Procesa las opciones si las hay
        if args[0].startswith("--"):
            if args[0].endswith("a"):  # Si opcion --a añade en filepath
                writePaths(args[1:], "a")
            else:  # Si opcion --w o por defecto sobreescribe en filepath
                writePaths(args[1:], "w")
        else:
            writePaths(args)


# Busca todas las imágenes introducidas
def findCoins(img_paths):

    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if img_paths == []:
        img_paths = readPaths()
        # Si faltan argumentos los pedimos
        if img_paths == []:
            print(
                "Selecciona la o las imagenes a identificar:\n"
                + "find PATH_1 PATH_2 ... PATH_N\n"
                + "Se pueden seleccionar carpetas"
            )
            return

    img_paths = dp.getFilesInFolders(img_paths)
    result = "["

    # Abrirmos la maquina virtual de java
    jvm.start()
    # Buscamos todas las monedas
    for pth in img_paths:
        try:
            # Si se espacifica que hayan multiples monedas en la imagen
            _, nc = dp.getClass(pth)
            # Busca una moneda en la imaegn
            info = findCoin(pth, nc)
            if info:
                # print(
                #     "La moneda en la imagen "
                #     + pth.split("/")[-1]
                #     + f" equivale a {info}"
                # )
                result += str(info) + ", "

            else:
                print(f"No se ha encontrado la moneda en la imágen {pth}")

        except cv2.error:  # Tratamos cuando no encuentra el archivo
            traceback.print_exc()
            print("Archivo no encontrado")
    print(result + "]")


def findCoin(img, ncoins=1):
    # Extraemos datos de la moneda
    im_data = extractData(img, ncoins)

    # -- DEBUG --
    # Imprime los datos por pantalla
    # print(data)

    # -- POR COMPLETAR --
    # Añadir identificacion por aprendizaje
    data = dp.initData()
    for key in data:
        if im_data.get(key) is not None:
            data[key].append(im_data[key])
        else:
            data[key].append(0)

    df = pd.DataFrame.from_dict(data)
    coin_class = find_from_dataset(df)
    info = dp.get_coin_info(coin_class)

    if not info:
        print(f"La clase es {coin_class}, pero no hay infomación de esta moneda")

    return info


def get_class(data):
    return "TEST"


def addCoins(img_paths):
    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if img_paths == []:
        img_paths = readPaths()
        # Si faltan argumentos los pedimos
        if img_paths == []:
            print(
                "Añade información de la moneda:\n"
                + "add PATH_1 PATH_2 ... PATH_N (CLASS_NAME)\n"
                + "Se pueden seleccionar carpetas"
            )
            return

    same_class = False
    if not os.path.exists(img_paths[-1]):
        same_class = True
        class_id = img_paths.pop(-1)

    try:
        start_time = time.time()

        # Abrirmos la maquina virtual de java
        jvm.start()
        img_paths = dp.getFilesInFolders(img_paths)
        # Extraemos datos de todas la imagenes especificadas
        data = dp.initData()
        for pth in img_paths:
            if same_class:
                _, nc = dp.getClass(pth)
            else:
                class_id, nc = dp.getClass(pth)

            im_data = extractData(pth, nc)
            im_data["CLASS"] = class_id

            for key in data:
                if im_data.get(key) is not None:
                    data[key].append(im_data[key])
                else:
                    data[key].append(0)

        # Escribimos los datos en el archivo arff
        # print(data)

        df = pd.DataFrame.from_dict(data)
        add_to_dataset(df)

        end_time = time.time()
        print(f"TIME ELAPSED: {round(end_time-start_time)}")
        # -- POR COMPLETAR --

    except cv2.error:  # Tratamos cuando no encuentra el archivo
        traceback.print_exc()
        print("Archivo no encontrado")
    except ValueError:  # Error en el formato del nombre del archivo
        traceback.print_exc()
        print("Error: Seguir formato -> add PATH_1,PATH_2,...PATH_N")


def setInfo(args):
    if len(args) < 2:
        print("Error: Seguir formato -> set-info CLASS NAME YEAR DESCRIPTION URL")
        return

    classId = args[0]
    info = {"NAME": args[1], "YEAR": args[2], "DESCRIPTION": args[3], "URL": args[4]}

    with open(dp.FILE_COIN_INFO, "r") as f:
        data = json.load(f)
    data.update({classId: info})
    with open(dp.FILE_COIN_INFO, "w") as f:
        json.dump(data, f)

    print(f"Añadida la información {info} a la clase {classId}")


def testData(img_paths):

    if img_paths == []:
        print(
            "Comprueba el funcionamiento del algoritmo de estracción de datos:\n"
            + "test-data PATH_1 PATH_2 ... PATH_N\n"
            + "Se pueden seleccionar carpetas"
        )
        return

    img_paths = dp.getFilesInFolders(img_paths)
    for pth in img_paths:
        print(f"Testing: {pth}")
        # Si hay multiples monedas en la imagen debe estar indicado con M
        _, nc = dp.getClass(pth)

        img_array = np.fromfile(pth, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cropped = cropCircles(image, nc)

        img = pth.split("\\")[-1]
        img = img.split("/")[-1]
        pth = f"{pth[:-len(img)-1]}_TESTED"
        if not os.path.exists(pth):
            os.makedirs(pth)
        img = img.split(".")
        for i, cr in enumerate(cropped):
            filename = f"{pth}/{img[0]}_{i}.{img[1]}"
            print(filename)
            is_success, im_buf_arr = cv2.imencode(".jpg", cr)
            im_buf_arr.tofile(filename)

            # cv2.imwrite(
            #     f'{dp.OUT_FOLDER}/EDGES_{img[0]}_{i}.{img[1]}', cv2.resize(edges, (255, 255)))


def invalidCommand(_):
    print("Comando no existe")


def processCommand():
    # Procesador de Comandos
    if len(sys.argv) < 2:  # Si no se escribe comando
        print("Comando necesario")
        return
    # Obtenemos el comando
    _, command, *args = sys.argv

    # Diccionario con los metodos de cada comando
    select_action = {
        "img": setImg,
        "find": findCoins,
        "add": addCoins,
        "set-info": setInfo,
        "test-data": testData,
    }
    # Accionador del metodo asignado al comando ejecutado
    select_action.get(command, invalidCommand)(args)


if __name__ == "__main__":
    try:
        processCommand()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
