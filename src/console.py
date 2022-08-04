from im_processing import ImProcessing
import data_processing as dp

import traceback
import sys
import os
import time
import pandas as pd

import csv
import cv2


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
    # Buscamos todas las monedas
    for pth in img_paths:
        try:
            # Si se espacifica que hayan multiples monedas en la imagen
            _, nc = dp.getClass(pth)
            # Busca una moneda en la imaegn
            findCoin(pth, nc)
        except cv2.error:  # Tratamos cuando no encuentra el archivo
            traceback.print_exc()
            print("Archivo no encontrado")


def findCoin(img, ncoins=1):
    # Extraemos datos de la moneda
    data = ImProcessing.extractData(img, ncoins)

    # -- DEBUG --
    # Imprime los datos por pantalla
    # print(data)

    # -- POR COMPLETAR --
    # Añadir identificacion por aprendizaje


def addCoins(img_paths):
    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if img_paths == []:
        img_paths = readPaths()
        # Si faltan argumentos los pedimos
        if img_paths == []:
            print(
                "Añade información de la moneda:\n"
                + "add PATH_1 PATH_2 ... PATH_N\n"
                + "Se pueden seleccionar carpetas"
            )
            return

    try:
        start_time = time.time()

        img_paths = dp.getFilesInFolders(img_paths)
        # Extraemos datos de todas la imagenes especificadas
        data = None
        for pth in img_paths:
            _, nc = dp.getClass(pth)

            d = ImProcessing.extractData(pth, nc)

            if data == None and d != None:
                data = d
            else:
                for k in d:
                    data[k] += d[k]

        # Escribimos los datos en el archivo csv
        # print(data)
        # writeCSV(pd.DataFrame.from_dict(data))
        pd.DataFrame.from_dict(data).to_csv(dp.FILE_CSV)
        end_time = time.time()
        print(f"TIME ELAPSED: {end_time-start_time}")
        # -- POR COMPLETAR --

    except cv2.error:  # Tratamos cuando no encuentra el archivo
        traceback.print_exc()
        print("Archivo no encontrado")
    except ValueError:  # Error en el formato del nombre del archivo
        traceback.print_exc()
        print("Error: Seguir formato -> add PATH_1,PATH_2,...PATH_N")


def writeCSV(data: pd.DataFrame):
    # Introduce los datos en file_csv
    data.to_csv(dp.FILE_CSV)
    # with open(dp.FILE_CSV, mode="w") as f:
    #     writer = csv.writer(f)
    #     # Atributos
    #     writer.writerow(data.keys())
    #     # Valores
    #     r = [len(v) for v in data.values()]
    #     writer.writerows(zip(*data.values()))


def setInfo(args):
    if len(args) < 2:
        print("Error: Seguir formato -> set-info CLASS {dict Info}")
        return

    classId = args[0]
    info = args[1]

    with open(dp.FILE_COIN_INFO, "a") as f:
        f.write(f"{classId}:{info}\n")


def testData(img_paths):
    if img_paths == []:
        img_paths = readPaths()
        # Si faltan argumentos los pedimos
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

        image = cv2.imread(pth)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cropped = ImProcessing.cropCircle(image, nc)

        img = pth.split("\\")[-1]
        img = img.split("/")[-1]
        pth = f"{pth[:-len(img)-1]}_CROPPED_TEST"
        if not os.path.exists(pth):
            os.makedirs(pth)
        img = img.split(".")
        for i, cr in enumerate(cropped):
            print(f"{pth}/{img[0]}_{i}.{img[1]}")
            cv2.imwrite(f"{pth}/{img[0]}_{i}.{img[1]}", cr)
            # edges = ImProcessing.edgesInside(cr)
            # cv2.imwrite(
            #     f'{dp.OUT_FOLDER}/EDGES_{img[0]}_{i}.{img[1]}', cv2.resize(edges, (255, 255)))


def invalidCommand(_):
    print("Comando no existe")


def processCommand():
    # Procesador de Comandos
    # Obtenemos el comando
    command = sys.argv[1:]
    if command == []:  # Si no se escribe comando
        print("Comando necesario")
        return

    # Diccionario con los metodos de cada comando
    select_action = {
        "img": setImg,
        "find": findCoins,
        "add": addCoins,
        "set-info": setInfo,
        "test-data": testData,
    }
    # Accionador del metodo asignado al comando ejecutado
    select_action.get(command[0], invalidCommand)(command[1:])


if __name__ == "__main__":
    processCommand()
