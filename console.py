from audioop import error
import traceback
import pandas as pd
from imProcessing import ImProcessing
import sys
import os
import csv
import cv2

file_paths = "path_to_data.txt"
file_csv = "DataSet.csv"
col_list = ['ID', 'OCR_1', 'OCR_2', 'OCR_3', 'HU_1', 'HU_2']


def readPaths():

    try:
        # Leemos el fichero con los paths
        with open(file_paths, 'r') as f:
            lines = f.readlines()
            coin_imgs = []  # Lista de paths de imagenes
            for p in lines:
                # Si acaba es direcctorio
                if os.path.isdir(p[:-1]):
                    # Introducimos todos los ficheros de la carpeta junto con su path a la lists
                    coin_imgs += [os.path.join(p[:-1], file)
                                  for file in os.listdir(p[:-1])]
                else:
                    # Introducimos el archivo a la lista
                    coin_imgs.append(p[:-1])

            return coin_imgs
    # Si no esta creado el archivo con los paths devuelve una lista vacía
    except FileNotFoundError:
        return []


def writePaths(paths, mode='w'):
    # Escribe en el fichero todos los paths de archivos o carpetas
    with open(file_paths, mode) as f:
        for line in paths:
            f.write(f'{line}\n')


def setImg(args):

    if args == []:  # Si no hay argumentos
        coin_imgs = readPaths()  # Lee file_paths
        if coin_imgs == []:  # Si no hay paths especificados
            print("Escribe img PATH para seleccionar la imagen de la moneda: " +
                  "find PATH_1 PATH_2 ... PATH_N\n" +
                  "OPCIONES: --a (Añadir a la lista), --w (Escribir encima de la lista)\n" +
                  "Se pueden seleccionar carpetas")

        else:
            # Muestra los paths especificados anteriormente
            print("Imagenes seleccionadas: " + ', '.join(coin_imgs))
    else:
        # Procesa las opciones si las hay
        if args[0][:2] == '--':
            if args[0][2] == 'a':  # Si opcion --a añade en filepath
                writePaths(args[1:], 'a')
            else:  # Si opcion --w o por defecto sobreescribe en filepath
                writePaths(args[1:], 'w')
        else:
            writePaths(args)


# Busca todas las imágenes introducidas
def findCoins(coin_imgs):
    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if coin_imgs == []:
        coin_imgs = readPaths()
        # Si faltan argumentos los pedimos
        if coin_imgs == []:
            print("Selecciona la o las imagenes a identificar:\n" +
                  "find PATH_1 PATH_2 ... PATH_N\n" +
                  "Se pueden seleccionar carpetas")
            return

    # Buscamos todas las monedas
    for pth in coin_imgs:
        try:
            # Si acaba es direcctorio
            if os.path.isdir(pth):
                coin_imgs.remove(pth)
                # Introducimos todos los ficheros de la carpeta junto con su path a la lists
                coin_imgs += [os.path.join(pth, file)
                              for file in os.listdir(pth)]
                continue

            # Obtenemos el nombre del archivo
            img = pth.split('\\')[-1]
            # Si se espacifica que hayan multiples monedas en la imagen
            if img[0] == 'M':
                # Leemos el numero de monedas especificados
                nc = int(img.split('_')[0][1:])
                # Busca nc monedas en la imagen
                findCoin(pth, nc)
            else:
                # Busca una moneda en la imaegn
                findCoin(pth)
        except cv2.error:  # Tratamos cuando no encuentra el archivo
            print("Archivo no encontrado")


def findCoin(img, ncoins=1):
    # Extraemos datos de la moneda
    data = ImProcessing.extractData(img, ncoins)

    # -- DEBUG --
    # Imprime los datos por pantalla
    print(data)

    # -- POR COMPLETAR --
    # Añadir identificacion por aprendizaje


def addCoins(args):
    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if args == []:
        print("Añade información de la moneda:\n" +
              "add PATH_1 PATH_2 ... PATH_N {Info}\n" +
              "Se pueden seleccionar carpetas")
        return

    try:
        # Separa imagenes de atributos
        if len(args) == 1:
            imgs = readPaths()
            info = args[0]
        else:
            imgs = args[0:-1]
            info = args[-1]

        # Obtenemos los atributos
        info = info[1:-1].split(",")
        info_dict = {}
        # Los metemos en un diccionario
        for atb in info:
            key_value = atb.split(":")
            info_dict.update({key_value[0]: key_value[1]})

        # Diccionario para introducir en el archivo csv
        data = {"ID": [],
                "HU_1": [],
                "HU_2": [],
                "OCR_1": [],
                "OCR_2": [],
                "OCR_3": [],
                }

        # Extraemos datos de todas la imagenes especificadas
        for pth in imgs:
            if os.path.isdir(pth):
                imgs.remove(pth)
                # Introducimos todos los ficheros de la carpeta junto con su path a la lists
                imgs += [os.path.join(pth, file)
                         for file in os.listdir(pth)]
                continue

            # Obtenemos el nombre del archivo
            img = pth.split('\\')[-1]
            # Si hay multiples monedas en la imagen debe estar indicado con M
            if img[0] == 'M':
                # Leemos la ID y el numero de monedas especificados
                nc = int(img.split('_')[0][1:])
                # Introducimos los datos obtenidos a la lista
                d = ImProcessing.extractData(pth, nc)
            else:
                d = ImProcessing.extractData(pth)

            for k in data:
                data[k] += d[k]

        # Escribimos los datos en el archivo csv
        writeCSV(data)
        # -- POR COMPLETAR --

    except cv2.error:  # Tratamos cuando no encuentra el archivo
        print("Archivo no encontrado")
    except ValueError:  # Error en el formato del nombre del archivo
        print("Error: Seguir formato -> PATH_1,PATH_2,...PATH_N{Info}")


def writeCSV(data):
    # Introduce los datos en file_csv
    with open(file_csv, mode='w') as f:
        writer = csv.writer(f)
        # Atributos
        writer.writerow(data.keys())
        # Valores
        writer.writerows(zip(*data.values()))


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
        "add": addCoins
    }
    # Accionador del metodo asignado al comando ejecutado
    select_action.get(command[0], invalidCommand)(command[1:])


if __name__ == "__main__":
    processCommand()
