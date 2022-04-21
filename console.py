import traceback
import pandas as pd
from imProcessing import ImProcessing
import sys
import os
import csv

file_paths = "path_to_data.txt"
file_csv = "DataSet.csv"
col_list = ['ID', 'OCR_1', 'OCR_2', 'OCR_3', 'HU_1', 'HU_2']

def readPaths():
    try:
        with open(file_paths, 'r') as f:
            lines = f.readlines()
            coin_imgs = []
            for p in lines:
                if p[-2]== '/' or p[-2] == '\\':
                    coin_imgs += map(p[:-1].__add__,os.listdir(p[:-1]))
                else:
                    coin_imgs.append(p[:-1])

            return coin_imgs
    except FileNotFoundError:
        return []

def writePaths(paths, mode='w'):
    with open(file_paths, mode) as f:
        for line in paths:
            f.write(f'{line}\n')

# Identifica la dirección de la imagen
def setImg(args):
    if args == []:
        coin_imgs = readPaths()
        if coin_imgs == []:
            print("Escribe img PATH para seleccionar la imagen de la moneda: "+
                    "find PATH_1 PATH_2 ... PATH_N\n" +
                    "OPCIONES: --a (Añadir a la lista), --w (Escribir encima de la lista)\n"+
                    "Se pueden seleccionar carpetas")

        else:
            print("Imagenes seleccionadas: " + ', '.join(coin_imgs))
    else:
        if args[0][:2] == '--':
            if args[0][2] == 'a':
                writePaths(args[1:], 'a')
            else:
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
    # Identifica todas las imagenes especificadas
    for pth in coin_imgs:
        try:
            # Si hay multiples monedas en la imagen debe estar indicado con M
            img = pth.split('/')[-1]
            if img[0] == 'M':
                nc = int(img.split('_')[0][1:])
                findCoin(pth, nc)
            else:
                findCoin(pth)
        except FileNotFoundError:  # Tratamos cuando no encuentra el archivo
            print("Archivo no encontrado")

def findCoin(img, ncoins = 1):
    # Extraemos datos de la moneda
    data = ImProcessing.extractData(img, ncoins)

    for d in data:
        ocr, hu, keyp = d[1], d[2], d[3]
        print(f'OCR: {ocr}')
        print(f'Hu: {hu}')
        print(f'KeyP: {keyp}')

    # -- POR COMPLETAR --
    # Añadir identificacion por aprendizaje



def addCoins(args):
    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if args == []:
        print("Añade información de la moneda:\n" +
                "add PATH_1 PATH_2 ... PATH_N {Info}\n" +
                "Se pueden seleccionar carpetas")
        return

    # Separa imagenes de atributos
    try:
        if len(args) == 1:
            imgs = readPaths()
            info = args[0]
        else:
            imgs = args[0:-1]
            info = args[-1]
        # Lee la información obtenida de cada moneda

        # Obtenemos los atributos
        info = info[1:-1].split(",")
        info_dict = {}
        # Los metemos en un diccionario
        for atb in info:
            key_value = atb.split(":")
            info_dict.update({key_value[0]: key_value[1]})
        print(info_dict)

        # Extraemos datos de todas la imagenes especificadas
        to_csv = []
        for pth in imgs:
            # Si hay multiples monedas en la imagen debe estar indicado con M
            img = pth.split('/')[-1]
            if img[0] == 'M':
                nc = int(img.split('_')[0][1:])
                print(img)
                id = int(img.split('_')[1][:])
                to_csv += getData(pth, ncoins=nc, id=id)
            else:
                to_csv += getData(pth, id=int(img.split('_')[0]))
            
        writeCSV(to_csv)
        # -- POR COMPLETAR --

    except FileNotFoundError:  # Tratamos cuando no encuentra el archivo
        print("Archivo no encontrado")
    except:
        traceback.print_exc()
        print("Error: Seguir formato -> PATH_1,PATH_2,...PATH_N{Info}")

def getData(img, id, ncoins = 1):
    # Extraemos datos de la moneda
    data = []
    data += ImProcessing.extractData(img, ncoins)

    to_csv = []
    for d in data:
        ocr, hu, keyp = d[1], d[2], d[3]
        # print(f'OCR: {ocr}')
        # print(f'Hu: {hu}')
        # print(f'KeyP: {keyp}')
        row = [str(id)]
        row += ocr[:min(len(ocr),3)]
        for i in range(len(row), 4):
            row.append('')
        row += str(hu[0]),str(hu[1])
        to_csv.append(row)

    return to_csv


def writeCSV(to_csv):
    # ['ID', 'OCR_1', 'OCR_2', 'OCR_3', 'HU_1', 'HU_2']
    with open(file_csv, mode='w') as f:
        csvwriter = csv.writer(f) 
        csvwriter.writerow(col_list) 
        csvwriter.writerows(to_csv)

def invalidCommand(args):
    print("Comando no existe")


def processCommand():
    command = sys.argv[1:]
    if command == []:
        print("Comando necesario")
        return

    select_action = {
        "img": setImg,
        "find": findCoins,
        "add": addCoins
    }
    select_action.get(command[0], invalidCommand)(command[1:])

if __name__ == "__main__":
    processCommand()
    