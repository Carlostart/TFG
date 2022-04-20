from cv2 import readOpticalFlow
from imProcessing import ImProcessing
import sys

file_paths = "path_to_data.txt"

def readPaths():
    try:
        with open(file_paths, 'r') as f:
            coin_imgs = f.readlines()
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
                    "OPCIONES: --a (Añadir a la lista), --w (Escribir encima de la lista)\n"
                    "Tambien puedes especificar si hay multiples monedas en la imagen:\n" +
                    "find (M)PATH [...]")
        else:
            print("Imagenes seleccionadas:\n" + ''.join(coin_imgs))
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
                    "Tambien puedes especificar si hay multiples monedas en la imagen:\n" +
                    "find (M)PATH [...]")
            return
    # Identifica todas las imagenes especificadas
    for img in coin_imgs:
        try:
            # Si hay multiples monedas en la imagen debe estar indicado con (M)
            if img[0:3] == '(M)':
                findCoin(img[3:], True)
            else:
                findCoin(img, False)
        except FileNotFoundError:  # Tratamos cuando no encuentra el archivo
            print("Archivo no encontrado")

def findCoin(img, multiple):
    print("Identificando imagen: " + img)
    # Extraemos datos de la moneda
    data = ImProcessing.extractData(img, multiple)
    print(data)

    # -- POR COMPLETAR --
    # Añadir identificacion por aprendizaje

def addCoin(coin_imgs):
    # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
    if coin_imgs == []:
        print("Añade información de la moneda:\n" +
                "add PATH_1 PATH_2 ... PATH_N {Info}\n" +
                "Tambien puedes especificar si hay multiples monedas en la imagen:\n" +
                "add (M)PATH{Info}")
        return

    # Separa imagenes de atributos
    if len(coin_imgs) == 1:
        imgs = readPaths()
        info = coin_imgs[0]
    else:
        imgs = coin_imgs[0:-1]
        info = coin_imgs[-1]
    # Lee la información obtenida de cada moneda
    # try:
    # Obtenemos los atributos
    info = info[1:-1].split(",")
    info_dict = {}
    # Los metemos en un diccionario
    for atb in info:
        key_value = atb.split(":")
        info_dict.update({key_value[0]: key_value[1]})
    print(info_dict)

    # Extraemos datos de todas la imagenes especificadas
    img_data = []
    for img in imgs:
        # Si se especifica que hay multiples monedas en la imagen
        if img[0:3] == '(M)':
            img_data.append(
                ImProcessing.extractData(img[3:], True)[0])
        else:
            img_data.append(ImProcessing.extractData(img, False))

    print(img_data)

    # -- POR COMPLETAR --
    # Añadir datos al DataSet del aprendizaje
    # except:
    print("Error: Seguir formato -> PATH_1,PATH_2,...PATH_N{Info}")

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
        "add": addCoin
    }
    select_action.get(command[0], invalidCommand)(command[1:])

if __name__ == "__main__":
    processCommand()
    