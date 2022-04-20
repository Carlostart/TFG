from imProcessing import ImProcessing


class Console:

    def __init__(self) -> None:
        self.coin_imgs = []

    # Identifica la dirección de la imagen
    def setImg(self, args):
        if args == []:
            if self.coin_imgs == []:
                print("Escribe img PATH para seleccionar la imagen de la moneda: "+
                      "find PATH_1 PATH_2 ... PATH_N\n" +
                      "Tambien puedes especificar si hay multiples monedas en la imagen:\n" +
                      "find (M)PATH [...]")
                
            else:
                print("Imagenes seleccionadas: " + ' '.join(self.coin_imgs))
        else:
            for img in args:
                self.coin_imgs.append(img)
        # print(coin_img)

    # Busca todas las imágenes introducidas
    def findCoins(self, args):
        # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
        if args == []:
            if self.coin_imgs != []:
                args = self.coin_imgs
            # Si faltan argumentos los pedimos
            else:
                print("Selecciona la o las imagenes a identificar:\n" +
                      "find PATH_1 PATH_2 ... PATH_N\n" +
                      "Tambien puedes especificar si hay multiples monedas en la imagen:\n" +
                      "find (M)PATH [...]")
                return
        # Identifica todas las imagenes especificadas
        for img in args:
            try:
                # Si hay multiples monedas en la imagen debe estar indicado con (M)
                if img[0:3] == '(M)':
                    self.findCoin(img[3:], True)
                else:
                    self.findCoin(img, False)
            except FileNotFoundError:  # Tratamos cuando no encuentra el archivo
                print("Archivo no encontrado")

    @staticmethod
    def findCoin(img, multiple):
        print("Identificando imagen: " + img)
        # Extraemos datos de la moneda
        data = ImProcessing.extractData(img, multiple)
        print(data)

        # -- POR COMPLETAR --
        # Añadir identificacion por aprendizaje

    @staticmethod
    def addCoin(args):
        # Si no se especifica ninguna dirección, identifica la anteriormente seleccionada
        if args == []:
            print("Añade información de la moneda:\n" +
                  "add PATH_1 PATH_2 ... PATH_N {Info}\n" +
                  "Tambien puedes especificar si hay multiples monedas en la imagen:\n" +
                  "add (M)PATH{Info}")
        else:

            # Separa imagenes de atributos
            imgs = args[0:-1]
            info = args[-1]
            # Lee la información obtenida de cada moneda
            try:
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
            except:
                print("Error: Seguir formato -> PATH_1,PATH_2,...PATH_N{Info}")

    @staticmethod
    def invalidCommand(args):
        print("Comando no existe")


console = Console()

# Lee comandos en bucle
while (True):
    command = input("~ ")
    if command == "":
        continue
    args = command.split(" ")

    select_action = {
        "img": console.setImg,
        "find": console.findCoins,
        "add": console.addCoin
    }
    select_action.get(args[0], console.invalidCommand)(args[1:])
