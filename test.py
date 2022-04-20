# %%
from multiprocessing import Event
from typing_extensions import Self
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imProcessing import ImProcessing

# %%

def prueba1():

    im = cv2.imread('Images/coin.jpg')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(im, contours, -1, (0, 255, 0), 1)

    for contour in contours:
        if cv2.contourArea(contour) < 80:
            cv2.drawContours(im, contour, -1, (255, 0, 0), 3)
    cv2.imshow('contour', img)
    cv2.waitKey(0)
    cv2.imwrite('contour.png', img)


def prueba2():

    # reading image
    img = cv2.imread('Images/coin.jpg')

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    _, contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, False), True)

        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 4:
            cv2.putText(img, 'Quadrilateral', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(img, 'Pentagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(img, 'Hexagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            cv2.putText(img, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # displaying the image after drawing contours
    cv2.imshow('shapes', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prueba3():

    img1 = prueba4()
    # img1 = cv2.resize(img1, (256, 256))
    gray = img1.copy()
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Create mask
    height, width = img1.shape
    mask = np.zeros((height, width), np.uint8)
    # edges = cv2.Canny(thresh, 200, 255)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1,
                               10000, param1=50, param2=30, minRadius=0, maxRadius=0)
    for i in circles[0, :]:
        i[2] = i[2]+4
        # Draw on mask
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255),
                   thickness=-1)    # print(circles)

    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img1, img1, mask=mask)

    cv2.imshow('detected ', masked_data)
    # Apply Threshold
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Find Contour
    contours = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop masked_data
    crop = masked_data[y:y+h, x:x+w]
    # Code to close Window
    cv2.imshow('detected Edge', crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prueba4():
    # -----Reading the image-----------------------------------------------------
    img = cv2.imread('Images/coin_b.jpg', 1)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel---------CLAHE (Contrast Limited Adaptive Histogram Equalization)----------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    return final

    #_____END_____#
# img = prueba4()


def prueba5():
    ip = ImProcessing('Images/coins.jpg')
    ip.preProcessing()
    img = ip.img
    blur = cv2.GaussianBlur(img, (7, 7), 1)
    h, w = img.shape[:2]
    print(h, w)
    cv2.imshow('gaussian blur', blur)

    # Morphological gradient

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('Morphological gradient', gradient)
    cv2.waitKey()

    edges = cv2.Canny(blur, 50, 100)
    cv2.imshow("as", edges)
    # Binarize gradient

    lowerb = np.array([0, 0, 0])
    upperb = np.array([15, 15, 15])
    binary = cv2.inRange(gradient, lowerb, upperb)
    cv2.imshow('Binarized gradient', binary)
    cv2.waitKey()


# %%
prueba5()

# %%
ip = ImProcessing('Images/coin3.jpg')
# ip.preProcessing()
ip.increaseContrast()
img = ip.img
gray = img.copy()
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

if thresh[0, 0] != 0:
    thresh = (255-thresh)
kernel = np.ones((5, 5), np.uint8)

print("asd")
thresh = cv2.dilate(thresh, kernel, iterations=5)
cv2.imshow("1", thresh)
cv2.waitKey(0)
thresh = cv2.erode(thresh, kernel)

cv2.imshow("2", thresh)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

foreground = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Cleanup up crystal foreground mask', foreground)
cv2.waitKey()
cv2.destroyAllWindows()

# -- Creamos una máscara --
height, width = img.shape
mask = np.zeros((height, width), np.uint8)

# Detectamos los circulos
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, 150,param1=50,param2=27,minRadius=60,maxRadius=120)

# circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=2, minDist=100,
# #                            param1=50, param2=50, minRadius=255, maxRadius=600)

circles = cv2.HoughCircles(foreground, cv2.HOUGH_GRADIENT, 4,
                           255, param1=50, param2=250, minRadius=127, maxRadius=0)
if circles is None:
    print("sirculoW")
    circles = cv2.HoughCircles(foreground, cv2.HOUGH_GRADIENT, 4,
                               255, param1=50, param2=100, minRadius=127, maxRadius=0)

for i in circles[0, :]:
    # i[2] = i[2]+4
    # Dibujamos los circulos en la máscara
    cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255),
               thickness=-1)

    cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0),
               thickness=2)

cv2.imshow("circles", gray)
# Aplicamos la máscara
masked_data = cv2.bitwise_and(img, img, mask=mask)

# -- Ajustamos la imagen al circulo --

# Buscamos contornos en la máscara
contours = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])

# Recortamos la segun los contornos de la máscara
img = masked_data[y:y+h, x:x+w]
img = cv2.resize(img, (256, 256))
cv2.imshow("final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# blur = cv2.GaussianBlur(img, (7, 7), 1)
# h, w = img.shape[:2]
# cv2.imshow("as", cv2.Canny(img, 50, 100))
# # Morphological gradient

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
# gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow('Morphological gradient', gradient)
# cv2.waitKey()

# edges = cv2.Canny(blur, 50, 100)
# cv2.imshow("as", edges)
# # Binarize gradient

# lowerb = np.array([0, 0, 0])
# upperb = np.array([15, 15, 15])
# binary = cv2.inRange(gradient, lowerb, upperb)
# cv2.imshow('Binarized gradient', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# %%
