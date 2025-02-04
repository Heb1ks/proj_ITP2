import cv2
import numpy as np
import easyocr
import imutils
from matplotlib import pyplot as pl

# cap = cv2.VideoCapture(камера)##для камеры
# while True:
#     suc, img = cap.read()


img = cv2.imread('.venv/images/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgFilter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(imgFilter, 30, 200)

cont = cv2.findContours(edges.copy(),cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]

pos = None
for i in cont:
    appox = cv2.approxPolyDP(i, 10 , True)
    if len(appox) == 4:
        pos = appox
        break

mask = np.zeros(gray.shape, np.uint8)
newImg = cv2.drawContours(mask,[pos],0,255,-1)
bitwiseImg = cv2.bitwise_and(img,img,mask=mask)


( x,y) =np.where(mask == 255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
cropp = gray[x1:x2,y1:y2]

text = easyocr.Reader(['en'])
text = text.readtext(cropp)

res = text[0][-2]
finalImg =cv2.putText(img,res,(x1,y2 + 60 ) , cv2.FONT_HERSHEY_SIMPLEX , 3 , (0,0,255) ,2)
finalImg = cv2.rectangle(img,(x1,x2),(x2,y2),(255,0,0),1)

print(text)
# print(pos)

pl.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
pl.show()
