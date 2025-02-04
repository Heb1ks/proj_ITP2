import cv2
import numpy as np
import easyocr
import imutils
from matplotlib import pyplot as plt

### Версия 1 (Не очень) не может распозновать номера некоторый нормально

# cap = cv2.VideoCapture(камера)##для камеры
# while True:
#     suc, img = cap.read()



def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edges = cv2.Canny(gray, 30, 200)
    return img, gray, edges


def find_license_plate_contour(edges):
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            return approx
    return None


def extract_plate(image, gray, plate_contour):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    bitwise = cv2.bitwise_and(image, image, mask=mask)

    (x, y) = np.where(mask == 255)
    if len(x) == 0 or len(y) == 0:
        return None

    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped = gray[x1:x2, y1:y2]
    return cropped, (x1, y1, x2, y2)


def recognize_text(image):
    reader = easyocr.Reader(['en'])
    text_results = reader.readtext(image)
    return text_results


def main(image_path):
    img, gray, edges = preprocess_image(image_path)
    plate_contour = find_license_plate_contour(edges)

    if plate_contour is None:
        print("Не удалось найти номерной знак")
        return

    plate, coords = extract_plate(img, gray, plate_contour)

    if plate is None:
        print("Ошибка: номерной знак не найден")
        return

    text_results = recognize_text(plate)

    if not text_results:
        print("OCR не распознал текст")
        return

    recognized_text = text_results[0][1]
    print("Распознанный номер:", recognized_text)

    x1, y1, x2, y2 = coords
    final_img = cv2.putText(img, recognized_text, (y1, x2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    final_img = cv2.rectangle(img, (y1, x1), (y2, x2), (255, 0, 0), 3)

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


image_path = ".venv/images/2.jpg"
main(image_path)


###(КАМЕРА НО ПЛОХО РАБОАЕТ)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_license_plate_rus_16stages.xml')
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray, 3, 5)
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     cv2.imshow("qw", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
















### Первая версия (Еще хуже чем 2)


# img = cv2.imread('.venv/images/8.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# imgFilter = cv2.bilateralFilter(gray, 11, 15, 15)
# edges = cv2.Canny(imgFilter, 30, 200)
#
# cont = cv2.findContours(edges.copy(),cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
# cont = imutils.grab_contours(cont)
# cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]
#
# pos = None
# for i in cont:
#     appox = cv2.approxPolyDP(i, 10 , True)
#     if len(appox) == 4:
#         pos = appox
#         break
#
# mask = np.zeros(gray.shape, np.uint8)
# newImg = cv2.drawContours(mask,[pos],0,255,-1)
# bitwiseImg = cv2.bitwise_and(img,img,mask=mask)
#
#
# ( x,y) =np.where(mask == 255)
# (x1,y1) = (np.min(x),np.min(y))
# (x2,y2) = (np.max(x),np.max(y))
# cropp = gray[x1:x2,y1:y2]
#
# text = easyocr.Reader(['en'])
# text = text.readtext(cropp)
#
# res = text[0][1]
# finalImg =cv2.putText(img,res,(x1,y2 + 60 ) , cv2.FONT_HERSHEY_SIMPLEX , 3 , (0,0,255) ,2)
# finalImg = cv2.rectangle(img,(x1,x2),(x2,y2),(255,0,0),1)
#
# print(text)
# # print(pos)
#
# plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
# plt.show()
