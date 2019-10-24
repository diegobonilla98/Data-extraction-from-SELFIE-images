import cv2
from keras.engine.saving import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image

using_webcam = True

if using_webcam:
    webcam = cv2.VideoCapture(0)

    check, frame = webcam.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    webcam.release()

else:
    img = load_img("test2.jpg", target_size=(128, 128))
    img = img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)

plt.imshow(img[0])
plt.axis('off')
plt.show()


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

predictions = model.predict(img)
print(predictions)

sex = predictions[0][0]
age = np.argmax(predictions[1][0])
race = np.argmax(predictions[2][0])
face = np.argmax(predictions[3][0])
expression = np.argmax(predictions[4][0])
glasses = np.argmax(predictions[5][0])
lipstick = predictions[6][0]
mouth = np.argmax(predictions[7][0])
hair = np.argmax(predictions[8][0])

if sex < 0.5:
    print("Hombre")
else:
    print("Mujer")

if age == 0:
    print("Bebe")
elif age == 1:
    print("Mochuelo")
elif age == 2:
    print("Adolescente")
elif age == 3:
    print("Joven")
elif age == 4:
    print("Mediana edad")
elif age == 5:
    print("Mayor")

if race == 0:
    print("Raza blanca")
elif race == 1:
    print("Raza negra")
elif race == 2:
    print("Raza asiatica")

if face == 0:
    print("Cara ovalada")
elif face == 1:
    print("Cara redonda")
elif face == 2:
    print("Cara forma de corazon")

if lipstick > 0.5:
    print("Con pintalabios")

if hair == 0:
    print("Pelo negro")
elif hair == 1:
    print("Pelo rubio")
elif hair == 2:
    print("Pelo castaño")
elif hair == 3:
    print("Pelirrojo")