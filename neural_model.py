from data_reader import read_data
from image_preprocessing import load_image
from keras import layers, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np

images_dict = read_data(15000)

print("Extrayendo datos.")
sex_data = [gender['is_female'] for gender in images_dict]
age_data = [[age['baby'], age['child'], age['teenager'], age['youth'], age['middle_age'], age['senior']]
            for age in images_dict]
race_data = [[race['white'], race['black'], race['asian']] for race in images_dict]
face_data = [[face['oval_face'], face['round_face'], face['heart_face']] for face in images_dict]
expression_data = [[expression['smiling'], expression['duck_face']]
                   for expression in images_dict]
glasses_data = [[glasses['wearing_glasses'], glasses['wearing_sunglasses']] for glasses in images_dict]
lipstick_data = [lipstick['wearing_lipstick'] for lipstick in images_dict]
mouth_data = [[mouth['frowning'], mouth['tongue_out'], mouth['mouth_open']] for mouth in images_dict]
hair_data = [[hair['black_hair'], hair['blond_hair'], hair['brown_hair'], hair['red_hair']]
             for hair in images_dict]

print("Procesando imagenes...")
img_tensors = [load_image("Selfie-dataset/images/" + img['file_name'] + ".jpg")
               for img in images_dict]

num_images = len(img_tensors)
img_tensors = np.array(img_tensors).reshape((num_images, 128, 128, 3))
sex_data = np.array(sex_data)
age_data = np.array(age_data)
race_data = np.array(race_data)
face_data = np.array(face_data)
expression_data = np.array(expression_data)
glasses_data = np.array(glasses_data)
lipstick_data = np.array(lipstick_data)
mouth_data = np.array(mouth_data)
hair_data = np.array(hair_data)

image_input = Input(shape=(128, 128, 3), dtype='float32', name='images')

x1 = layers.SeparableConv2D(64, (3, 3), activation='relu')(image_input)
x1 = layers.SeparableConv2D(64, (3, 3), activation='relu')(x1)
x1 = layers.MaxPooling2D((2, 2))(x1)

x2 = layers.SeparableConv2D(128, (3, 3), activation='relu')(x1)
x2 = layers.SeparableConv2D(128, (3, 3), activation='relu')(x2)
x2 = layers.MaxPooling2D((2, 2))(x2)

x3 = layers.SeparableConv2D(256, (3, 3), activation='relu')(x2)
x3 = layers.SeparableConv2D(256, (3, 3), activation='relu')(x3)
x3 = layers.MaxPooling2D((2, 2))(x3)

x4 = layers.SeparableConv2D(512, (3, 3), activation='relu')(x3)
x4 = layers.SeparableConv2D(512, (3, 3), activation='relu')(x4)
x4 = layers.MaxPooling2D((2, 2))(x4)

flat = layers.GlobalMaxPooling2D()(x4)

img_output = layers.Dense(1024, activation='relu')(flat)
img_output = layers.BatchNormalization()(img_output)
img_output = layers.Dense(1024, activation='relu')(img_output)
img_output = layers.Dropout(0.5)(img_output)

sex_predictions = layers.Dense(1, activation='sigmoid', name='sex')(img_output)
age_predictions = layers.Dense(6, activation='softmax', name='age')(img_output)
race_predictions = layers.Dense(3, activation='softmax', name='race')(img_output)
face_predictions = layers.Dense(3, activation='softmax', name='face')(img_output)
expression_predictions = layers.Dense(2, activation='softmax', name='expression')(img_output)
glasses_predictions = layers.Dense(2, activation='softmax', name='glasses')(img_output)
lipstick_predictions = layers.Dense(1, activation='sigmoid', name='lipstick')(img_output)
mouth_predictions = layers.Dense(3, activation='softmax', name='mouth')(img_output)
hair_predictions = layers.Dense(4, activation='softmax', name='hair')(img_output)

model = Model(image_input, [sex_predictions, age_predictions, race_predictions,
                            face_predictions, expression_predictions, glasses_predictions,
                            lipstick_predictions, mouth_predictions, hair_predictions])

model.compile(optimizer='adam',
              loss=['binary_crossentropy', 'categorical_crossentropy',
                    'categorical_crossentropy', 'categorical_crossentropy',
                    'categorical_crossentropy', 'categorical_crossentropy',
                    'binary_crossentropy', 'categorical_crossentropy',
                    'categorical_crossentropy'],
              loss_weights=[7., 1., 1., 1., 1., 1., 7., 1., 1.],
              metrics=['acc'])

checkpoint = ModelCheckpoint(filepath="weights-improvement-{epoch:02d}.hdf5",
                             save_best_only=True, mode='max')
model.fit(img_tensors, [sex_data, age_data, race_data,
                        face_data, expression_data, glasses_data,
                        lipstick_data, mouth_data, hair_data], epochs=20,
          batch_size=32, validation_split=0.2, callbacks=[checkpoint])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Modelo salvado.")
