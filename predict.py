from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os


# Disable training on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_path = "resources/prediction/oak_1.jpg"

model = keras.models.load_model("ResNet50V1_12")

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

prediction = model.predict(img_preprocessed)

tree_type = ['Alder', 'Beech', 'Birch', 'Chestnut', 'Ginkgo biloba', 'Hornbeam', 'Horse chestnut',
             'Linden', 'Oak', 'Oriental plane', 'Pine', 'Spruce']

for i in range(len(prediction[0])):
    tree = tree_type[i] + ': '
    print(tree, prediction[0][i])
