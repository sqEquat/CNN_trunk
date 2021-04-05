import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import os


# Not using GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_path = "resources/prediction/alder_2.jpg"
val_path = "resources/TRUNK12_test/Val"
batch_size = 1
img_shape = (224, 224)
tree_types = ['Alder', 'Beech', 'Birch', 'Chestnut', 'Ginkgo biloba', 'Hornbeam', 'Horse chestnut',
             'Linden', 'Oak', 'Oriental plane', 'Pine', 'Spruce']


def img_predict(path):
    img = image.load_img(path, target_size=img_shape)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)/255
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)

    return prediction


def generator_predict(path):
    val_datagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
    validation = val_datagen.flow_from_directory(val_path, batch_size=batch_size,
                                                 class_mode='categorical', target_size=img_shape)
    prediction = model.predict(validation)

    return prediction, validation


def pred_analyze(prediction, generator):
    y_pred = np.argmax(prediction, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(generator.classes, y_pred))
    # con_mat = tf.math.confusion_matrix(generator.classes, y_pred).numpy()
    # con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    # con_mat_df = pd.DataFrame(con_mat_norm,
    #                           index=tree_types,
    #                           columns=tree_types)
    #
    # figure = plt.figure(figsize=(8, 8))
    # sns.heatmap(con_mat_df, annot=True, cmap=cm.Blues)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    print('Classification Report')
    print(classification_report(generator.classes, y_pred, target_names=tree_types))


if __name__ == '__main__':

    directory = "resources/TRUNK12_test/Val/chestnut"

    model = keras.models.load_model("models/resnet50/trunk12_040521_47_0.8625.h5")

    print(os.listdir(directory))
    for filename in os.listdir(directory):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            print(os.path.join(directory, filename))
            pred = img_predict(os.path.join(directory, filename))
            print(pred)
            # for i in range(len(pred[0])):
            #     tree = tree_types[i] + ': '
            #     print(tree, pred[0][i])
