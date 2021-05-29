import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
from scipy.signal import resample
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import os


# Not using GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_path = "resources/prediction/tree1/oak_1_c.jpg"
val_path = "resources/TRUNK12_test/Val"
batch_size = 10
img_shape = (224, 224)
tree_types = ['Alder', 'Beech', 'Birch', 'Chestnut', 'Ginkgo biloba', 'Hornbeam', 'Horse chestnut',
             'Linden', 'Oak', 'Oriental plane', 'Pine', 'Spruce']


def img_predict(path, cnn_model):
    img = image.load_img(path, target_size=img_shape)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch) / 255
    prediction = cnn_model.predict(img_preprocessed)

    return prediction


def generator_predict(path):
    val_datagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
    validation = val_datagen.flow_from_directory(path, batch_size=batch_size,
                                                 class_mode='categorical', target_size=img_shape)
    prediction = model.predict(validation)

    return prediction, validation


def pred_analyze(prediction, generator):
    y_pred = np.argmax(prediction, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(generator.labels, y_pred))

    con_mat = tf.math.confusion_matrix(generator.labels, y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=tree_types,
                              columns=tree_types)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print('Classification Report')
    print(classification_report(generator.labels, y_pred, target_names=tree_types))


if __name__ == '__main__':

    model = keras.models.load_model("models/resnet50/resnet50_trunk12_74_0.9250.h5")
    # model.summary()
    # pred = img_predict(image_path)
    # for i in range(len(pred[0])):
    #     tree = tree_types[i] + ': '
    #     print(tree, pred[0][i])

    directory = "resources/prediction/pine"

    count = 1
    confusion_str = [0 for i in range(0, 12)]
    for filename in os.listdir(directory):
        if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpg"):

            pred = img_predict(os.path.join(directory, filename), model)[0]
            idx = np.argmax(pred)
            confusion_str[idx] += 1
            print(count, filename, tree_types[idx])
            count += 1

    print(confusion_str)

    # for filename in os.listdir(directory):
    #     if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpg"):
    #         print('==============================================')
    #         print(os.path.join(directory, filename))
    #         pred = img_predict(os.path.join(directory, filename), model)
    #         for i in range(len(pred[0])):
    #             tree = tree_types[i] + ': '
    #             print(tree, pred[0][i])
