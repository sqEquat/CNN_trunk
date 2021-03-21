import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


# Disable training on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_datagen = image.ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                         zoom_range=0.2, vertical_flip=True)

test_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory("resources/TRUNK12_test/Train", batch_size=8,
                                                    class_mode='categorical', target_size=(224, 224))
validation_generator = test_datagen.flow_from_directory("resources/TRUNK12_test/Val", batch_size=8,
                                                        class_mode='categorical', target_size=(224, 224))

base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(12, activation='softmax'))

base_model.summary()

base_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

resnet_history = base_model.fit(train_generator, validation_data=validation_generator,
                                steps_per_epoch=16, epochs=100)

base_model.save("ResNet50V1_12_64_64")

print('\nHistory dict: ', resnet_history.history)
print("'loss': ", resnet_history.history['loss'])
print("'acc': ", resnet_history.history['acc'])
print("'val_loss': ", resnet_history.history['val_loss'])
print("'val_acc': ", resnet_history.history['val_acc'])
