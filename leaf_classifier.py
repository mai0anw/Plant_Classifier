import numpy as np
import tensorflow as tf
from keras import layers, models
import cv2 as cv
import matplotlib.pyplot as plt
import os

#F1 Score implementation:

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_actual, y_pred, sample_weight=None):
        self.precision.update_state(y_actual, y_pred, sample_weight)
        self.recall.update_state(y_actual, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


#First Step: Split the data into training testing and val 

#img_height = 512
#img_width = 256

batch_size = 32
img_height = 128
img_width = 128

img_size = (img_height, img_width)

#join all training data together so it is easier to split

#LeafSnap images
base_path = os.path.join("LeafSnap dataset", "leafsnap-dataset", "dataset", "images")
field_path = os.path.join(base_path, "field")
lab_path = os.path.join(base_path, "lab")

#LeafSnap segmented
seg_base_path = os.path.join("LeafSnap dataset", "leafsnap-dataset", "dataset", "segmented")
seg_field_path = os.path.join(base_path, "field")
seg_lab_path = os.path.join(base_path, "lab")



image_paths = []
labels = []

species_folders = sorted(set(os.listdir(field_path)) | set(os.listdir(lab_path)) | set(os.listdir(seg_field_path)) | set(os.listdir(seg_lab_path)))

label_to_index = {label: idx for idx, label in enumerate(species_folders)}


def gather_images(source_path):
    for species in os.listdir(source_path):
        species_path = os.path.join(source_path, species)
        if os.path.isdir(species_path):
            for img_file in os.listdir(species_path):
                if img_file.lower().endswith(('.jpg', '.jpeg')): #isolates all jpg. all pics are jpg but just in case
                    image_paths.append(os.path.join(species_path, img_file))
                    labels.append(label_to_index[species])

gather_images(field_path)
gather_images(lab_path)
gather_images(seg_field_path)
gather_images(seg_lab_path)

paths_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def preprocess(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    #resize
    img = tf.image.resize(img, img_size)
    #normalize
    img = tf.cast(img, tf.float32) / 255.0
    #returns (preprocessed image, label)
    return img, tf.one_hot(label, depth=len(species_folders))


#Dataset.map to create a dataset of image, label pairs:
full_dataset = paths_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

dataset_size = len(image_paths)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)

full_dataset = full_dataset.shuffle(len(image_paths), reshuffle_each_iteration=False)

train_ds = full_dataset.take(train_size).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = full_dataset.skip(train_size).take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = full_dataset.skip(train_size + val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Number of species:", len(species_folders))
print("Classes:", species_folders)
#print(label_to_index)


#####
#CNN MODEL

input_shape = (128, 128, 3)
num_classes = len(species_folders)

model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
    
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall'),
                  F1Score(),
                  tf.keras.metrics.CategoricalAccuracy(name='accuracy')
                ])

#Add Precision, Recall, and F1 score too


#model.add 

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()



