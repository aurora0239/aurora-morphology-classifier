# %% Import libraries
import os
import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Convolution2D, MaxPooling2D, Flatten,
                          Dense, Dropout, GlobalAveragePooling2D)
from keras.applications import VGG16
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from keras.preprocessing import image
from datetime import datetime
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
import seaborn as sns

# %% Reproducibility seeds (seed=62; do not change unless retraining)
seed_value = 62
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Directory for Image containing training and validation dataset
train_dir = "/data/Work/Recent Work/Maitri/Programs/Python_prog/VGG16_model/Training_data"
valid_dir = "/data/Work/Recent Work/Maitri/Programs/Python_prog/VGG16_model/validation_data"
save_dir = "/data/Work/Recent Work/Maitri/Programs/Python_prog/VGG16_model/save_labeled_data"
# %% Image dimensions (VGG16 default input)
img_width, img_height = 224, 224

# %% VGG16 convolutional base (ImageNet weights, no top)
conv_base = VGG16(weights='imagenet', include_top=False,
                  input_shape=(img_width, img_height, 3))
conv_base.summary()

# %% Image Data Generator (with augmentation for training)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=70,
    horizontal_flip=True,
    vertical_flip=True,
)
batch_size = 32

# %% Feature extraction from VGG16 convolutional base
def extract_features(directory, sample_count):
    generator = datagen.flow_from_directory(
        directory, target_size=(img_width, img_height),
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    features = np.zeros(shape=(sample_count, 7, 7, 512))
    labels   = np.zeros(shape=(sample_count, len(generator.class_indices)))

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch    = conv_base.predict(inputs_batch)
        batch_size_actual = len(features_batch)
        features[i * batch_size: i * batch_size + batch_size_actual] = features_batch
        labels[i   * batch_size: i * batch_size + batch_size_actual] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# %% Parallelised feature extraction
def parallel_extract_features(directories, sample_counts):
    features, labels = [], []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_features, d, c)
                   for d, c in zip(directories, sample_counts)]
        for future in as_completed(futures):
            f, l = future.result()
            features.append(f)
            labels.append(l)
    return np.vstack(features), np.vstack(labels)

# %% Extract features — adjust sample counts to match your dataset
train_features,      train_labels      = parallel_extract_features([TRAIN_DIR], [3521])
validation_features, validation_labels = parallel_extract_features([VALID_DIR], [1493])

# %% Model definition: GlobalAveragePooling + 7-class softmax head
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
model.add(Dense(7, activation='softmax'))

lr_schedule = ExponentialDecay(initial_learning_rate=1e-5,
                               decay_steps=10000, decay_rate=0.96)
optimizer   = Adam(learning_rate=lr_schedule)
reduce_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=1e-7)
early_stop  = EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# %% Train (run each time the session is restarted)
from tensorflow.keras.callbacks import CSVLogger
csv_logger = CSVLogger('history.csv', append=True) #To save the history for references
history = model.fit(
    train_features, train_labels,
    epochs=150, batch_size=batch_size,
    validation_data=(validation_features, validation_labels),
    callbacks=[csv_logger]
)

# %% Training history plots
def plot_history(history):
    acc     = history.history['acc']
    val_acc = history.history['val_acc']
    loss    = history.history['loss']
    val_loss= history.history['val_loss']
    epochs  = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, acc,      'green',   label='Training accuracy',   linewidth=3)
    plt.plot(epochs, val_acc,  'skyblue', label='Validation accuracy', linewidth=3)
    plt.plot(epochs, loss,     'blue',    label='Training loss',       linewidth=3)
    plt.plot(epochs, val_loss, 'gray',    label='Validation loss',     linewidth=3)
    plt.xlabel("Epochs", fontsize=40)
    plt.ylabel("Accuracy / Loss", fontsize=40)
    plt.xticks(fontsize=40); plt.yticks(fontsize=40)
    plt.legend(fontsize=20); plt.grid(True); plt.tight_layout()
    plt.show()

plot_history(history)

# %% Prediction: classify a single image and annotate it
def prediction(img_path):
    org_img    = Image.open(img_path)
    img_tensor = image.img_to_array(
        image.load_img(img_path, target_size=(img_width, img_height))
    ) / 255.0

    features        = conv_base.predict(img_tensor.reshape(1, img_width, img_height, 3))
    pred            = model.predict(features)
    classes         = ["Arc", "Cloudy", "Diffused", "Discrete", "Moon", "No_Aurora", "Twilight"]
    predicted_class = classes[np.argmax(pred[0])]
    prediction_val  = np.max(pred[0])

    # Parse datetime from filename  (format: Station_YYYY_MM_DD__HH_MM_SS.jpg)
    file_name    = os.path.basename(img_path)
    datetime_str = file_name.replace("Bharati_", "").replace("Maitri_", "").replace(".jpg", "")
    datetime_obj = datetime.strptime(datetime_str, "%Y_%m_%d__%H_%M_%S")

    # Annotate image
    draw      = ImageDraw.Draw(org_img)
    font      = ImageFont.truetype("arial.ttf", 20)
    text1     = f'Predicted: {predicted_class} ({prediction_val:.2f})'
    text1_bbox = draw.textbbox((10, 10), text1, font=font)
    draw.rectangle([text1_bbox[0]-2, text1_bbox[1]-2,
                    text1_bbox[2]+2, text1_bbox[3]+2], fill="black")
    draw.text((10, 10), text1, fill="yellow", font=font)

    text2      = f'Bharati: {datetime_obj} UT'
    text2_bbox = draw.textbbox((10, org_img.height - 30), text2, font=font)
    bottom_y   = org_img.height - 30 - (text2_bbox[3] - text2_bbox[1])
    draw.rectangle([text2_bbox[0]-2, bottom_y-2,
                    text2_bbox[2]+2, bottom_y+(text2_bbox[3]-text2_bbox[1])+2], fill="black")
    draw.text((10, bottom_y), text2, fill="yellow", font=font)

    os.makedirs(SAVE_DIR, exist_ok=True)
    org_img.save(os.path.join(SAVE_DIR, file_name))
    return predicted_class, pred[0], file_name

# %% Batch prediction (parallelised) over a directory
def parallel_prediction(directory):
    class_mapping = {
        "Arc": 1, "Cloudy": -3, "Twilight": -2,
        "Diffused": 3, "Discrete": 2, "Moon": -4, "No_Aurora": -1
    }
    results = []

    def predict_and_collect(img_path):
        predicted_class, prediction_probs, file_name = prediction(img_path)
        if predicted_class is None:
            return
        datetime_str = (file_name.replace("Bharati_", "")
                                 .replace("Maitri_", "")
                                 .replace(".jpg", "").replace(".png", ""))
        dt = datetime.strptime(datetime_str, "%Y_%m_%d__%H_%M_%S")
        results.append({
            "datetime":               dt,
            "file_name":              file_name,
            "predicted_class":        predicted_class,
            "class_value":            class_mapping[predicted_class],
            "prediction_probabilities": float(np.max(prediction_probs))
        })

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(predict_and_collect, os.path.join(directory, f))
            for f in os.listdir(directory)
            if f.lower().endswith(('.jpg', '.png'))
        ]
        for future in as_completed(futures):
            future.result()

    return pd.DataFrame(results)

# %% Run batch prediction
os.makedirs(SAVE_DIR, exist_ok=True)
df = parallel_prediction(PRED_DIR)
df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved predictions → {CSV_OUTPUT}")

# %% Validation evaluation
predictions      = model.predict(validation_features)
predicted_labels = np.argmax(predictions, axis=1)
true_labels      = np.argmax(validation_labels, axis=1)
class_labels     = ["Arc", "Cloudy", "Diffused", "Discrete", "Moon", "No_Aurora", "Twilight"]
