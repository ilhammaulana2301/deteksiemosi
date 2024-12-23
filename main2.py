import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Argumen baris perintah
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Plot akurasi dan kurva kerugian
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

# Direktori dataset
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Membuat model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')

elif mode == "display":
    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Netral", 5: "Sedih", 6: "Terkejut"}

    # Membuka dua kamera
    cap_laptop = cv2.VideoCapture(0)  # Kamera laptop
    cap_hp = cv2.VideoCapture(1)  # Kamera HP (sesuaikan indeks kamera)

    if not cap_laptop.isOpened():
        print("Gagal membuka kamera laptop.")
    if not cap_hp.isOpened():
        print("Gagal membuka kamera HP.")

    if not cap_laptop.isOpened() or not cap_hp.isOpened():
        print("Tidak dapat melanjutkan tanpa kedua kamera.")
        exit()

    # Loop untuk membaca frame dari kedua kamera
    while True:
        ret_laptop, frame_laptop = cap_laptop.read()
        ret_hp, frame_hp = cap_hp.read()

        if not ret_laptop and not ret_hp:
            print("Gagal membaca data dari kedua kamera.")
            break

        # Proses cerminan dan resizing frame
        if ret_laptop:
            frame_laptop = cv2.flip(frame_laptop, 1)  # Cermin horizontal
            gray_laptop = cv2.cvtColor(frame_laptop, cv2.COLOR_BGR2GRAY)
            faces_laptop = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(
                gray_laptop, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces_laptop:
                cv2.rectangle(frame_laptop, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                roi_gray = gray_laptop[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame_laptop, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if ret_hp:
            frame_hp = cv2.flip(frame_hp, 1)  # Cermin horizontal
            gray_hp = cv2.cvtColor(frame_hp, cv2.COLOR_BGR2GRAY)
            faces_hp = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(
                gray_hp, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces_hp:
                cv2.rectangle(frame_hp, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                roi_gray = gray_hp[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame_hp, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Gabungkan kedua frame
        if ret_laptop and ret_hp:
            frame_laptop = cv2.resize(frame_laptop, (640, 480))
            frame_hp = cv2.resize(frame_hp, (640, 480))
            combined_frame = np.hstack((frame_laptop, frame_hp))
        else:
            combined_frame = frame_laptop if ret_laptop else frame_hp

        # Tampilkan hasil
        cv2.imshow('Deteksi Emosi - Kamera Laptop & HP', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    cap_laptop.release()
    cap_hp.release()
    cv2.destroyAllWindows()
