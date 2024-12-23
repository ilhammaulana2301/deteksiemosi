# Import library untuk memproses gambar, model training, dan akses webcam
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Menghindari pesan peringatan dari TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Menambahkan argumen baris perintah untuk memilih mode (training atau display)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Fungsi untuk menampilkan grafik akurasi dan loss
def plot_model_history(model_history):
    # Membuat subplots untuk menampilkan akurasi dan kerugian
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

# Direktori dataset yang digunakan untuk pelatihan dan validasi
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709  # Jumlah data latih
num_val = 7178     # Jumlah data validasi
batch_size = 64    # Ukuran batch untuk pelatihan
num_epoch = 50     # Jumlah epoch pelatihan

# Menggunakan ImageDataGenerator untuk pra-pemrosesan gambar dan augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Generator data pelatihan
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Generator data validasi
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Membuat model CNN untuk deteksi emosi
model = Sequential()
# Layer konvolusi pertama
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# Layer konvolusi kedua
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling
model.add(Dropout(0.25))  # Dropout untuk mencegah overfitting

# Layer konvolusi berikutnya
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Dropout

# Flatten layer untuk mengubah data ke vektor
model.add(Flatten())
model.add(Dense(1024, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout untuk fully connected layer
model.add(Dense(7, activation='softmax'))  # Layer output untuk 7 kelas emosi

# Jika mode adalah "train", lakukan pelatihan model
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    # Melakukan pelatihan model menggunakan data pelatihan dan validasi
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    # Menampilkan grafik akurasi dan kerugian
    plot_model_history(model_info)
    # Menyimpan bobot model
    model.save_weights('model.h5')

# Jika mode adalah "display", buka kamera dan lakukan deteksi emosi
elif mode == "display":
    model.load_weights('model.h5')  # Memuat bobot model yang sudah dilatih
    cv2.ocl.setUseOpenCL(False)  # Menonaktifkan OpenCL untuk kecepatan

    # Dictionary untuk label emosi
    emotion_dict = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Netral", 5: "Sedih", 6: "Terkejut"}

    # Membuka kamera laptop
    cap = cv2.VideoCapture(0)  # Kamera laptop

    # Cek apakah kamera terbuka
    if not cap.isOpened():
        print("Gagal membuka kamera.")
        exit()

    # Loop untuk membaca frame dari kamera
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca data dari kamera.")
            break

        # Proses frame
        frame = cv2.flip(frame, 1)  # Cermin horizontal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah ke grayscale
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Terapkan Gaussian Blur
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Deteksi wajah dan prediksi emosi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            confidence = prediction[0][maxindex] * 100  # Menghitung persentase akurasi
            text = f"{emotion_dict[maxindex]} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Menampilkan hasil deteksi emosi
        cv2.imshow('Deteksi Emosi - Kamera Laptop', frame)

        # Keluar dari loop jika menekan tombol 's'
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Melepaskan kamera dan menutup jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()
