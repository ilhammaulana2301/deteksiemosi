import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load Model dan Haar Cascade
face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
classifier = load_model(r"model.h5")

# Label Emosi
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Netral', 'Sedih', 'Terkejut']

# Coba membuka kamera laptop dan kamera HP
cap_laptop = cv2.VideoCapture(0)  # Kamera laptop
cap_hp = cv2.VideoCapture(1)  # Kamera HP (sesuaikan indeks berdasarkan deteksi kamera)

# Periksa apakah kedua kamera berhasil dibuka
if not cap_laptop.isOpened():
    print("Gagal membuka kamera laptop. Pastikan terhubung.")
if not cap_hp.isOpened():
    print("Gagal membuka kamera HP. Periksa koneksi dan indeks kamera.")

if not cap_laptop.isOpened() or not cap_hp.isOpened():
    print("Tidak dapat melanjutkan tanpa akses ke kedua kamera.")
    exit()

def preprocess_roi(roi_gray):
    """Fungsi preprocessing untuk ROI wajah."""
    # Resize ke ukuran model
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Histogram Equalization untuk meningkatkan kontras
    roi_gray = cv2.equalizeHist(roi_gray)

    # Gaussian Blur untuk mengurangi noise
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    # Normalisasi: piksel menjadi 0 hingga 1
    roi_gray = roi_gray.astype('float') / 255.0

    # Konversi ke array dan tambahkan dimensi untuk input ke model
    roi = img_to_array(roi_gray)
    roi = np.expand_dims(roi, axis=0)

    return roi

while True:
    # Baca frame dari kamera laptop
    ret_laptop, frame_laptop = cap_laptop.read()
    # Baca frame dari kamera HP
    ret_hp, frame_hp = cap_hp.read()

    if not ret_laptop or not ret_hp:
        print("Gagal membaca frame dari salah satu kamera.")
        break

    # Cerminkan frame kamera laptop dan HP
    frame_laptop = cv2.flip(frame_laptop, 1)  # Cermin horizontal (efek seperti cermin)
    frame_hp = cv2.flip(frame_hp, 1)

    # Proses kamera laptop
    gray_laptop = cv2.cvtColor(frame_laptop, cv2.COLOR_BGR2GRAY)
    faces_laptop = face_classifier.detectMultiScale(gray_laptop, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_laptop:
        cv2.rectangle(frame_laptop, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray_laptop[y:y + h, x:x + w]

        # Preprocessing ROI
        roi = preprocess_roi(roi_gray)

        # Prediksi emosi
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        accuracy = prediction.max() * 100  # Persentase akurasi
        label_with_accuracy = f"{label}: {accuracy:.2f}%"  # Gabungkan label dan akurasi
        label_position = (x, y - 10)
        cv2.putText(frame_laptop, label_with_accuracy, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Proses kamera HP
    gray_hp = cv2.cvtColor(frame_hp, cv2.COLOR_BGR2GRAY)
    faces_hp = face_classifier.detectMultiScale(gray_hp, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_hp:
        cv2.rectangle(frame_hp, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray_hp[y:y + h, x:x + w]

        # Preprocessing ROI
        roi = preprocess_roi(roi_gray)

        # Prediksi emosi
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        accuracy = prediction.max() * 100  # Persentase akurasi
        label_with_accuracy = f"{label}: {accuracy:.2f}%"  # Gabungkan label dan akurasi
        label_position = (x, y - 10)
        cv2.putText(frame_hp, label_with_accuracy, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Gabungkan kedua frame
    frame_laptop = cv2.resize(frame_laptop, (640, 480))
    frame_hp = cv2.resize(frame_hp, (640, 480))
    combined_frame = np.hstack((frame_laptop, frame_hp))

    # Tampilkan hasil
    cv2.imshow('Deteksi Emosi - Kamera Laptop & HP (Cermin)', combined_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepas kamera dan tutup jendela
cap_laptop.release()
cap_hp.release()
cv2.destroyAllWindows()
