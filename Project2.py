import cv2
import os
import time
import logging
from playsound import playsound
import smtplib
from email.message import EmailMessage
import face_recognition
import geocoder
import numpy as np
from datetime import datetime

# === Configuration ===
KNOWN_IMAGES_PATH = r"C:\Users\Aastha Chaurasia\OneDrive\Desktop\landing_page\image_folder"
ALERT_SOUND_PATH = r"C:\Users\Aastha Chaurasia\OneDrive\Desktop\landing_page\alert.mp3"
EMAIL_SENDER = "spiderabhay4321@gmail.com"
EMAIL_PASSWORD = "abcl cfhs giwm elvj"
EMAIL_RECEIVER = "antiterroristgrop@gmail.com"
COOLDOWN_PERIOD = 60  # seconds
FACE_DISTANCE_THRESHOLD = 0.45  # Lower means stricter matching

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)

# === Load Known Faces ===
def load_known_faces(path):
    images = []
    class_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.jfif')

    for file_name in os.listdir(path):
        if file_name.lower().endswith(valid_extensions):
            full_path = os.path.join(path, file_name)
            img = cv2.imread(full_path)
            if img is None:
                logging.warning(f"Could not read {file_name}")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                images.append(encodings[0])
                class_names.append(os.path.splitext(file_name)[0])
                logging.info(f"Loaded encoding for: {file_name}")
            else:
                logging.warning(f"No faces found in {file_name}")
    return images, class_names

# === Send Email Alert ===
def send_email(name, image_path, location):
    msg = EmailMessage()
    msg['Subject'] = f"⚠️ Suspect Detected: {name}"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(f"Suspect Name: {name}\nLocation: {location}\nTime: {datetime.now()}")

    try:
        with open(image_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logging.info("✅ Email alert sent.")
    except Exception as e:
        logging.error(f"❌ Failed to send email: {e}")

# === Play Alert Sound ===
def play_alert():
    if os.path.exists(ALERT_SOUND_PATH):
        playsound(ALERT_SOUND_PATH)
    else:
        logging.error("❌ Alert sound file not found.")

# === Main ===
def main():
    logging.info("🔍 Loading known faces...")
    known_encodings, known_names = load_known_faces(KNOWN_IMAGES_PATH)
    if not known_encodings:
        logging.error("No valid face encodings found.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Webcam not accessible.")
        return

    logging.info("🎥 Webcam started. Press 'q' to quit.")
    last_alert_time = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            continue

        # Resize frame for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)

            if distances[best_match_index] < FACE_DISTANCE_THRESHOLD:
                name = known_names[best_match_index].upper()
                top, right, bottom, left = face_location
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, f"{name}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                current_time = time.time()
                if name not in last_alert_time or (current_time - last_alert_time[name] > COOLDOWN_PERIOD):
                    last_alert_time[name] = current_time

                    detected_img_path = f"detected_{name}_{int(current_time)}.jpg"
                    cv2.imwrite(detected_img_path, frame)

                    # Get location
                    location = geocoder.ip('me').latlng or "Unknown"

                    logging.info(f"🚨 Suspect Detected: {name} | Distance: {distances[best_match_index]:.2f}")
                    play_alert()
                    send_email(name, detected_img_path, location)

        cv2.imshow("Suspect Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Run ===
if __name__ == "__main__":
    main()
