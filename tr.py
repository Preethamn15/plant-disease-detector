"""
Plant Disease Detection with GUI + Telegram Alerts (Text + Photo)
Python 3.6–3.8 compatible version
"""

"""
Plant-leaf disease detector with GUI, real-time camera preview
and Telegram alerts (message + photo).
Compatible with Python 3.6–3.8 (uses typing.Tuple).

Install dependencies:
    pip install numpy opencv-python pillow requests tensorflow
"""

# ───────────────────────── Imports ─────────────────────────
import os
import cv2
import numpy as np
import requests
import tkinter as tk
from typing import Tuple
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import telepot
from tkinter import ttk

# ─────────────────────── Configuration ─────────────────────
BOT_TOKEN = "8020947857:AAG42UbLxawg32X5NhRj97dmGTgxUvDy07k"   # ◀── Replace safely
CHAT_ID   = "1837835131"      # ◀── Replace safely
bot = telepot.Bot(BOT_TOKEN)
MODEL_PATH = "best_vgg16_model.h5"      # Trained model
IMG_SIZE   = 128                        # Model input size
UNKNOWN_THRESHOLD = 0.60                # “Unknown” if confidence lower

# Labels in the same order you trained the model
# Example: Use real class names from training
disease_classes = [
    "Bacterial Spot",
    "Leaf Blight",
    "Rust",
    "Powdery Mildew",
    "Early Blight",
    "Healthy"
]


# ───────────────────── Telegram helpers ─────────────────────
def send_telegram_message(text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url,
                      json={"chat_id": CHAT_ID,
                            "text": text,
                            "parse_mode": "Markdown"},
                      timeout=10)
    except Exception as exc:
        print(f"[Telegram] message failed: {exc}")

def send_telegram_photo(img_path: str, caption: str = "") -> None:
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(img_path, "rb") as f:
            files = {"photo": f}
            data  = {"chat_id": CHAT_ID,
                     "caption": caption,
                     "parse_mode": "Markdown"}
            requests.post(url, data=data, files=files, timeout=20)
    except Exception as exc:
        print(f"[Telegram] photo failed: {exc}")

def notify_telegram(img_path: str, label: str, confidence: float) -> None:
    """Send both a text message and the image with caption."""
    msg = (f"🌿 *Plant-leaf prediction*\n"
           f"Result   : *{label}*\n"
           f"Confidence: *{confidence:.2%}*")
    caption = f"{label} ({confidence:.2%})"
    send_telegram_message(msg)
    send_telegram_photo(img_path, caption)

# ─────────────────────── Load model ─────────────────────────
try:
    model2 = load_model(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Error", f"Model loading failed:\n{e}")
    raise SystemExit

# ──────────────────────── Utilities ─────────────────────────
def path_to_tensor(img_path: str,
                   width: int = IMG_SIZE,
                   height: int = IMG_SIZE) -> np.ndarray:
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    return x

def classify_image(img_array: np.ndarray) -> Tuple[str, float]:
    preds = model2.predict(img_array, verbose=0)
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if confidence < UNKNOWN_THRESHOLD or pred_class >= len(disease_classes):
        return "diseased", confidence
    return disease_classes[pred_class], confidence

# ─────────────────── GUI Callbacks ────────────────────────
def display_solution_in_main(label: str) -> None:
    try:
        causes_folder = "solutions"
        path = os.path.join(causes_folder, f"{label}.txt")
        if not os.path.exists(path):
            solution_box.config(state=tk.NORMAL)
            solution_box.delete("1.0", tk.END)
            solution_box.insert(tk.END, "❌ No solution file found.")
            solution_box.config(state=tk.DISABLED)
            return

        with open(path, "r", encoding="utf-8") as f:
            solution_text = f.read()

        solution_box.config(state=tk.NORMAL)
        solution_box.delete("1.0", tk.END)
        solution_box.insert(tk.END, solution_text)
        solution_box.config(state=tk.DISABLED)

    except Exception as exc:
        messagebox.showerror("Error", f"Error loading solution:\n{exc}")


def predict_image() -> None:
    rep = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
    )
    if not rep:
        return

    img_array = path_to_tensor(rep)
    label, confidence = classify_image(img_array)

    # GUI preview
    # GUI preview
    img = Image.open(rep).resize((300, 300), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    image_label.configure(image=photo, text="")  # Remove text placeholder
    image_label.image = photo  # Keep reference to avoid being cleared



    result_text.config(text=f"Predicted: {label}  ({confidence:.2%})")


    # Telegram
    notify_telegram(rep, label, confidence)

    # Show causes (skip generic “diseased” & “Normal”)
    if label not in ("Normal", "diseased"):
        display_solution_in_main(label)
    else:
        solution_box.config(state=tk.NORMAL)
        solution_box.delete("1.0", tk.END)
        solution_box.insert(tk.END, "No disease detected or solution not applicable.")
        solution_box.config(state=tk.DISABLED)


def real_time_detection() -> None:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not found.")
        return

    last_label, last_conf = None, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_arr = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_arr = np.expand_dims(img_arr, axis=0) / 255.0
        label, confidence = classify_image(img_arr)

        last_label, last_conf = label, confidence  # save latest

        cv2.putText(frame, f"{label}: {confidence:.2%}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)
        cv2.imshow("Real-Time Detection  (c = capture, q = quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('c'):
            tmp = "last_capture.jpg"
            cv2.imwrite(tmp, frame)

            notify_telegram(tmp, last_label, last_conf)

            cap.release()
            cv2.destroyAllWindows()
            if last_label not in ("Normal", "diseased"):
                open_causes_file(last_label)
            return

    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────── GUI setup ─────────────────────────
def open_main_window() -> None:
    global result_text, image_label

    root = tk.Tk()
    root.title("🌿 Plant Disease Detector")
    root.geometry("1000x700")
    root.configure(bg="#f0f5f5")

    style = ttk.Style()
    style.theme_use("clam")

    header = tk.Label(root, text="🌱 Plant Disease Detection System",
                      font=("Helvetica", 26, "bold"),
                      fg="#2e8b57", bg="#f0f5f5")
    header.pack(pady=20)

    content_frame = tk.Frame(root, bg="#f0f5f5")
    content_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    # Image Preview Section
    preview_frame = tk.Frame(content_frame, bg="#f0f5f5")
    preview_frame.grid(row=0, column=0, padx=20, pady=20)

    image_label = tk.Label(preview_frame, bg="#d0e0e3", relief=tk.RIDGE)

    image_label.pack()

    # Action Buttons Section
    button_frame = tk.Frame(content_frame, bg="#f0f5f5")
    button_frame.grid(row=0, column=1, padx=20, pady=20)

    ttk.Button(button_frame, text="📷 Predict from Image",
               command=predict_image, width=30).pack(pady=10)

    ttk.Button(button_frame, text="🔍 Real-Time Detection",
               command=real_time_detection, width=30).pack(pady=10)

    # Prediction Result
    result_text = tk.Label(root, text="", font=("Arial", 14), fg="black", bg="#f0f5f5")
    result_text.pack(pady=20)

    solution_box_label = tk.Label(root, text="💡 Suggested Cure / Solution:",
                                  font=("Segoe UI", 14, "bold"), fg="#2f855a", bg="#f0f5f5")
    solution_box_label.pack()

    global solution_box
    solution_box = tk.Text(root, height=20, wrap=tk.WORD, font=("Segoe UI", 12),
                           bg="#ffffff", fg="#333333", relief=tk.SOLID, bd=1)
    solution_box.pack(padx=40, pady=10, fill=tk.BOTH)
    solution_box.config(state=tk.DISABLED)


    root.mainloop()

# ────────────────────────── Main ────────────────────────────
if __name__ == "__main__":
    open_main_window()
