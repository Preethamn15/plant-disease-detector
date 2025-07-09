import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Load pre-trained model
try:
    model2 = load_model("best_vgg16_model.h5")
except Exception as e:
    messagebox.showerror("Error", f"Model loading failed: {str(e)}")
    exit()

# Disease class names
disease_classes = ['diseased', 'diseased', 'diseased', 'diseased', 'diseased', 'Normal']

# Confidence threshold for unknown classification
UNKNOWN_THRESHOLD = 0.7

# Function to preprocess image
def path_to_tensor(img_path, width=128, height=128):  
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    return x

def classify_image(img_array):
    """ Classifies the image and returns the predicted label with confidence. """
    pred = model2.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    
    if confidence < UNKNOWN_THRESHOLD or pred_class >= len(disease_classes):
        return "diseased", confidence
    else:
        return disease_classes[pred_class], confidence


def real_time_detection():
    cap = cv2.VideoCapture(0)  
    last_detected_disease = None  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_resized = cv2.resize(frame, (128, 128))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        label, confidence = classify_image(img_array)
        
        if label != "diseased":
            last_detected_disease = label  
        
        display_text = f"{label}: {confidence:.4f}"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)
        
        cv2.imshow("Real-Time Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and last_detected_disease:
            cap.release()  
            cv2.destroyAllWindows()  
            open_causes_file(last_detected_disease) 
            return  

    cap.release()
    cv2.destroyAllWindows()


def predict_image():
    rep = filedialog.askopenfilename()
    if not rep:
        return
    
    img = path_to_tensor(rep)
    label, confidence = classify_image(img)

    new_img = Image.open(rep)
    new_img = new_img.resize((200, 200), Image.LANCZOS)
    new_photo = ImageTk.PhotoImage(new_img)
    
    image_label.configure(image=new_photo)
    image_label.image = new_photo  
    image_label.place(x=540, y=180)  
    
    result_text.place(x=450, y=500)
    result_text.delete("1.0", tk.END)  
    result_text.insert(tk.END, f'The image is predicted as: {label}')
    
    if label != "diseased":
        open_causes_file(label)


def open_causes_file(disease_name):
    try:
        causes_folder = "causes"
        cause_file_path = os.path.join(causes_folder, f"{disease_name}.txt")

        if os.path.exists(cause_file_path):
            with open(cause_file_path, "r", encoding="utf-8") as file:
                cause_text = file.read()
            
            causes_window = tk.Toplevel()
            causes_window.title(f"Causes of {disease_name}")
            causes_window.geometry("600x400")
            
            text_widget = tk.Text(causes_window, wrap=tk.WORD, font=("Arial", 12))
            text_widget.insert(tk.END, cause_text)
            text_widget.pack(expand=True, fill=tk.BOTH)
        else:
            messagebox.showerror("Error", f"File not found: {cause_file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def open_main_window():
    global result_text, image_label  

    main_window = tk.Tk()
    main_window.title("Plant Disease Detection")
    main_window.geometry("800x600")  
    
    bg_img = Image.open("backk.jpg")
    bg_img = bg_img.resize((1300, 700), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = tk.Label(main_window, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
    
    title_label = tk.Label(main_window, text="Plant Disease Prediction", font=("Times New Roman", 20, "bold"), fg="green")
    title_label.place(x=490, y=50)
    
    predict_button = tk.Button(main_window, text="Predict", command=predict_image, font=("Arial", 14), bg="lightblue")
    predict_button.place(x=600, y=390)
    
    real_time_button = tk.Button(main_window, text="Real-Time Detection", command=real_time_detection, font=("Arial", 14), bg="lightgreen")
    real_time_button.place(x=535, y=450)
    
    result_text = tk.Text(main_window, height=2, width=40, font=("Arial", 12))
    result_text.place_forget()
    
    image_label = tk.Label(main_window)
    image_label.place(x=200, y=200)  
    
    main_window.mainloop()

# Open main window directly
open_main_window()

