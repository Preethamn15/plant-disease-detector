
import cv2
import numpy as np
from tkinter import Tk, filedialog


# Open file dialog to select an image
file_path = filedialog.askopenfilename(title="Select a Plant Leaf Image",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png")])

# Load the image (replace this line with your image path or use filedialog)
image = cv2.imread(file_path)
original = image.copy()

# Resize for better visibility (optional)
image = cv2.resize(image, (512, 512))
original = image.copy()

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for dark/black spots
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 70])  # V < 70 means it's dark

# Create mask for black spots
mask = cv2.inRange(hsv, lower_black, upper_black)

# Morphological operations to remove noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours of dark spots
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw circles around small dark spots
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 5 < area < 300:  # Only tiny holes
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(original, (int(x), int(y)), int(radius), (0, 255, 0), 2)

# Show results
cv2.imshow("Original with Circles", original)
cv2.imshow("Dark Area Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


