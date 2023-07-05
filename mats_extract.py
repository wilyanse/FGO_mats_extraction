import pytesseract
from PIL import Image
import os

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Extract text from image and store in Google Sheets
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    # Perform OCR on the image
    text = pytesseract.image_to_string(image)
    if text:
        print(text)
    else:
        print('No text found in the image.')

# Set the directory containing the images
images_dir = 'E:\Code\FGO_mats_extraction\screenshots'

# Iterate over all image files in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Construct the full file path
        file_path = os.path.join(images_dir, filename)

        print("file: " + filename)

        extract_text_from_image(file_path)