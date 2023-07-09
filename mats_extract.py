import pytesseract
from PIL import Image
import os
import cv2
from cv2 import dnn_superres

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

def invert(image_name, img_path, target):
    img = cv2.imread(img_path)
    inverted_image = cv2.bitwise_not(img)
    inverted_image_path = target + "\\" + image_name + "inverted.png"
    cv2.imwrite(inverted_image_path, inverted_image)
    return inverted_image_path

def grayscale(image_name, img_path, target):
    img = cv2.imread(img_path)
    bnw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bnw_image_path = target + "\\" + image_name + "bnw.png"
    cv2.imwrite(bnw_image_path, bnw_image)
    return bnw_image_path


def upscaling(image_name, img_path, target):
    sr = dnn_superres.DnnSuperResImpl_create()
    image = cv2.imread(img_path)
    path = "E:\Code\FGO_mats_extraction\EDSR_x2.pb"
    sr.readModel(path)
    sr.setModel("edsr", 2)
    upscaled_image = sr.upsample(image)
    upscaled_image_path = target + "\\" + image_name + "up.png"
    cv2.imwrite(upscaled_image_path, upscaled_image)
    return upscaled_image_path


# Set the directory containing the images
images_dir = 'E:\Code\FGO_mats_extraction\screenshots'
preprocessing_dir = 'E:\Code\FGO_mats_extraction\preprocess'

# Iterate over all image files in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Construct the full file path
        file_path = os.path.join(images_dir, filename)

        print("file: " + filename)
        print(file_path)
        extract_text_from_image(file_path)

        # invert
        inverted_file_path = invert(filename, file_path, preprocessing_dir)
        print(inverted_file_path)
        extract_text_from_image(inverted_file_path)

        # bnw
        bnw_file_path = grayscale(filename, file_path, preprocessing_dir)
        print(bnw_file_path)
        extract_text_from_image(bnw_file_path)

        # bnw -> invert
        bnw_invert_filepath = invert(filename + "bnw", bnw_file_path, preprocessing_dir)
        print(bnw_invert_filepath)
        extract_text_from_image(bnw_file_path)