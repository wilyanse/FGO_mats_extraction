import pytesseract
from PIL import Image
import os
import cv2
from cv2 import dnn_superres
import numpy as np

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


def search_and_crop_image(image_path, search_image_path, name_dimensions, count_dimensions, target):
    # Load the images
    image = cv2.imread(image_path)
    search_image = cv2.imread(search_image_path)

    if image is None or search_image is None:
        print("Failed to load image files.")
        return

    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)

    # Find the search image in the original image
    result = cv2.matchTemplate(gray_image, gray_search_image, cv2.TM_CCOEFF_NORMED)

    # Set a threshold to filter out weaker matches
    threshold = 0.9
    loc = np.where(result >= threshold)

    if loc[0].size > 0:
        # Perform cropping for each match
        for pt in zip(*loc[::-1]):
            # Get the coordinates of the matched region
            top_left = (pt[0] - name_dimensions[0], pt[1] - name_dimensions[1])
            bottom_right = (pt[0] + count_dimensions[0], pt[1] + count_dimensions[1])

            if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] <= image.shape[1] and bottom_right[1] <= image.shape[0]:
                # Crop the image
                cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                cropped_image_path = target + "\\" + f"{top_left}_cropped.png"
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cv2.imwrite(cropped_image_path, cropped_image)
    else:
        print("No matches found.")

def convert_image_to_text(images_dir, preprocessing_dir):
    for filename in os.listdir(images_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full file path
            file_path = os.path.join(images_dir, filename)

            search_and_crop_image(file_path, held_img, material_name_dimensions, material_count_dimensions, preprocessing_dir)

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
            bnw_invert_file_path = invert(filename + "bnw", bnw_file_path, preprocessing_dir)
            print(bnw_invert_file_path)
            extract_text_from_image(bnw_invert_file_path)

# Example usage

# Set the directory containing the images
images_dir = 'E:\Code\FGO_mats_extraction\screenshots'
preprocessing_dir = 'E:\Code\FGO_mats_extraction\preprocess'
held_img = "E:\Code\FGO_mats_extraction\screenshots\held.jpg"
material_name_dimensions = (20, 84)
material_count_dimensions = (450, 80)

convert_image_to_text(images_dir, preprocessing_dir)