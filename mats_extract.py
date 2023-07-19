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
    # inverts the image colors
    inverted_image = cv2.bitwise_not(img)
    # turns into grayscale
    bnw_inverted_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

    # sharpened
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpen = cv2.filter2D(bnw_inverted_image, -1, sharpen_kernel)
    sharp_bnw_inverted_image = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # removes noise
    kernel = np.ones((1, 1), np.uint8)
    nr_bnw_inverted_image = cv2.dilate(sharp_bnw_inverted_image, kernel, iterations=1)
    nr_bnw_inverted_image = cv2.erode(nr_bnw_inverted_image, kernel, iterations=1)
    nr_bnw_inverted_image = cv2.morphologyEx(nr_bnw_inverted_image, cv2.MORPH_CLOSE, kernel)
    nr_bnw_inverted_image = cv2.medianBlur(nr_bnw_inverted_image, 3)

    blur_nr_bnw_inverted_image = cv2.bilateralFilter(nr_bnw_inverted_image,9,75,75)

    # removes borders
    # contours, heirarchy = cv2.findContours(nr_bnw_inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    # cnt = cntsSorted[-1]
    # x, y, w, h = cv2.boundingRect(cnt)
    # nb_nr_bnw_inverted_image = nr_bnw_inverted_image[y:y+h, x:x+w]

    blur_nr_bnw_inverted_image_path = target + "\\" + image_name + "NB_NR_BNW_inverted.png"
    cv2.imwrite(blur_nr_bnw_inverted_image_path, blur_nr_bnw_inverted_image)
    return blur_nr_bnw_inverted_image_path

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

def noise_removal(image_name, img_path, target):
    image = cv2.imread(img_path)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    nr_image = cv2.medianBlur(image, 3)
    nr_image_path = target + "\\" + image_name + "NR.png"
    cv2.imwrite(nr_image_path, nr_image)
    return (nr_image_path)

def remove_borders(image):
    contours, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


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
            name_top_left = (pt[0] - name_dimensions[0], pt[1] - name_dimensions[1])
            name_bottom_right = (pt[0] + count_dimensions[0], pt[1])
            count_top_left = (pt[0] + 150, pt[1] + 20)
            count_bottom_right = (pt[0] + count_dimensions[0], pt[1] + count_dimensions[1])

            if name_top_left[0] >= 0 and name_top_left[1] >= 0 and name_bottom_right[0] <= image.shape[1] and name_bottom_right[1] <= image.shape[0]:
                # Crop the image
                cropped_image = image[name_top_left[1]:name_bottom_right[1], name_top_left[0]:name_bottom_right[0]]
                cropped_image_path = target + "\\" + f"{name_top_left}_name_cropped.png"
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cv2.imwrite(cropped_image_path, cropped_image)
            
            if count_top_left[0] >= 0 and count_top_left[1] >= 0 and count_bottom_right[0] <= image.shape[1] and count_bottom_right[1] <= image.shape[0]:
                # Crop the image
                cropped_image = image[count_top_left[1]:count_bottom_right[1], count_top_left[0]:count_bottom_right[0]]
                cropped_image_path = target + "\\" + f"{count_top_left}_count_cropped.png"
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cv2.imwrite(cropped_image_path, cropped_image)
    else:
        print("No matches found.")

def convert_image_to_text(images_dir, crops_dir, preprocessing_dir):
    for filename in os.listdir(images_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full file path
            file_path = os.path.join(images_dir, filename)

            search_and_crop_image(file_path, held_img, material_name_dimensions, material_count_dimensions, crops_dir)

    for filename in os.listdir(crops_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full file path
            file_path = os.path.join(crops_dir, filename)

            # inverted and bnw
            preprocessed_file_path = invert(filename, file_path, preprocessing_dir)
            print(preprocessed_file_path)
            extract_text_from_image(preprocessed_file_path)


            

# Set the directory containing the images
images_dir = 'E:\Code\FGO_mats_extraction\screenshots'
crops_dir = 'E:\Code\FGO_mats_extraction\crops'
preprocessing_dir = 'E:\Code\FGO_mats_extraction\preprocess'
held_img = 'E:\Code\FGO_mats_extraction\screenshots\held.jpg'
material_name_dimensions = (20, 84)
material_count_dimensions = (400, 77)

convert_image_to_text(images_dir, crops_dir, preprocessing_dir)