import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd

# function to generate laplacian
def process_laplacian(image):
    image = image.resize((600,400))
    image = np.asarray(image)
    image = image / 255.0
    return cv2.Laplacian(image, cv2.CV_64F)

# storing the directory of images
train_clear_images_dir = "CERTH_ImageBlurDataset/TrainingSet/Undistorted"
train_blur_images_dir1 = "CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred"
train_blur_images_dir2 = "CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred"
val_digital_images_dir = "CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet"
val_natural_images_dir = "CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet"

print(" Total Training Images Found ".center(100,"*"))
print("Undistorted Images:", len(os.listdir(train_clear_images_dir)))
print("Naturally-Blurred Images:", len(os.listdir(train_blur_images_dir1)))
print("Artificially-Blurred Images:", len(os.listdir(train_blur_images_dir2)))

# processing the images and generating the variance and maximum laplacian for of image
print()
print("Processing the Images in Training Directory...")
clear_img_max_laplacian = []
clear_img_var_laplacian = []
for i in os.listdir(train_clear_images_dir):
    
    image = Image.open(os.path.join(train_clear_images_dir, i)).convert('L')
    laplacian = process_laplacian(image)
    
    clear_img_max_laplacian.append(laplacian.max())
    clear_img_var_laplacian.append(laplacian.var())

print("Processing of clear images for training done...")


blurred_img_max_laplacian = []
blurred_img_var_laplacian = []
for i in os.listdir(train_blur_images_dir1):
    
    image = Image.open(os.path.join(train_blur_images_dir1, i)).convert('L')
    laplacian = process_laplacian(image)

    
    blurred_img_max_laplacian.append(laplacian.max())
    blurred_img_var_laplacian.append(laplacian.var())

for i in os.listdir(train_blur_images_dir2):
    
    image = Image.open(os.path.join(train_blur_images_dir2, i)).convert('L')
    laplacian = process_laplacian(image)
    
    blurred_img_max_laplacian.append(laplacian.max())
    blurred_img_var_laplacian.append(laplacian.var())


print("Processing of blur images for training done...")

print("Saving the data in train.csv file...")

labels = np.append(np.zeros(len(clear_img_max_laplacian)), np.ones(len(blurred_img_max_laplacian)))

laplacian_max = clear_img_max_laplacian + blurred_img_max_laplacian
laplacian_var = clear_img_var_laplacian + blurred_img_var_laplacian

train_data = pd.DataFrame({
    'Laplacian_Max': laplacian_max,
    'Laplacian_Var': laplacian_var,
    'Label': labels
})

train_data = train_data.sample(frac=1).reset_index(drop=True)

train_data.to_csv('train.csv', index = False)

print("Saving the data in train.csv file done...")


print("Processing the images in Validation Directory...")

validation_data1 = pd.read_excel("CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx")
validation_data2 = pd.read_excel("CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx")

validation_data1 = validation_data1.rename({"MyDigital Blur":"Images", "Unnamed: 1":"Labels"}, axis='columns')

validation_data2 = validation_data2.rename({"Image Name":"Images", "Blur Label":"Labels"}, axis='columns')

natural_clear_images = validation_data2.loc[validation_data2["Labels"]==-1, 'Images'].apply(lambda x: x.strip()+'.jpg').values
digital_clear_images = validation_data1.loc[validation_data1["Labels"]==-1, 'Images'].apply(lambda x: x.strip()).values

natural_blur_images = validation_data2.loc[validation_data2["Labels"]==1, 'Images'].apply(lambda x: x.strip()+'.jpg').values
digital_blur_images = validation_data1.loc[validation_data1["Labels"]==1, 'Images'].apply(lambda x: x.strip()).values

print()
print("Total Natural Clear Images Found: ", len(natural_clear_images))
print("Total Digital Clear Images Found: ", len(digital_clear_images))
print("Total Natural Blur Images Found: ", len(natural_blur_images))
print("Total Digital Blur Images Found: ", len(digital_blur_images))
print()

val_clear_img_max_laplacian = []
val_clear_img_var_laplacian = []

val_blur_img_max_laplacian = []
val_blur_img_var_laplacian = []

for i in natural_clear_images:
    image = Image.open(os.path.join(val_natural_images_dir, i)).convert('L')
    laplacian = process_laplacian(image)
    
    val_clear_img_max_laplacian.append(laplacian.max())
    val_clear_img_var_laplacian.append(laplacian.var())

for i in digital_clear_images:
    image = Image.open(os.path.join(val_digital_images_dir, i)).convert('L')
    laplacian = process_laplacian(image)
    
    val_clear_img_max_laplacian.append(laplacian.max())
    val_clear_img_var_laplacian.append(laplacian.var())

print("Processing of clear images for validation done...")

for i in natural_blur_images:
    image = Image.open(os.path.join(val_natural_images_dir, i)).convert('L')
    laplacian = process_laplacian(image)
    
    val_blur_img_max_laplacian.append(laplacian.max())
    val_blur_img_var_laplacian.append(laplacian.var())

for i in digital_blur_images:
    image = Image.open(os.path.join(val_digital_images_dir, i)).convert('L')
    laplacian = process_laplacian(image)

    
    val_blur_img_max_laplacian.append(laplacian.max())
    val_blur_img_var_laplacian.append(laplacian.var())

print("Processing of blur images for validation done...")

val_laplacian_max = val_clear_img_max_laplacian + val_blur_img_max_laplacian
val_laplacian_var = val_clear_img_var_laplacian + val_blur_img_var_laplacian

print("Saving the validation data in validation.csv file")

labels = np.append(np.zeros(len(val_clear_img_max_laplacian)), np.ones(len(val_blur_img_max_laplacian)))

val_data = pd.DataFrame({
    'Laplacian_Max': val_laplacian_max,
    'Laplacian_Var': val_laplacian_var,
    'Label': labels
})

val_data = val_data.sample(frac=1).reset_index(drop=True)

val_data.to_csv("validation.csv", index = False)

print("Saving the validation data done...")