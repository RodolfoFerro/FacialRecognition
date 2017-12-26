# ===============================================================
# Author: Rodolfo Ferro Pérez
# Email: ferro@cimat.mx
# Twitter: @FerroRodolfo
#
# Script: Image downloader using ImageSoup.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ===============================================================

from imagesoup import ImageSoup
from tqdm import tqdm
import os


# Set number of images and terms to look for:
n_images = 100
terms = "Barack Obama"

# Define paths:
db_path = "../db/original/"

# Clean folders:
os.system("rm ../db/original/*")
os.system("rm ../db/train/*")

# Create soup and search
print("Looking for images...")
soup = ImageSoup()
images = soup.search('"{}"'.format(terms), n_images=n_images)

# Save trainin percentage (train_per) of images in training folder:
n_images = len(images)

print("Downloading images of '{}'...".format(terms))
for i in tqdm(range(n_images)):
    try:
        images[i].to_file(train_path + "img_{:0>4}.jpg".format(i + 1))
    except Exception:
        pass
