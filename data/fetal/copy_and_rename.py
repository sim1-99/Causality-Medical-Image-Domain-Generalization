import os
import shutil

images_folder = '/home/schiarella/nnUNet_data/nnUNet_raw/Dataset005_Brains/imagesTr'
labels_folder = '/home/schiarella/nnUNet_data/nnUNet_raw/Dataset005_Brains/labelsTr'
destination_folder = '/home/schiarella/Causality-Medical-Image-Domain-Generalization/FeTA_24_GIN-IPA/B'

# Copy all files from source folder to destination folder
for filename in os.listdir(images_folder):
    if "kcl" in filename or "SB" in filename:
        source_file = os.path.join(images_folder, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_folder)

for filename in os.listdir(labels_folder):
    if "kcl" in filename or "SB" in filename:
        source_file = os.path.join(labels_folder, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_folder)

# Sort files in the destination folder alphabetically
sorted_files = sorted(os.listdir(destination_folder))

# Rename files based on the specified conditions
image_count = 150
label_count = 150

for filename in sorted_files:
    file_path = os.path.join(destination_folder, filename)
    if "_0000" in filename:
        new_filename = f"image_{image_count}.nii.gz"
        image_count += 1
        print(filename, " renamed as: ", new_filename)
    else:
        new_filename = f"label_{label_count}.nii.gz"
        label_count += 1
        print(filename, " renamed as: ", new_filename)
    print("image count: ", image_count, "label count: ", label_count)

    new_file_path = os.path.join(destination_folder, new_filename)
    os.rename(file_path, new_file_path)