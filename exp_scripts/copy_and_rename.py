import os
import shutil

source_folder = '/home/schiarella/nnUNet_data/FeTA_24_FT'
destination_folder = '/home/schiarella/nnUNet_data/FeTA_24_GIN-IPA'

# Copy all files from source folder to destination folder
for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)
    if os.path.isfile(source_file):
        shutil.copy(source_file, destination_folder)

# Sort files in the destination folder alphabetically
sorted_files = sorted(os.listdir(destination_folder))

# Rename files based on the specified conditions
image_count = 1
label_count = 1

for filename in sorted_files:
    file_path = os.path.join(destination_folder, filename)
    if "_T2w" in filename:
        new_filename = f"image_{image_count}.nii.gz"
        image_count += 1
        print(filename, " renamed as: ", new_filename)
    elif "_label" in filename:
        new_filename = f"label_{label_count}.nii.gz"
        label_count += 1
        print(filename, " renamed as: ", new_filename)
    print("image count: ", image_count, "label count: ", label_count)

    new_file_path = os.path.join(destination_folder, new_filename)
    os.rename(file_path, new_file_path)