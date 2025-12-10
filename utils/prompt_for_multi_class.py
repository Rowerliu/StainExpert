import os
from glob import glob

def create_txt_files(image_folder):
    image_files = glob(os.path.join(image_folder, "*.png"))

    for image_file in image_files:
        image_name = os.path.basename(image_file)
        cls = image_name.split("_")[0]
        txt_name = image_name.replace('.png', '.txt')
        txt_path = os.path.join(image_folder, txt_name)
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f'picture of a {cls}')

image_folder = r'F:\01_Data\01_acrobat\Acrobat_512_Multi\03_HE-A-IHC-B\test_B'
create_txt_files(image_folder)