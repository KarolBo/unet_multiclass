from glob import glob
from os.path import join, basename
from os import rename
from PIL import Image
import matplotlib.pyplot as plt


def clean_filenames(*folders):
    for folder in folders:
        img_list = glob(join(folder, '*.png'))
        for filename in img_list:
            new_filename = filename.replace('_lesions_benign', '')
            new_filename = new_filename.replace('_lesions_malignant', '')
            new_filename = new_filename.replace('_microcalc_benign', '')
            new_filename = new_filename.replace('_microcalc_malignant', '')
            rename(filename, new_filename)


def fill_gaps_in_labels(img_folder, label_folders):
    img_list = glob(join(img_folder, '*.png'))
    for label_folder in label_folders:
        label_list = glob(join(label_folder, '*.png'))
        label_idx = 0
        for img_path in img_list:
            img_name = basename(img_path)
            if basename(label_list[label_idx]) == img_name:
                label_idx += 1
            else:
                with Image.open(img_path) as img:
                    dummy_label = Image.new(img.mode, img.size)
                    dummy_label.save(join(label_folder, img_name))


def validate_generator(gen):
    images, labels = gen.__getitem__(0)
    for x, y in zip(images, labels):
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(x, cmap='gray')
        ax[1].imshow(y, cmap='gray')
        plt.show()

# clean_filenames('./data/lesions_malignant', './data/microcalc_benign')
# fill_gaps_in_labels('./data/images',
#                     ['./data/lesions_malignant', './data/microcalc_benign'])
