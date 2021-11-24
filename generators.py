import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


class MyGenerator(Sequence):
    def __init__(self, batch_size, train_path, image_folder, mask_folders, aug_dict, target_size, seed=101):
        self.image_iterator = self._build_dir_iterator(batch_size, train_path, image_folder,
                                                       aug_dict, target_size, seed)
        self.mask_iterators = []
        for mask_folder in mask_folders:
            iterator = self._build_dir_iterator(batch_size, train_path, mask_folder,
                                                aug_dict, target_size, seed)
            self.mask_iterators.append(iterator)

    def _build_dir_iterator(self, batch_size, train_path, subfolder, aug_dict, target_size, seed):
        datagen = ImageDataGenerator(rescale=1./255,
                                     **aug_dict)

        dir_iterator = datagen.flow_from_directory(train_path,
                                                   classes=[subfolder],
                                                   class_mode=None,
                                                   color_mode='grayscale',
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   seed=seed)

        return dir_iterator

    def __len__(self):
        return self.image_iterator.__len__()

    def __getitem__(self, idx):
        x = self.image_iterator.__getitem__(idx)
        y = np.zeros(x.shape)

        for cls, mask_iterator in enumerate(self.mask_iterators):
            y_temp = mask_iterator.__getitem__(idx)
            y[y_temp > 0] = cls + 1

        return x, y
