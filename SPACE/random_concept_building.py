"""
Copyright 2021 Lukas Kreisköther

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import PIL.Image
import numpy as np
import os
import random


class RandomConceptBuilder:
    """RandomConceptBuilder objects capsule the functionality for building random concept images necessary for using the
    TCAV framework in industrial usecases. For that random crops from defined sets of images (e.g. from good class
    when testing the bad class) with size crop_size are build. The random concept images are stored in folders
    with name prefix 'random500_' so that they can be used for the TCAV framework.
    """

    def __init__(self, path, folders_for_building, store_fmt, image_shape, crop_size, num_fold=30,
                 num_imgs_per_fold=100):
        """Initializes a RandomConceptBuilder object.
        Args:
            path (str): path which leads to the directory in which the folders are laying based upon which the random
            concept images should be build (e.g. '/home/lukas/Documents/02_Data/FGUSS_subsets_grey/').
            folders_for_building (list of str): list of strings for all folders in the directory from which the algorithm should
            choose images to build the random concept images (e.g. ['good'] or ['one', 'two', 'three'])
            image_shape (list of int): list with len=2 which defines the shape the produced images should have
            (normally equals the input size of the model to investigate).
            crop_size (list of int): list with len=3 defining the size of the random crops (e.g. [56, 56, 3]).
            num_fold (int): number of folders of random concept images the algorithm should build.
            num_imgs_per_fold (int): number of images per folder for the folders of random concept images.
            store_fmt (str): store format of produced images.
        """
        self.path = path
        self.folders_for_building = folders_for_building
        self.name_prefix = 'random500_'
        self.store_fmt = store_fmt
        self.image_shape = image_shape
        self.crop_size = crop_size
        self.num_fold = num_fold
        self.num_imgs_per_fold = num_imgs_per_fold
        if len(self.folders_for_building) == 1:
            self.X_names = [str(self.folders_for_building[0] + '/' + name) for name in
                            os.listdir(self.path + self.folders_for_building[0])
                            if not os.path.isdir(self.path + self.folders_for_building[0] + '/' + name)]
        else:
            X_temp = []
            for folder_name in self.folders_for_building:
                X_temp = X_temp + ([str(folder_name + '/' + name) for name in os.listdir(self.path + folder_name)
                                    if not os.path.isdir(self.path + self.folders_for_building[0] + '/' + name)])
            self.X_names = X_temp
        np.random.shuffle(self.X_names)
        self.img_tensor = tf.placeholder(tf.float32, shape=(self.image_shape[0], self.image_shape[1], 3))
        self.out = tf.image.random_crop(value=self.img_tensor, size=self.crop_size)

    def build_random_concept_image(self, img):
        """Method for building the random concept image from an input image.
        Args:
            img (numpy.ndarray[float]): image to build a random concept image from.

        Returns: PIL.Image: Random concept image as PIL.Image.
        """
        img = np.array(img, dtype=np.float32)
        with tf.Session():
            i = self.out.eval(feed_dict={self.img_tensor: img})
        i = np.tile(i, (int(img.shape[0] / i.shape[0]), int(img.shape[1] / i.shape[1]), 1))
        img = np.pad(array=i, pad_width=((0, img.shape[0] % i.shape[0]), (0, img.shape[1] % i.shape[1]), (0, 0)),
                     mode='wrap')
        return PIL.Image.fromarray(img.astype(np.uint8))

    def build(self):
        """Method to call to start building the concept images. Function looks how many
        images are already in the folders and fills the folders respectively.
        """
        for i in range(self.num_fold):
            sub_fold = self.name_prefix + str(i)
            if not os.path.isdir(self.path + sub_fold):
                try:
                    os.mkdir(self.path + sub_fold + '/')
                except Exception as e:
                    print("Creation of the directory %s failed" % sub_fold)
                    print(e)
                else:
                    print("Successfully created the directory %s " % sub_fold)
            num_files = len([name for name in os.listdir(self.path + sub_fold) if
                             os.path.isfile(os.path.join(self.path + sub_fold, name))])
            if not (num_files == self.num_imgs_per_fold):
                for j in range(self.num_imgs_per_fold - num_files):
                    img = random.choice(self.X_names)
                    img = np.array(PIL.Image.open(tf.gfile.Open(self.path + '/' + img, 'rb')).convert('RGB'),
                                   dtype=np.float32)
                    # todo: resize (right now, we don't do it since images have to be in right size for TCAV anyway)
                    img_ran = self.build_random_concept_image(img)
                    img_ran.save(self.path + sub_fold + '/' + str(num_files + j) + '.' + self.store_fmt)
