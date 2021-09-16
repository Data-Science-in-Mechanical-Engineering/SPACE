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

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import PIL
import tensorflow as tf
import numpy as np
import PIL.Image
import math
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn import preprocessing
from sklearn.decomposition import PCA


def preprocess_input(img):
    """Adds 4-th dimension to image tensor and scales image values from [0,255] to [-1,1].
    Args:
        img (numpy.ndarray[float]): image to preprocess.

    Returns:
        numpy.ndarray[float]: preprocessed image tensor.
    """
    img = np.expand_dims(img, axis=0)
    img = ((img / 127.5) - 1)
    return img


def unscale_input(img):
    """Reverses scaling of image values from [-1,1] to [0,255].
    Args:
        img (numpy.ndarray[float]): image to scale

    Returns:
        numpy.ndarray[float]: unscaled image
    """
    return ((img + 1) * 127.5)


def unpreprocess_input(img):
    """Reverses prepocessing of images by squeezing first dimension of the image
    tensor and scaling of image values from [-1,1] to [0,255].
    Args:
        img (numpy.ndarray[float]): image to unpreprocess.

    Returns:
        numpy.ndarray[float]: unprepocessed image.
    """
    img = np.squeeze(img, axis=0)
    img = np.interp(img, (img.min(), img.max()), (0, 255))
    return img.astype('uint8')


def grad_cam(g,
             sess,
             image,
             category_index,
             nb_classes,
             target_size,
             conv_tensor_name,
             input_tensor_name='x:0',
             output_tensor_name='cf/StatefulPartitionedCall/dense_1/MatMul:0'):
    """Implementation of Grad-CAM (https://arxiv.org/abs/1610.02391) for frozen graph models based on tf1.
    In short this function produces a heatmap with values corresponding to weighted (the weights represent the averaged
    importance of a feature map of the defined convolutional layer wrt to the choosen class) feature map values of
    the defined convolutional layer.
    Args:
        g (tensorflow.Graph): graph of the model we are working with.
        sess (tensorflow.Session): tensorflow session we are working with.
        image (numpy.ndarray[float]): the image we want to apply gradcam on.
        category_index (numpy.ndarray[int]): the class index we want our heatmap for.
        nb_classes (int): number of classes.
        target_size (list of int): list with len=2 defining the size we want our heatmap in.
        conv_tensor_name (str): name of the tensor corresponding to the concolutional layer we want to use for gradcam
            (quote from gradcam paper: "We find that Grad-CAM maps become progressively worse as we move to earlier
            convolutional layers as they have smaller receptive fields and only focus on less semantic local features."
            --> use e.g. last concolutional layer).
        input_tensor_name (str): name of the input tensor of the model.
        output_tensor_name (str): name of the output tensor of the model.

    Returns:
        (numpy.ndarray[int], numpy.ndarray[float]): with
            np.uint8(cam): map for visualizing with e.g. matplotlib.
            heatmap: tensor with shape (1, target_size[0], target_size[1]) containing heat values from 0 (not important)
            to 1 (very important) for every pixel wrt to the defined class.
    """
    one_hot = tf.sparse_to_dense(category_index, [nb_classes], 1.0)
    signal = tf.multiply(g.get_tensor_by_name(output_tensor_name), one_hot)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, g.get_tensor_by_name(conv_tensor_name))[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run(
        [g.get_tensor_by_name(conv_tensor_name), norm_grads],
        feed_dict={g.get_tensor_by_name(input_tensor_name): image})
    output = output[0]
    grads_val = grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.array(PIL.Image.fromarray(cam).resize(target_size, PIL.Image.BILINEAR), dtype=np.float32)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cm = plt.get_cmap('coolwarm')
    cam = cm(np.uint8(255 * heatmap))
    cam = (cam[:, :, :3] * 255).astype(np.uint8)
    cam = np.float32(cam) + np.float32(unscale_input(image))
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


class ClusterSearcher:
    """Capsules the whole functionality from building possible concept images as image patches based on input images,
    calculating the importance of these patches, resizing of the patchtes and then clustering of activations
    of these important image patches from all images to analyze in the bootleneck layer defined. This whole
    functionality will be applied by running the run method.
    """

    def __init__(self,
                 segments_per_image,
                 epsilon_dbscan,
                 min_samples_dbscan,
                 image_shape,
                 slices_per_axis,
                 num_classes,
                 model_str,
                 activation_tensor_gradcam,
                 activation_tensor_tcav,
                 load_format,
                 store_format,
                 load_path,
                 store_path,
                 cluster_method='optics',
                 distance_metric='manhattan',
                 standardize=True,
                 dim_pca=30,
                 input_tensor='x:0',
                 output_tensor='cf/StatefulPartitionedCall/activation/Softmax:0'):
        """Initializes a ClusterSearch object.
        Args:
            segments_per_image (float): defines the fraction of image patches with heat value > 0 to consider for
            possible concepts. More important possible concept images
            (what means bigger heat value) will be chosen first.
            epsilon_dbscan (int): epsilon value for dbscan clustering (only important when dbscan should be applied).
            min_samples_dbscan (int): min_samples value for dbscan clustering
            (only important when dbscan should be applied).
            image_shape (list of int): list with len=2 which defines the shape of the input images
            (needed for building graphs).
            slices_per_axis (int): number of slices per axis the image will be sliced in
            ("slices_per_axis" has to divide "image_shape[0]" by an integer (assumption: image is square) ).
            num_classes (int): number of classes for the model in usage (needed for gradcam).
            model_str (str): path to the frozen graph model.
            activation_tensor_gradcam (str): name of the tensor used for gradcam.
            activation_tensor_tcav (str): name of the tensor used as bottleneck for clustering and TCAV.
            load_format (str): format of images to load (e.g. 'png').
            store_format (str): format for storing the resulting concept images (e.g. 'png').
            load_path (str): path to the work directory with images to analyze.
            store_path (str): path to the work directory for storing of folders of possible concept images.
            cluster_method (str): defines the cluster method to use ('optics' or 'dbscan').
            distance_metric (str): defines distance metric used for clustering (must be one of the
            options allowed by sklearn.metrics.pairwise_distances)
            standardize (boolean): decides if standardization is applied or not.
            dim_pca (int | None): resulting dimension for pca dimension reduction
            (if None no standardization will be applied).
            input_tensor (str): name of the model input tensor.
            output_tensor (str): name of the model output tensor.
        """
        self.segments_per_image = segments_per_image
        self.epsilon_dbscan = epsilon_dbscan
        self.min_samples_dbscan = min_samples_dbscan
        self.image_shape = image_shape
        self.slices_per_axis = slices_per_axis
        self.num_classes = num_classes
        self.model_str = model_str
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.activation_tensor_gradcam = activation_tensor_gradcam
        self.activation_tensor_tcav = activation_tensor_tcav
        self.load_format = load_format
        self.store_format = store_format
        self.load_path = load_path
        self.store_path = store_path
        self.cluster_method = cluster_method
        self.distance_metric = distance_metric
        self.standardize = standardize
        self.dim_pca = dim_pca

        # shall divide image size by an integer
        assert ((self.image_shape[
                     0] % self.slices_per_axis) == 0), '"slices_per_axis" has to divide "image_shape[0]" by an' \
                                                       'integer (assumption: image is square)'
        self.slice_length = int(self.image_shape[0] / self.slices_per_axis)

        # prepare operation for patch extraction
        self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, self.image_shape[0], self.image_shape[1], 3))
        patches = tf.image.extract_patches(images=self.img_placeholder,
                                           sizes=[1, self.slice_length, self.slice_length, 1],
                                           strides=[1, self.slice_length, self.slice_length, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')

        # prepare operation for reshaping
        self.patches = tf.reshape(patches,
                                  shape=(
                                      patches.shape[1] * patches.shape[2], self.slice_length,
                                      self.slice_length,
                                      3))

    def run(self):
        """Method to call to start a cluster search run consisting of production of potentially important concept
        images, resizing (via tiling) of these patches, clustering of the activations of these resized patches in
        the defined bottleneck layer and storing the images of the regarding clusters in folders in the work directory.
        Returns: number of clusters found.
        """
        # get list with the names of the images to examine
        list_input_imgs = [name for name in os.listdir(self.load_path) if
                           not os.path.isdir(self.load_path + '/' + name)]

        # load frozen graph model
        with tf.gfile.FastGFile(self.model_str, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        g = tf.get_default_graph()
        sess = tf.Session()

        # define lists for storing important image patches from images
        res_list = []

        # produce squared equidistant segmentation mask
        list_arr = []
        # from top to bottom
        for i in range(self.slices_per_axis):
            offset = i * self.slices_per_axis  # e.g. 0, 8, 16, ...
            list_row = []
            # from left to right
            for j in range(self.slices_per_axis):
                list_row.append(np.full(shape=(self.slice_length, self.slice_length), fill_value=int(j + offset)))
            list_arr.append(np.hstack(list(list_row[i] for i in range(len(list_row)))))
        segments_tiles = np.vstack(list(list_arr[i] for i in range(len(list_arr))))

        # produce list with resized most important image patches
        for filename in list_input_imgs:
            img = np.array(PIL.Image.open(tf.gfile.Open(self.load_path + '/' + filename, 'rb')).convert('RGB').resize(self.image_shape, PIL.Image.BILINEAR),
                           dtype=np.float32) #todo: resize

            preprocessed_input = preprocess_input(img)

            # produce heatmap
            predictions = sess.run(g.get_tensor_by_name(self.output_tensor),
                                   feed_dict={g.get_tensor_by_name(self.input_tensor): preprocessed_input})
            predicted_class = np.argmax(predictions)
            cam, heatmap = grad_cam(g=g, sess=sess, image=preprocessed_input, category_index=predicted_class,
                                    conv_tensor_name=self.activation_tensor_gradcam,
                                    input_tensor_name=self.input_tensor, output_tensor_name=self.output_tensor,
                                    nb_classes=self.num_classes, target_size=self.image_shape)

            img_unprepro = unpreprocess_input(preprocessed_input)

            with tf.Session():
                patches = self.patches.eval(feed_dict={self.img_placeholder: np.expand_dims(img_unprepro, axis=0)})

            # build list of tuples with (image patch of segment, average heat value of the segment)
            segs_list = []

            for i in range(patches.shape[0]):
                seg_mask = np.full((segments_tiles.shape[0], segments_tiles.shape[1]), i)
                seg_heat = np.where(np.equal(seg_mask, segments_tiles), heatmap, 0)
                if np.count_nonzero(seg_heat) > 0:
                    value_heat = np.sum(seg_heat) / np.count_nonzero(seg_heat)
                    segs_list.append((patches[i], value_heat))

            # order segments regarding their average importance
            segs_list_sorted = sorted(segs_list, key=lambda t: t[1], reverse=True)

            # tile patches to input size
            for i in range(math.ceil((self.segments_per_image * len(segs_list_sorted)))):
                seg = segs_list_sorted[i][0]
                i = np.tile(seg, (int(self.image_shape[0] / seg.shape[0]), int(self.image_shape[1] / seg.shape[1]), 1))
                img = np.pad(array=i,
                             pad_width=(
                                 (0, self.image_shape[0] % i.shape[0]), (0, self.image_shape[1] % i.shape[1]), (0, 0)),
                             mode='wrap')
                res_list.append(img)

        # get activations of bottleneck layer for resized patches
        list_act_bn = []
        for act in res_list:
            act_bn = sess.run(g.get_tensor_by_name(self.activation_tensor_tcav),
                              feed_dict={g.get_tensor_by_name(self.input_tensor): preprocess_input(np.array(act))})
            list_act_bn.append(act_bn.flatten())

        # standardize activations for clustering
        if self.standardize == True:
            list_act_bn = list(preprocessing.scale(np.array(list_act_bn)))
            print('Standardization done!')

        if self.dim_pca is not None:
            pca = PCA(n_components=self.dim_pca)
            list_act_bn = list(pca.fit_transform(np.array(list_act_bn)))
            print('Explained variation through PCA: {}'.format(sum(pca.explained_variance_ratio_)))

        # apply dense based clustering
        if self.cluster_method == 'dbscan':
            clusters_res = DBSCAN(eps=self.epsilon_dbscan,
                                  min_samples=self.min_samples_dbscan,
                                  metric=self.distance_metric).fit(np.array(list_act_bn))

        elif self.cluster_method == 'optics':
            clusters_res = OPTICS(metric=self.distance_metric).fit(np.array(list_act_bn))

        else:
            raise ValueError('Define appropriate clustering method!')
        print('Number of clusters: ', int(len(np.unique(clusters_res.labels_))))

        # check if outliers were found
        if -1 in clusters_res.labels_:
            print('Outliers found!')
            # store images (outliers were found)
            for i in range(len(np.unique(clusters_res.labels_)) - 1):
                mask = np.full((clusters_res.labels_.shape[0]), i)
                arr = np.array(res_list)[np.equal(mask, clusters_res.labels_)]
                os.mkdir(self.store_path + '/concept_' + str(i))
                for counter, img in enumerate(arr):
                    PIL.Image.fromarray(img.astype(np.uint8)).save(
                        self.store_path + '/concept_' + str(i) + '/' + 'img_' + str(
                            counter) + '.' + self.store_format)
            os.mkdir(self.store_path + '/outliers')
            mask = np.full((clusters_res.labels_.shape[0]), -1)
            arr = np.array(res_list)[np.equal(mask, clusters_res.labels_)]
            for counter, img in enumerate(arr):
                PIL.Image.fromarray(img.astype(np.uint8)).save(
                    self.store_path + '/outliers' + '/' + 'img_' + str(counter) + '.' + self.store_format)

        else:
            # store images (no outliers were found)
            print('No outliers found!')
            for i in range(len(np.unique(clusters_res.labels_))):
                mask = np.full((clusters_res.labels_.shape[0]), i)
                arr = np.array(res_list)[np.equal(mask, clusters_res.labels_)]
                os.mkdir(self.store_path + '/concept_' + str(i))
                for counter, img in enumerate(arr):
                    PIL.Image.fromarray(img.astype(np.uint8)).save(
                        self.store_path + '/concept_' + str(i) + '/' + 'img_' + str(
                            counter) + '.' + self.store_format)

        # return num of clusters
        return len(np.unique(clusters_res.labels_))
