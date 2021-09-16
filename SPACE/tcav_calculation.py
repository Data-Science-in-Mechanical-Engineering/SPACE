"""
Copyright 2018 Google LLC
Modifications copyright 2021 Lukas Kreisköther

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
import tensorflow.keras.backend as K
import SPACE.model as tcav_model
import SPACE.tcav as tcav
import SPACE.utils as utils
import SPACE.activation_generator as act_gen
import tensorflow as tf
import SPACE.utils_plot as utils_plot
import numpy as np


class TCAVImageModelWrapper(tcav_model.ImageModelWrapper):
    """Modified version of PublicImageModelWrapper in TCAV's models.py.
    For more information on class and the classes methods: https://github.com/tensorflow/tcav"""

    def __init__(self,
                 sess,
                 labels,
                 image_shape,
                 input_tensor,
                 output_tensor,
                 activation_tensor_tcav,
                 activation_tensor_tcav_name):
        super(self.__class__, self).__init__(image_shape)

        self.sess = sess
        self.labels = labels
        self.model_name = ''  # name does not matter for us

        # load the graph from the backend
        graph = tf.get_default_graph()

        # get endpoint tensors
        self.ends = {'input': graph.get_tensor_by_name(input_tensor),
                     'prediction': graph.get_tensor_by_name(output_tensor)}

        # set bottleneck tensors
        self.bottlenecks_tensors = {activation_tensor_tcav_name:
            graph.get_tensor_by_name(
                activation_tensor_tcav)}

        # Construct gradient ops.
        with graph.as_default():
            self.y_input = tf.placeholder(tf.int64, shape=[None])

            self.pred = tf.expand_dims(self.ends['prediction'][0], 0)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.one_hot(
                        self.y_input,
                        self.ends['prediction'].get_shape().as_list()[1]),
                    logits=self.pred))

        self._make_gradient_tensors()

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)

    @staticmethod
    def get_bottleneck_tensors(bottleneck):
        """Add Inception bottlenecks and their pre-Relu versions to endpoints dict.
        Bottleneck layers should be layers that the output entirely depends on.
        """
        graph = tf.get_default_graph()
        bn_endpoints = {}
        for op in graph.get_operations():
            if bottleneck in op.type:
                name = op.name.split('/')[0]
                bn_endpoints[name] = op.outputs[0]
        return bn_endpoints

    def adjust_prediction(self, pred_t):
        # return vector with estimated class probabilities
        return pred_t

    def _make_gradient_tensors(self):
        """Makes gradient tensors for all bottleneck tensors.
        """
        print("self.loss: ", self.loss)
        print("self.bt: ", self.bottlenecks_tensors)

        self.bottlenecks_gradients = {}
        for bn in self.bottlenecks_tensors:
            self.bottlenecks_gradients[bn] = tf.gradients(
                self.loss, self.bottlenecks_tensors[bn])[0]

    def get_gradient(self, acts, y, bottleneck_name, example):
        """Return the gradient of the loss with respect to the bottleneck_name.

        Args:
          acts: activation of the bottleneck
          y: index of the logit layer
          bottleneck_name: name of the bottleneck to get gradient wrt.
          example: input example. Unused by default. Necessary for getting gradients
            from certain models, such as BERT.

        Returns:
          the gradient array.
        """

        return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
            self.bottlenecks_tensors[bottleneck_name]: acts,
            self.y_input: y
        })

    def reshape_activations(self, layer_acts):
        """Reshapes layer activations as needed to feed through the model network.
        Override this for models that require reshaping of the activations for use
        in TCAV.
        Args:
          layer_acts: Activations as returned by run_examples.
        Returns:
          Activations in model-dependent form; the default is a squeezed array (i.e.
          at most one dimensions of size 1).
        """
        return np.asarray(layer_acts).squeeze()


class TCAVScoreCalculator:
    """Capsules the whole TCAV functionality. Based on random images stored in folders with names random500_i
    (default value defined by TCAV framework) TCAV scores for all defined concepts will be calculated"""
    def __init__(self,
                 model_str,
                 source_dir,
                 num_random_exps,
                 labels,
                 target,
                 concepts,
                 image_shape,
                 max_examples,
                 scale_mode,
                 resize_mode,
                 padding_val,
                 activation_tensor_tcav,
                 activation_tensor_tcav_name,
                 activation_tensor_tcav_type,
                 input_tensor='x:0',
                 output_tensor='cf/StatefulPartitionedCall/activation/Softmax:0'):
        """Initializes a TCAVScoreCalculator object.
        Args:
            model_str (str): path to the frozen graph model.
            source_dir (str): path to the work directory with images to analyze (in target folder),
            potential concepts to test and folders of random concept images.
            num_random_exps (int): number of random concepts which will be used for TCAV.
            labels (list of str): list of strings corresponding to output nodes of the model (first string in list
            corresponds to first output node etc.).
            target (str): string defining the class for which TCAV will be performed (one of strings from labels)
            concepts (list of str | None): list which defines concepts to be tested with TCAV
            (e.g. concepts = ['concept1', 'concept2', 'concept3']). If None, concepts to be tested will be determined
            automatically (all folders which are not containing 'random', 'tcav_class_test' or the name of any label
            will be used as concept folders).
            image_shape (list of int): list with len=2, which defines the shape of the input images.
            max_examples (int): maximmal number of examples which will be loaded randomly from the defined image folders
            from TCAV framework. TCAV framework sometimes has memory issues when using too many (big) images for one
            target or concept. Reducing this number (max_examples=30 should normally work for our use cases) should help
            then.
            scale_mode (str): defines the mode for scaling the images as preprocessing. Can be 'symm' (scale to [-1,1])
            or 'asymm' (scale to [0,1]).
            resize_mode (str): defines the mode for resizing concept images ('bilinear': bilinear interpolation,
            'padding': center padding, 'random_padding': padding with random position of concept images in image plane,
            'tiling': tiling with symmetric filling of the edges).
            padding_val ((int, int, int) | None): tuple, which defines the rgb value for padding. Can be None if no
            padding is used.
            activation_tensor_tcav (str): name of the tensor of the bottleneck layer for TCAV.
            activation_tensor_tcav_name (str): name of the bottleneck layer for TCAV.
            activation_tensor_tcav_type (str): type of the bottleneck tensor for TCAV.
            input_tensor (str): name of the model input tensor.
            output_tensor (str): name of the model output tensor.
        """

        self.model_str = model_str
        self.source_dir = source_dir
        self.num_random_exps = num_random_exps
        self.labels = labels
        self.target = target
        self.concepts = concepts
        self.image_shape = image_shape
        self.max_examples = max_examples
        self.scale_mode = scale_mode
        self.resize_mode = resize_mode
        self.padding_val = padding_val
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.activation_tensor_tcav = activation_tensor_tcav
        self.activation_tensor_tcav_name = activation_tensor_tcav_name
        self.activation_tensor_tcav_type = activation_tensor_tcav_type

    def run(self):
        """Method to call to start the TCAV calculation.
        Returns: list: tcav results.
        """
        tf.keras.backend.clear_session()

        # get tensorflow.keras session from backend
        sess = K.get_session()

        # load frozen graph model
        with tf.gfile.FastGFile(self.model_str, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        # insert folder names for images (target, concepts, random) and potentially saved CAVs/activations
        source_dir = self.source_dir
        project_name = source_dir + '/tcav_class_test'
        working_dir = project_name
        activation_dir = working_dir + '/activations/'
        cav_dir = working_dir + '/cavs/'

        utils.make_dir_if_not_exists(activation_dir)
        utils.make_dir_if_not_exists(working_dir)
        utils.make_dir_if_not_exists(cav_dir)

        # parameters for training (value 'alphas' is used for training the linear classifier)

        # constant that multiplies the regularization term of the cost function of the linear classifier
        # for more information:
        # (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
        alphas = [0.1]
        num_random_exp = self.num_random_exps

        # insert target class to examine. Only one per run possible
        target = self.target

        # insert concepts
        if self.concepts is None:
            concepts = [name for name in os.listdir(self.source_dir) if
                        (os.path.isdir(self.source_dir + '/' + name)
                         and (not 'random' in name)
                         and (not 'tcav_class_test' in name)
                         and (not name in self.labels))]
        else:
            concepts = self.concepts

        # insert layers to be examined
        bottlenecks = [self.activation_tensor_tcav_name]  # name of bottleneck layer
        bottleneck = self.activation_tensor_tcav_type  # type of bottleneck layer

        # insert label names corresponding to the outputnodes (same order)
        labels = self.labels
        image_shape = [self.image_shape[0], self.image_shape[0], 3]
        max_examples = self.max_examples
        scale_mode = self.scale_mode
        resize_mode = self.resize_mode
        padding_val = self.padding_val

        # get instance of model wrapper
        mymodel = TCAVImageModelWrapper(sess=sess, labels=labels,
                                        image_shape=image_shape, input_tensor=self.input_tensor,
                                        output_tensor=self.output_tensor,
                                        activation_tensor_tcav=self.activation_tensor_tcav,
                                        activation_tensor_tcav_name=self.activation_tensor_tcav_name)

        # create your image activation generator
        act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=max_examples,
                                                         normalize_image=True, resize_mode=resize_mode,
                                                         scale_mode=scale_mode, padding_val=padding_val)

        # create your TCAV experiment
        mytcav = tcav.TCAV(sess=sess,
                           target=target, concepts=concepts, bottlenecks=bottlenecks,
                           activation_generator=act_generator, alphas=alphas,
                           cav_dir=cav_dir, num_random_exp=num_random_exp)

        # show results
        results = mytcav.run(run_parallel=False)
        utils_plot.plot_results(results, num_random_exp=num_random_exp)

        # return tcav results
        return results
