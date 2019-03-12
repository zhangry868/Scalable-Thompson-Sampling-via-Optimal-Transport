# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Bayesian NN using Particle-based Bayesian Sampling.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pdb

from absl import flags
from bandits.core.contextual_dataset import ContextualDataset
from bandits.core.bayesian_nn import BayesianNN

FLAGS = flags.FLAGS

# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.algorithms.svgd_neural import svgd_bayesnn
from bandits.algorithms.svgd_neural import ensemble_nn
from bandits.algorithms.svgd_neural import dgf_nn


class PiposteriorBNNSampling(BanditAlgorithm):
  """Posterior Sampling algorithm based on a Bayesian neural network."""

  def __init__(self, name, hparams, bnn_model='SVGD', M = 20):
    """Creates a PosteriorBNNSampling object based on a specific optimizer.

    The algorithm has two basic tools: an Approx BNN and a Contextual Dataset.
    The Bayesian Network keeps the posterior based on the optimizer iterations.

    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters of the algorithm.
      bnn_model: Type of BNN. By default RMSProp (point estimate).
    """

    self.name = name
    self.hparams = hparams
    self.optimizer_n = hparams.optimizer

    self.training_freq = hparams.training_freq
    self.training_epochs = hparams.training_epochs
    self.t = 0
    self.data_h = ContextualDataset(hparams.context_dim, hparams.num_actions,
                                    hparams.buffer_s)

    # to be extended with more BNNs (BB alpha-div, GPs, SGFS, constSGD...)
    bnn_name = '{}-bnn'.format(name)
    self.bnn = ParticleNeuralBanditModel(hparams, bnn_name, bnn_model, M)
    
  def action(self, context):
    """Selects action for context based on Thompson Sampling using the BNN."""

    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      # round robin until each action has been taken "initial_pulls" times
      return self.t % self.hparams.num_actions
    
    ######## Predictions ########
    c = context.reshape((1, self.hparams.context_dim))
    output = self.bnn.predict(c)
    return np.argmax(output)

  def update(self, context, action, reward):
    """Updates data buffer, and re-trains the BNN every training_freq steps."""

    self.t += 1
    self.data_h.add(context, action, reward)
    
    if self.t % self.training_freq == 0:
      if self.hparams.reset_lr:
        self.bnn.assign_lr()
      self.bnn.train(self.data_h, self.training_epochs)

class ParticleNeuralBanditModel(BayesianNN):
  """Implements an approximate Bayesian NN using Particle optimization."""

  def __init__(self, hparams, name="piBNN", bnn_model='SVGD', M = 20):

    self.name = name
    self.hparams = hparams

    self.n_in = self.hparams.context_dim
    self.n_out = self.hparams.num_actions
    self.layers = self.hparams.layer_sizes
    self.init_scale = self.hparams.init_scale
    self.f_num_points = None
    if "f_num_points" in hparams:
      self.f_num_points = self.hparams.f_num_points

    self.times_trained = 0
    self.cleared_times_trained = self.hparams.cleared_times_trained
    self.initial_training_steps = self.hparams.initial_training_steps
    self.training_schedule = np.linspace(self.initial_training_steps,
                                         self.hparams.training_epochs,
                                         self.cleared_times_trained)
    self.M = M
    self.verbose = getattr(self.hparams, "verbose", True)
    self.build_model(bnn_model)

#############################REQUIRED#############################
  def build_model(self, bnn_model, activation_fn=tf.nn.relu):
    """Defines the actual NN model with fully connected layers.

    The loss is computed for partial feedback settings (bandits), so only
    the observed outcome is backpropagated (see weighted loss).
    Selects the optimizer and, finally, it also initializes the graph.

    Args:
      activation_fn: the activation function used in the nn layers.
    """

    if self.verbose:
      print("Initializing model {}.".format(self.name))

    print(self.M)

    # Build network.
    if bnn_model =='SVGD':
      self.model = svgd_bayesnn(context_dim=self.n_in, num_actions=self.n_out, M = self.M)
    elif bnn_model =='DGF':
      self.model = dgf_nn(context_dim=self.n_in, num_actions=self.n_out, M = self.M)
    elif bnn_model =='Ensemble':
      self.model = ensemble_nn(context_dim=self.n_in, num_actions=self.n_out, M = self.M)

  def assign_lr(self):
    """Resets the learning rate in dynamic schedules for subsequent trainings.

    In bandits settings, we do expand our dataset over time. Then, we need to
    re-train the network with the new data. The algorithms that do not keep
    the step constant, can reset it at the start of each *training* process.
    """

    decay_steps = 1
    if self.hparams.activate_decay:
      current_gs = self.sess.run(self.global_step)
      self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr,
                                            self.global_step - current_gs,
                                            decay_steps,
                                            self.hparams.lr_decay_rate)
                                              
#############################REQUIRED#############################
  def train(self, data, num_steps):
    """Trains the BNN for num_steps, using the data in 'data'.

    Args:
      data: ContextualDataset object that provides the data.
      num_steps: Number of minibatches to train the network for.

    Returns:
      losses: Loss history during training.
    """

    if self.times_trained < self.cleared_times_trained:
      num_steps = int(self.training_schedule[self.times_trained])
    self.times_trained += 1

    losses = []

    if self.verbose:
      print("Training {} for {} steps...".format(self.name, num_steps))

    for step in range(num_steps):
      x, y, weights = data.get_batch_with_weights(self.hparams.batch_size)
      # pdb.set_trace()
      loss = self.model.train(x, y, weights)
      losses.append(loss)

    if step % self.hparams.freq_summary == 0:
      if self.hparams.show_training:
        print("{} | step: {}, loss: {}".format(self.name, global_step, loss))
        self.summary_writer.add_summary(summary, global_step)

    return losses

  def predict(self, data):
    return self.model.predict(data)
