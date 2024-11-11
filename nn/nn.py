from collections.abc import Iterator
from . import utils
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence()))

class Neuron:
  def __init__(self, num_weights: int):
    self.weights = self._build_weights(num_weights)
    self.activation = 0.0
    self.bias = (0.5 - (-0.5))*rs.rand() + (-0.5)
    self.delta = 0.0

  def __repr__(self) -> str:
    return f"Neuron(activation={self.activation}, num_weights={len(self.weights)})"

  def _build_weights(self, n: int) -> list[float]:
    # random floats between -0.5 and 0.5
    return (0.5 - (-0.5))*rs.rand(n) + (-0.5)

class Layer:
  def __init__(self, size: int, num_inp_weights: int):
    self.neurons = [Neuron(num_inp_weights) for _ in range(size)]

  def __repr__(self):
    return f"Layer(size={len(self.neurons)}, neurons={self.neurons})"

  def __iter__(self) -> Iterator[Neuron]:
    return self.neurons.__iter__()

  def __len__(self) -> int:
    return len(self.neurons)

class Network:
  def __init__(self,
               x_train,
               x_test,
               y_train,
               y_test,
               hidden_layers: list[int],
               output_layer_size: int,
               learning_rate: float=0.1,
               pre_process: callable=None):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.hidden_layers = self._build_hidden_layers(hidden_layers)
    self.output_layer = self._build_output_layer(output_layer_size)
    self.learning_rate = learning_rate
    self.pre_process = pre_process
    self.act_func = utils.tanh
    self.d_act_func = utils.d_tanh

  def _build_hidden_layers(self, layers: list[int]) -> list[Layer]:
    hidden: list[Layer] = []
    for i, l in enumerate(layers):
      if i == 0:
        # num_inp_weights = size of input layer for first hidden layer
        hidden.append(
          Layer(size=l, num_inp_weights=len(np.concatenate(self.x_train[0])))
        )
        continue
      hidden.append(
        Layer(size=l, num_inp_weights=len(hidden[i-1].neurons))
      )
    return hidden

  def _build_output_layer(self, size) -> Layer:
    return Layer(size, len(self.hidden_layers[-1].neurons))

  def _backprop_output(self, target: list[float]) -> None:
    '''
    build error terms for each neuron of output layer for target training example
    '''
    for k, out_k in enumerate(self.output_layer):
      # derivative of activation function output times error
      err = (target[k] - out_k.activation)
      out_k.delta = self.d_act_func(out_k.activation) * err

  def _backprop_hidden(self, hidden_layer: Layer, prev_layer: Layer) -> None:
    for h, n_h in enumerate(hidden_layer):
      sum_prev_deltas = 0
      for prev_k in prev_layer:
        w_hk = prev_k.weights[h]
        d_w_hk = w_hk * prev_k.delta
        sum_prev_deltas += d_w_hk
      # derivative of the activation function output times sum of the 
      # prev layer error times connecting weight
      n_h.delta = self.d_act_func(n_h.activation) * sum_prev_deltas

  def _update_weights(self, training_input: list[float]) -> None:
    inputs = []
    layers = self.hidden_layers + [self.output_layer]
    for k, layer in enumerate(layers):
      if k == 0:
        inputs = training_input
      else:
        inputs = [n.activation for n in self.hidden_layers[k-1]]
      for n in layer:
        # update bias
        n.bias += n.delta * self.learning_rate
        for j, w_ij in enumerate(n.weights):
          # new weight is old weight plus learning rate times corresponding 
          # input times the error of the neuron
          n.weights[j] = w_ij + (self.learning_rate * inputs[j] * n.delta)

  def _feed_forward(self, inp_activations: list[float], current: Layer) -> None:
    for n in current:
      dot = np.dot(n.weights, inp_activations) + n.bias
      # activation function
      n.activation = self.act_func(dot)

  def _predict(self, inp: list[float]) -> list[float]:
    '''
    returns raw activations of output layer
    '''
    self._feed_forward(inp, self.hidden_layers[0])
    for i, layer in enumerate(self.hidden_layers):
      if i+1 >= len(self.hidden_layers):
        self._feed_forward([n.activation for n in self.hidden_layers[-1]], self.output_layer)
        continue
      self._feed_forward([n.activation for n in layer], self.hidden_layers[i+1])
    return [n.activation for n in self.output_layer]

  def _train(self, start: int, end: int) -> None:
    for i, t in enumerate(self.x_train[start:end]):
      label = self.y_train[i+start]
      inp = self.pre_process(t) if self.pre_process else inp
      self._predict(inp)
      target = [0]*len(self.output_layer)
      target[label] = 1
      self._backprop_output(target)
      j = len(self.hidden_layers)-1
      prev_layer = self.output_layer
      while j >= 0:
        self._backprop_hidden(self.hidden_layers[j], prev_layer)
        prev_layer = self.hidden_layers[j]
        j -= 1
      self._update_weights(inp)

  def train(self, batches: int, batch_size: int) -> None:
    for i in range(batches):
      start = i * batch_size
      end = (i+1) * batch_size
      if start > 0: start += 1
      print(f"starting training batch {i+1}, using training slice [{start}, {end}]")
      self._train(start, end)

  def predict_dist(self, inp: any) -> list[float]:
    '''
    returns softmax prob distribution of output values
    '''
    inp = self.pre_process(inp) if self.pre_process else inp
    p = self._predict(inp)
    return utils.softmax(p)

  def predict_one(self, inp: any) -> int:
    inp = self.pre_process(inp) if self.pre_process else inp
    p = utils.softmax(self._predict(inp))
    choice = 0
    for i, _ in enumerate(p):
      if p[i] > p[choice]: choice = i
    return choice
