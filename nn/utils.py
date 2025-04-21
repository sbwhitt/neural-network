import numpy as np
import matplotlib.pyplot as plt

def show_images(images: list[list[np.ndarray]], titles: list[str]):
  '''
  from: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
  '''
  cols = 5
  rows = int(len(images)/cols) + 1
  plt.figure(figsize=(30,20))
  index = 1
  for x in zip(images, titles):
    image = x[0]
    title_text = x[1]
    plt.subplot(rows, cols, index)
    plt.imshow(image, cmap=plt.cm.gray)
    if (title_text != ''):
      plt.title(title_text, fontsize=15);
    index += 1

def tanh(x: float) -> float:
  return np.tanh(x)

def d_tanh(tanh_x: float) -> float:
  return (1 - tanh_x**2)

def relu(x: float) -> float:
  return 0 if x < 0 else x

def d_relu(relu: float) -> float:
  r = 0 if relu <= 0 else 1
  return r

def sigmoid(x: float) -> float:
  return 1 / (1 + np.exp(-x))

def d_sigmoid(sigmoid: float) -> float:
  return sigmoid * (1 - sigmoid)

def softmax(z: list[float]) -> list[float]:
  return np.exp(z) / sum(np.exp(z))
