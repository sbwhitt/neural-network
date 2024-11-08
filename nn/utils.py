import numpy as np
import matplotlib.pyplot as plt

def show_images(images: list[list[np.ndarray]], titles: list[str]):
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

def activation(x: float) -> float:
    return 1 if x > 0 else -1

def tanh(x: float) -> float:
  return np.tanh(x)

def d_tanh(tanh_x: float) -> float:
  return (1 - tanh_x**2)

# def d_tanh(x: float) -> float:
#   return 1 - (np.tanh(x)**2)

def relu(x: float) -> float:
  return x if x > 0 else 0

def d_relu(relu: float) -> float:
  return 1 if relu > 0 else 0

def softmax(z: list[float]) -> list[float]:
  return np.exp(z) / sum(np.exp(z))

def sum_squared_errors(outcome: list[float], desired: list[float]) -> float:
  return 0.5 * sum((np.array(outcome) - np.array(desired))**2)
