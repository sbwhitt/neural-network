import numpy as np
import nn
from nn.mnist_loader import MnistDataloader

loader = MnistDataloader(
  "data/train-images-idx3-ubyte/train-images-idx3-ubyte",
  "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
  "data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
  "data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
)
(x_train, y_train), (x_test, y_test) = loader.load_data()

print(len(x_train), "training examples")
print(len(x_test), "testing examples")
network = nn.Network(
  x_train=x_train,
  x_test=x_test,
  y_train=y_train,
  y_test=y_test,
  hidden_layers=[32, 32],
  output_layer_size=10
)

batch_size = 60000
print("\ntraining with batch size", batch_size)
network.train(batch_size)

total = len(x_test)
correct = 0
choice_counts = [0]*10
correct_counts = [0]*10
for i, test in enumerate(x_test):
  inp = np.concatenate(test) / 255
  label = y_test[i]
  res = network.predict(inp)
  choice = 0
  for i, r in enumerate(res):
    if r > res[choice]: choice = i
  choice_counts[choice] = choice_counts[choice]+1
  correct_counts[label] = correct_counts[label]+1
  if choice == label:
    correct += 1
  # else:
  #   print("incorrect choice", choice, "for label", label)

print()
print(f"{correct} out of {total} correct, score: {(correct/total)*100}%")
print("choice counts", choice_counts)
print("correct counts", correct_counts)
