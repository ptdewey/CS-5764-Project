# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optimizers
import torch.nn as nn # Neural network layers
from torch.nn.functional import relu, softmax, log_softmax # Activation functions 
import torchvision

# DATA INGESTION
trainingSplit = torchvision.datasets.MNIST("../assets/Datasets/", download = True, 
	train = True, transform = torchvision.transforms.ToTensor())
testingSplit = torchvision.datasets.MNIST("../assets/Datasets/", download = True, 
	train = False, transform = torchvision.transforms.ToTensor())

# DATA WRANGLING
trainingData = torch.utils.data.DataLoader(trainingSplit, batch_size = 10, shuffle = True)
testingData = torch.utils.data.DataLoader(testingSplit, batch_size = 10, shuffle = False)

# NEURAL NETWORK
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(28*28, 10**2) # Entry to HL1
		self.fc2 = nn.Linear(10**2, 8**2) # HL1 to HL2
		self.fc3 = nn.Linear(8**2, 5**2) # HL2 to HL3
		self.fc4 = nn.Linear(5**2, 10) # HL3 to Output

	def forwardPropagation(self, x):
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))
		x = relu(self.fc3(x))
		x = self.fc4(x)
		return x
	
	def train(self, trainingData: torch.utils.data.dataloader.DataLoader, 
		numberOfEpochs: int, costFunction: torch.nn.modules.loss, 
		optimizer: torch.optim.Optimizer, learningRate: float):
		   
		optimizer = optimizer(self.parameters(), learningRate)
		costs = []

		for epoch in range(numberOfEpochs):
			optimizer.zero_grad() # Remove gradients from previous backprop
			cost = 0.0 # Not sure if this is needed because of zero_grad()
			for observation in trainingData:
				image, label = observation
				prediction = self.forwardPropagation(image.view(-1, 784))
				loss = costFunction(prediction, label)
				cost += loss

			# Drop weights
			np.save("../assets/Weights/Entry-HL1/Epoch_" + str(epoch + 1), self.fc1._parameters["weight"].detach().numpy())
			np.save("../assets/Weights/HL1-HL2/Epoch_" + str(epoch + 1), self.fc2._parameters["weight"].detach().numpy())
			np.save("../assets/Weights/HL2-HL3/Epoch_" + str(epoch + 1), self.fc3._parameters["weight"].detach().numpy())
			
			costs.append(cost)
			print("Cost: ", costs[epoch])
			cost.backward()
			optimizer.step()

network = NeuralNetwork()
network.zero_grad()
network.train(trainingData, 100, nn.CrossEntropyLoss(), optimizers.Adam, 0.01)

for epoch in range(0, 100):
	plt.subplots(10, 10, figsize = (25, 25), dpi = 300)

	for j in range(0, 10**2):
		plt.subplot(10, 10, j + 1)
		data = np.load("../assets/Weights/Entry-HL1/Epoch_" + str(epoch + 1) + ".npy")
		image = data[j, :].reshape((28, 28))
		plt.imshow(image)
		plt.suptitle("Epoch " + str(epoch + 1), y = 0.99)
		plt.title("Entry into HL1 - Neuron " + str(j + 1))
	plt.tight_layout()
	plt.savefig("../assets/Weights/Entry-HL1/Epoch_" + str(epoch + 1) + ".png", dpi = 300)
	plt.close()

for epoch in range(0, 100):
	plt.subplots(8, 8, figsize = (25, 25), dpi = 300)

	for j in range(0, 8**2):
		plt.subplot(8, 8, j + 1)
		data = np.load("../assets/Weights/HL1-HL2/Epoch_" + str(epoch + 1) + ".npy")
		image = data[j, :].reshape((10, 10))
		plt.imshow(image)
		plt.suptitle("Epoch " + str(epoch + 1), y = 0.99)
		plt.title("HL1 into HL2 - Neuron " + str(j + 1))
	plt.tight_layout()
	plt.savefig("../assets/Weights/HL1-HL2/Epoch_" + str(epoch + 1) + ".png", dpi = 300)
	plt.close()

for epoch in range(0, 100):
	plt.subplots(5, 5, figsize = (25, 25), dpi = 300)

	for j in range(0, 5**2):
		plt.subplot(5, 5, j + 1)
		data = np.load("../assets/Weights/HL2-HL3/Epoch_" + str(epoch + 1) + ".npy")
		image = data[j, :].reshape((5, 5))
		plt.imshow(image)
		plt.suptitle("Epoch " + str(epoch + 1), y = 0.99)
		plt.title("HL2 into HL3 - Neuron " + str(j + 1))
	plt.tight_layout()
	plt.savefig("../assets/Weights/HL2-HL3/Epoch_" + str(epoch + 1) + ".png", dpi = 300)
	plt.close()



