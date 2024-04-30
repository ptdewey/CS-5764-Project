# IMPORT PACKAGES
import matplotlib.pyplot as plt # Data visualization
import numpy as np # Linear algebra
import os # OS interfacing
import pandas as pd # Dataframe manipulation
import shutil # File operations
import sklearn.datasets as datasets # Datasets
import sys # CLI arguments
import torch # Tensors
import torch.optim as optimizers # Neural network optimizers
import torch.nn as nn # Neural network layers
import torch.nn.functional as functions # Layer functions (e.g., ReLU, Softmax)

# DATA INESTION
dataset = sys.argv[1]
rawData = pd.read_csv("../assets/Datasets/" + str(dataset) + ".csv", sep = ',')

# DATA WRANGLING
X = torch.tensor(rawData[['x', 'y']].to_numpy()).float()
y = torch.tensor(rawData['cluster'].to_numpy())
data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), 
	shuffle = True)
del(X, y)

# MODELING
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(2, 100) # Entry to Output
		#self.fc2 = nn.Linear(10, 10)
		#self.fc3 = nn.Linear(10, 10)
		#self.fc4 = nn.Linear(10, 10)
		#self.fc5 = nn.Linear(10, 10)
		self.fc6 = nn.Linear(100, 2)

	def forwardPropagation(self, x):
		x = functions.relu(self.fc1(x))
		#x = functions.sigmoid(self.fc2(x))
		#x = functions.sigmoid(self.fc3(x))
		#x = functions.sigmoid(self.fc4(x))
		#x = functions.sigmoid(self.fc5(x))
		x = self.fc6(x)
		return x
	
	def train(self, trainingData: torch.utils.data.dataloader.DataLoader, 
		numberOfEpochs: int, costFunction: torch.nn.modules.loss, 
		optimizer: torch.optim.Optimizer, learningRate: float):

		optimizer = optimizer(self.parameters(), learningRate)
		costs = np.zeros(numberOfEpochs) # Store overall cost at each epoch

		# if os.path.exists("contourfigs/") and os.path.isdir("contourfigs/"):
		# 	shutil.rmtree("contourfigs")
		
		# os.mkdir("contourfigs/")

		for epoch in range(numberOfEpochs):

			# 1. Compute cost
			optimizer.zero_grad() # Remove gradients from previous backprop
			cost = 0.0 # Not sure if this is needed because of zero_grad()
			for observation in trainingData:
				X, label = observation
				prediction = self.forwardPropagation(X.view(-1, 2))
				loss = costFunction(prediction, label)
				cost += loss
			costs[epoch] = cost
			#print("Cost: ", costs[epoch])

			# 2. Generate decisiom boundaries
			with torch.no_grad():
				# Generate grid domain
				x = np.linspace(rawData['x'].min() - 1, rawData['x'].max() + 1, 
					num = 100)
				y = np.linspace(rawData['y'].min() - 1, rawData['y'].max() + 1, 
					num = 100)
				x, y = np.meshgrid(x, y)
				inputSpace = 
					torch.tensor(np.vstack([x.flatten(), y.flatten()]).T)

				# Compute predicted class probabilities
				outputSpace = self.predict(inputSpace.float()).numpy()
				outputSpace = np.max(outputSpace, axis = 1).reshape(x.shape[0], 
					x.shape[0])

				# Create melted data frame
				# currentEpoch = np.ones((x.shape[0] * y.shape[0], 1)) * (epoch + 1)
				# decisionBoundary = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), outputSpace, currentEpoch])
				# decisionBoundaries.append(decisionBoundary)
	
				plt.contourf(x, y, outputSpace, cmap = "RdBu")
				plt.scatter(rawData['x'], rawData['y'], c = rawData['cluster'], cmap = "RdBu", edgecolors = "black")
				plt.title("Epoch " + str(epoch + 1) + "\nCost: " + str(round(costs[epoch], ndigits = 2)))
				plt.xlabel("$x$")
				plt.ylabel("$y$")
				plt.savefig("../assets/Decision-Boundaries/" + str(dataset).capitalize() + "/Epoch_" + str(epoch + 1) + ".png", dpi = 300)
			
			# 3. Backpropagate and update weights
			cost.backward()
			optimizer.step()

	def predict(self, x: torch.tensor):
		with torch.no_grad():
			return(self.forwardPropagation(x))

network = NeuralNetwork()
network.zero_grad()
network.train(data, 100, nn.CrossEntropyLoss(), optimizers.Adam, 0.01)


