import torch
import torch.nn.functional as F
import numpy as np


class SimpleMLP(torch.nn.Module):
	def __init__(self, input_shape):
		super(SimpleMLP, self).__init__()
		self.lin1 = torch.nn.Linear(np.asarray(input_shape)[1:].prod(), 512)
		self.lin2 = torch.nn.Linear(512, 128)
		self.lin3 = torch.nn.Linear(128, 1)

	def forward(self, x):
		x = x.view(-1, num_flat_features(x))
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		x = x.view(-1)
		return x

class LargeCNN(torch.nn.Module):
	def __init__(self, input_shape):
		super(LargeCNN, self).__init__()

		self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2, padding=(4,4))
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=(4,4))
		self.bn2 = torch.nn.BatchNorm2d(64)
		self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=(2,2))
		self.bn3 = torch.nn.BatchNorm2d(128)
		self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=(2,2))
		self.bn4 = torch.nn.BatchNorm2d(128)

		self.pool = torch.nn.MaxPool2d(2, 2)

		self.lin1 = torch.nn.Linear(7680, 128)
		self.lin2 = torch.nn.Linear(128, 128)
		self.lin3 = torch.nn.Linear(128, 1)

	def forward(self, x):
		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn2(self.conv2(x))))
		x = self.pool(F.relu(self.bn3(self.conv3(x))))
		x = self.pool(F.relu(self.bn4(self.conv4(x))))
		x = x.view(-1, num_flat_features(x))
		x = F.dropout(self.lin1(x), p=0.3, training=True)
		x = F.dropout(self.lin2(x), p=0.5, training=True)
		x = self.lin3(x)
		x = x.view(-1)
		return x

class SmallCNN(torch.nn.Module):
	def __init__(self, input_shape):
		super(SmallCNN, self).__init__()

		self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2, padding=(4,4))
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=(4,4))
		self.bn2 = torch.nn.BatchNorm2d(64)
		self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=(2,2))
		self.bn3 = torch.nn.BatchNorm2d(128)

		self.pool = torch.nn.MaxPool2d(2, 2)

		self.lin1 = torch.nn.Linear(7680, 128)
		self.lin2 = torch.nn.Linear(128, 1)

	def forward(self, x):
		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn2(self.conv2(x))))
		x = self.pool(F.relu(self.bn3(self.conv3(x))))
		x = x.view(-1, num_flat_features(x))
		x = F.dropout(x, p=0.2)
		x = F.dropout(self.lin1(x), p=0.3, training=True)
		x = F.dropout(self.lin2(x), p=0.5, training=True)
		x = x.view(-1)
		return x

def num_flat_features(x):
	size = x.size()[1:]  # all dimensions except the batch dimension
	num_features = 1
	for s in size:
		num_features *= s
	return num_features