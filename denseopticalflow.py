import torch
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import normalize

from utils import show_video
from models import SimpleMLP, SmallCNN, LargeCNN

BATCH_SIZE = 128
NUM_EPOCHS = 5
model_class_map = {'largecnn': LargeCNN, 'smallcnn': SmallCNN, 'mlp': SimpleMLP}

class Farneback:

	def main(self, args):
		if args.new_data and os.path.exists("data/x_train.npy"):
			os.remove("data/x_train.npy")
			time.sleep(1)
			
		if not os.path.exists("data/x_train.npy"):
			print("No training data found, creating new preprocessed file using Farneback approach")
			self.preprocess(args.show_video, args.downsample_dim)
		
		# load preprocessed optical flow data
		print('load preprocessed training data')
		with open("data/x_train.npy", 'rb') as f:
			x_train = np.load(f).astype(np.float32)
		
		# load speed target variable data
		with open("data/train.txt", 'r') as f:
			y_train = np.loadtxt(f)[:-1].astype(np.float32)
		print(f'training data (x,y) shapes: {x_train.shape}, {y_train.shape}')
		
		# initialize model
		net = model_class_map[args.model](x_train.shape)
		# load weights if we resume training
		if args.resume_training:
			net.load_state_dict(torch.load('data/'), strict=False)
		
		self.train(x_train, y_train, net)

	def preprocess(self, show_vid=False, downsample_dim=0.5):
		cap = cv.VideoCapture("data/train.mp4")
		n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
		ret, frame = cap.read()
		# only use bottom half of image
		frame = frame[200:400] 
		# calculated downsampled dimensions
		dim = np.array(frame.shape[:-1])*downsample_dim
		dim = dim.astype(np.int32)
		# downsample image
		frame = cv.resize(frame, (dim[1], dim[0]), interpolation = cv.INTER_NEAREST)
		# make grayscale
		gray_0 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		# initialize empty data vector for result
		preprocessed_data = np.zeros((n_frames - 1, 2, dim[1], dim[0]), dtype=np.float32)
		
		for t in tqdm(range(n_frames - 1), desc='preprocessing data'):
			ret, frame = cap.read()
			# only use bottom half of image
			frame = frame[200:400]
			# downsample 
			frame = cv.resize(frame, (dim[1], dim[0]), interpolation = cv.INTER_NEAREST)
			# convert to grayscale
			gray_1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			
			# calc the optical flow vectors between two subsequent frames
			flow = cv.calcOpticalFlowFarneback(gray_0, gray_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

			# initialize visualization mask
			vis_mask = np.zeros_like(frame)
			
			# switch frames for next loop
			gray_0 = gray_1

			# convert vectors to polar data
			magnitude, angle = cv.cartToPolar(flow[...,0], flow[...,1])
			vis_mask[...,1] = 255
			# set image hue to vector direction
			vis_mask[...,0] = angle*180/np.pi/2
			# set color to vector magnitude
			vis_mask[...,2] = (magnitude*15).astype(np.int)

			preprocessed_data[t] = vis_mask[...,[0,2]].swapaxes(2,0)

			# visualize processed data
			if show_vid == True:
				show_video(vis_mask)

		# normalize channels using minmax
		preprocessed_data[:,0,...] *=  180
		preprocessed_data[:,1,...] *= 255

		print('saving preprocessed data...')
		np.save("data/x_train.npy", preprocessed_data)
		print('done!')

	def train(self, x, y, model):
		loss_fn = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

		# train-validation split
		val_idx = np.random.choice(np.arange(y.shape[0]), size=np.ceil(y.shape[0]*0.2).astype(int), replace=False)
		train_idx = list(set(np.arange(y.shape[0])) - set(val_idx))
		x_train, y_train = x[train_idx], y[train_idx]
		x_val, y_val = x[val_idx], y[val_idx]
		train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
		train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
		validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
		validation_loader = torch.utils.data.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=True)

		train_loss, val_loss = [], []
		for i in tqdm(range(NUM_EPOCHS), desc='epochs'):

			# training loop for one epoch
			pbar = tqdm(train_loader, desc=f'epoch {i}, training loss = {0.0}, validation loss = {0.0}')
			for x,y in pbar:
				optimizer.zero_grad()
				outputs = model.forward(x)
				loss = loss_fn(outputs, y)
				loss.backward()
				optimizer.step()
				loss = loss.item()
				train_loss.append(loss)

				# calculate loss on random batch from validation set
				x,y = next(iter(validation_loader))
				outputs = model.forward(x)
				loss = loss_fn(outputs, y)
				val_loss.append(loss)

				pbar.set_description(f'epoch {i}, training loss = {train_loss[-1]}, validation loss = {val_loss[-1]}')

			torch.save(model.state_dict(), "data/")
				
		# plot training graph
		plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='train loss')
		plt.plot(np.arange(len(val_loss)), val_loss, color='orange', label='validation loss')
		plt.xlabel('training step')
		plt.ylabel('mse')
		plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run dense optical flow using Farneback approach')
	parser.add_argument('--model', choices=['largecnn', 'smallcnn', 'mlp'], default='mlp', help='what model to use (largecnn, smallcnn, mlp)')
	parser.add_argument('--show_video', action='store_true', help='shows video during preprocessing')
	parser.add_argument('--resume_training', action='store_true', help='resumes training where it left off')
	parser.add_argument('--new_data', action='store_true', help='delete existing data and repeat preprocessing')
	parser.add_argument('--show_training_plot', action='store_true', help='plot training loss graph after training')
	parser.add_argument('--downsample_dim', type=float, default=0.5, help='ratio by which to downsample pixels in each dimension')
	args = parser.parse_args()
	fb = Farneback()
	fb.main(args)