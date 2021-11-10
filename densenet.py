import torch 
import torch.nn as nn 



class Densenet(nn.Module):
	def __init__(self, input_size = (256, 256), N_class = 5):
		super().__init__()
		self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
		self.model.classifier = torch.nn.Linear(in_features = 1024, out_features = N_class, bias = True)
		#self.model.activatef = torch.nn.Sigmoid()

	def forward(self, x):
		return self.model(x)

