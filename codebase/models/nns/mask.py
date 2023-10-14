import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
def dag_right_linear(input, weight, bias=None):
	if input.dim() == 2 and bias is not None:
		# fused op is marginally faster
		ret = torch.addmm(bias, input, weight.t())
	else:
		output = input.matmul(weight.t())
		if bias is not None:
			output += bias
		ret = output
	return ret
	
def dag_left_linear(input, weight, bias=None):
	if input.dim() == 2 and bias is not None:
		# fused op is marginally faster
		ret = torch.addmm(bias, input, weight.t())
	else:
		output = weight.matmul(input)
		if bias is not None:
			output += bias
		ret = output
	return ret



class MaskLayer(nn.Module):
	def __init__(self, n_topics=4,flag=0,z2_dim=4):
		super().__init__()
		self.n_topics = n_topics
		self.elu = nn.ELU()
		self.netlist=[]
		self.in_dim=self.n_topics
		self.z2_dim = z2_dim
		# if flag==1:
		# 	self.in_dim=1
		for i in range(self.n_topics):
			net = nn.Sequential(
				nn.Linear(self.z2_dim , self.z2_dim),
				nn.ELU(),
				nn.Linear(self.z2_dim, self.z2_dim),
			)
			self.netlist.append(net.to(device))
	def mix(self, z):
		zy = z.view(-1, self.n_topics*self.z2_dim)
		rx_list=[]
		zy_list = torch.split(zy, self.z2_dim, dim = 1)
		for i in range(self.n_topics):
			temp = self.netlist[i](zy_list[i])
			rx_list.append(temp)
		h = torch.cat(rx_list, dim=1)
		return h
   
   
class Attention(nn.Module):
	def __init__(self, in_features, bias=False):
		super().__init__()
		self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
		self.sigmd = torch.nn.Sigmoid()
	
	def attention(self, z, e):
		a = torch.mul(z.permute(0,2,1).matmul(self.M).permute(0,2,1),e)
		a = self.sigmd(a)
		A = torch.softmax(a, dim = 1)
		e = torch.mul(A,e)
		return e, A
	
class DagLayer(nn.Linear):
	def __init__(self, in_features, out_features,i = False, bias=False):
		super(Linear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.i = i
		self.a = torch.zeros(out_features,out_features)
		self.a = self.a
		self.A = nn.Parameter(self.a)
		self.b = torch.eye(out_features)
		self.b = self.b
		self.B = nn.Parameter(self.b)
		self.I = nn.Parameter(torch.eye(out_features))
		self.I.requires_grad=False
		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
			
	def mask_z(self,x):
		self.B = self.A
		x = torch.matmul(self.B.t(), x.view(-1,self.out_features,1)).view(-1,self.out_features)
		return x
		
	def mask_u(self,x):
		self.B = self.A
		x = x.view(-1, x.size()[1], 1)
		x = torch.matmul(self.B.t(), x.view(-1,self.out_features,1)).view(-1,self.out_features)
		return x
		

	def calculate_dag(self, x, v):
		if x.dim()>2:
			x = x.permute(0,2,1)
		x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias) 	   
		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
		return x,v
		
		
	def calculate_gaussian_ini(self, x, v):
		print(self.A)		
		if x.dim()>2:
			x = x.permute(0,2,1)
			v = v.permute(0,2,1)
		x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
		v = F.linear(v, torch.mul(torch.inverse(self.I - self.A),torch.inverse(self.I - self.A)), self.bias)
		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
			v = v.permute(0,2,1).contiguous()
		return x, v
	#def encode_
	def forward(self, x):
		x = x * torch.inverse((self.A)+self.I)
		return x
	def calculate_gaussian(self, x, v):
		print(self.A)
		if x.dim()>2:
			x = x.permute(0,2,1)
			v = v.permute(0,2,1)
		x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
		v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
		v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
			v = v.permute(0,2,1).contiguous()
		return x, v
	#def encode_
	def forward(self, x):
		x = x * torch.inverse((self.A)+self.I)
		return x
	  

class Encoder(nn.Module):
	def __init__(self, n_topics, n_hidden=256, vocabulary_size=2000,z2_dim=4, y_dim=4):
		super().__init__()
		self.n_topics = n_topics
		self.y_dim = y_dim
		self.channel = 4 
		self.net = nn.Sequential(
			nn.Linear(vocabulary_size+y_dim, n_hidden),
			nn.ELU(),
			nn.Linear(n_hidden, n_hidden),
			nn.ELU(),
			nn.Linear(n_hidden, 2 * n_topics*z2_dim),
			nn.ELU(),
			nn.BatchNorm1d(2 * n_topics*z2_dim, eps=0.001, momentum=0.001, affine=True)
		)

	def conditional_encode(self, x, l):
		x = x.view(-1, self.channel*96*96)
		x = F.elu(self.fc1(x))
		l = l.view(-1, 4)
		x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
		x = F.elu(self.fc3(x))
		x = self.fc4(x)
		m, v = ut.gaussian_parameters(x, dim=1)
		return m,v

	def encode(self, x, y=None):
		xy = x if y is None else torch.cat((x, y), dim=1)
		h = self.net(xy)
		m, v = ut.gaussian_parameters(h, dim=1)
		return m, v
   
   
class Decoder_DAG(nn.Module):
	def __init__(self, n_topics, y_dim=0,channel = 4,  vocabulary_size=2000, topic_embeddings_size=128,flag=0,z2_dim=4):
		super().__init__()
		self.n_topics = n_topics
		self.y_dim = y_dim

		self.vocabulary_size = vocabulary_size
		self.channel = channel
		self.z2_dim = z2_dim
		self.topic_embeddings_size = topic_embeddings_size
		
		self.theta_softmax=nn.Softmax(dim=1)
		self.theta_linear=nn.Linear(z2_dim*n_topics, n_topics)
		self.theta_batch_nor = nn.BatchNorm1d(n_topics, eps=0.001, momentum=0.001, affine=True)
		self.beta_softmax = nn.Softmax(dim=1)
		self.rec_softmax = nn.Softmax(dim=1)
		topic_embeddings_mat = Parameter(torch.Tensor(n_topics, topic_embeddings_size))
		torch.nn.init.xavier_normal_(topic_embeddings_mat.data, gain=1)
		self.register_parameter('topic_embeddings_mat', topic_embeddings_mat)
		word_embeddings_mat = Parameter(torch.Tensor(topic_embeddings_size, vocabulary_size))
		torch.nn.init.xavier_normal_(word_embeddings_mat.data, gain=1)
		self.register_parameter('word_embeddings_mat', word_embeddings_mat)
	def decode_sep(self, z, u, y=None):
		self.theta = self.theta_softmax(self.theta_batch_nor(self.theta_linear(z.view(-1,self.z2_dim*self.n_topics))))
		self.beta = self.beta_softmax(torch.mm(self.topic_embeddings_mat, self.word_embeddings_mat))
		logits = torch.log(torch.mm(self.theta, self.beta)+1e-10)		
		return logits