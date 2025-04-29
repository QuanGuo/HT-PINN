import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np

class PhysicsInformedNN(nn.Module):

  def __init__(self, layer_dim_u, layer_dim_K, input_K=None, inv_params=None, num_pumps=25):
    super(PhysicsInformedNN, self).__init__()


    self.weights = []

    self.preds = None

    self.loss = 0.0

    self.loss_list = []   
    self.loss_dict = {'neum':[0.0], 'diri':[0.0],'u':[0.0],'f':[0.0],'K':[0.0],'pump':[0.0]}
    self.loss_container = []

    def block(in_feat, out_feat, normalize=False):
      layers = [nn.Linear(in_feat, out_feat)]
      if normalize:
          layers.append(nn.BatchNorm1d(out_feat, 0.8))
      # layers.append(nn.LeakyReLU(0.2, inplace=True))
      layers.append(nn.Tanh())
      return layers

    self.forward_models = []
    for i in range(num_pumps):
      self.modules = []
      for j in range(len(layer_dim_u)-2):
        input_dim, output_dim = layer_dim_u[j], layer_dim_u[j+1],
        self.modules += block(input_dim, output_dim, normalize=False)
      self.modules.append(nn.Linear(output_dim, layer_dim_u[j+2]))
      self.forward_models.append(nn.Sequential(*self.modules))


    self.modules = []
    for j in range(len(layer_dim_K)-2):
      input_dim, output_dim = layer_dim_K[j], layer_dim_K[j+1],
      self.modules += block(input_dim, output_dim, normalize=False)
    self.modules.append(nn.Linear(output_dim, layer_dim_K[j+2]))

    self.inverse_model = nn.Sequential(*self.modules)


    for i in range(num_pumps): 
      loss_dict = {'neum':[], 'diri':[],'u':[],'f':[],'K':[],'pump':[]}
      self.loss_container.append(loss_dict)


  def net_u(self, x, y, t, pid): # head u, including Dirichlet BCs
    H = torch.cat((x,y,t),1)
    u = self.forward_models[pid](H)
    return u
  
  def net_K(self, x, y): # hydraulic conductivity K

    H = torch.cat((x,y),1)
    K = self.inverse_model(H)

    return K
  

  def net_du(self, x, y, t, pid): # first-order derivative match, inlcuding Neumann BCs

    u = self.net_u(x, y, t, pid)

    u_x = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_y = grad(u.sum(), y, create_graph=True, retain_graph=True)[0]
    u_t = grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
    return u_x.requires_grad_(True), u_y.requires_grad_(True), u_t.requires_grad_(True)

  def net_dK(self, x, y): # first-order derivative of K
    K = self.net_K(x, y)#, self.weights_u, self.biases_u)

    K_x = grad(K.sum(), x, create_graph=True)[0]
    K_y = grad(K.sum(), y, create_graph=True)[0]

    return K_x.requires_grad_(True), K_y.requires_grad_(True)


  def net_f(self, x, y, t, pid): # general PDE match, usually formulated in higher-order

    u_x, u_y, u_t = self.net_du(x, y, t, pid)
    u_yy = grad(u_y.sum(), y, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x, create_graph=True)[0]

    K = self.net_K(x, y)
    K_x, K_y = self.net_dK(x, y)

    f = (K*(u_yy + u_xx) + K_x*u_x + K_y*u_y) - u_t*0.0001

    return f.requires_grad_(True)

  def forward(self, x_tensors, y_tensors, t_tensors, pid, keys=None):

    if keys is None:
      keys = x_tensors.keys()
    else:
      preds = dict()
      for i in keys:
          preds[i] = None

    for i in keys:

      if i == 'neum':
        dudx_pred, dudy_pred, dudt = self.net_du(x_tensors[i], y_tensors[i], t_tensors[i], pid)
        preds[i] = dudy_pred

      elif i == 'f':
        f_pred = self.net_f(x_tensors[i], y_tensors[i], t_tensors[i], pid)
        preds[i] = f_pred

      elif i == 'u':
        u_pred = self.net_u(x_tensors[i], y_tensors[i], t_tensors[i], pid) 
        preds[i] = u_pred
          
      elif i == 'K':
        K_pred = self.net_K(x_tensors[i], y_tensors[i])

        preds[i] = K_pred
          
      elif i == 'diri':
        diri_pred = self.net_u(x_tensors[i], y_tensors[i], t_tensors[i], pid) 
        preds[i] = diri_pred

      elif i == 'pump':
        p_pred = self.net_f(x_tensors[i], y_tensors[i], t_tensors[i], pid)
        preds[i] = p_pred

    return preds

  def loss_func(self, pred_dict, true_dict, pump_id, weights=None):
  
    loss = torch.tensor(0.0, dtype=torch.float32)
    keys = pred_dict.keys()

    if weights is None:
      weights = dict()
      for i in keys:
        weights[i] = 1.0

    for i in keys:
      res = pred_dict[i] - true_dict[i]
      loss += weights[i]*torch.mean(res.pow(2))
      r = torch.mean(res.pow(2)).item()
      self.loss_container[pump_id][i].append(r*weights[i])
    return loss.requires_grad_()


  def unzip_train_dict(self, train_dict, keys=None):
    if keys is None:
      keys = train_dict.keys()

    x_tensors = dict()
    y_tensors = dict()
    t_tensors = dict()
    true_dict = dict()

    for i in keys:
      if i == "K":
        x_tensors[i] = train_dict[i][0]
        y_tensors[i] = train_dict[i][1]
        true_dict[i] = train_dict[i][2]
      else:
        x_tensors[i] = train_dict[i][0]
        y_tensors[i] = train_dict[i][1]
        t_tensors[i] = train_dict[i][2]
        true_dict[i] = train_dict[i][3]

    return (x_tensors, y_tensors, t_tensors, true_dict)

  def concat_t_to_train_dict(self, train_dict, t, keys=None):
    if keys is None:
      keys = ["neum", "diri", "u", "pump", "f"]
    for k in keys:
      train_dict[k] = self.concat_t_to_tensor(train_dict[k], t)

    return train_dict

  def concat_t_to_tensor(self, spatial_tensor, t):

    (x_tensors, y_tensors, true_dict) = spatial_tensor

    t_tensors = torch.ones_like(x_tensors, requires_grad=True)
    t_tensors = t_tensors * t

    return (x_tensors, y_tensors, t_tensors, true_dict)


  def train(self, iter, data_batch, loss_func, optimizer, pred_keys=None, loss_weights=None, pump_id_list=[0], print_interval=1000):
      
    if pred_keys is None:
      pred_keys= data_batch[0].keys()
    for i in range(iter):
      optimizer.zero_grad()
      loss = 0.0
      for pump_id in pump_id_list:
        train_dict = data_batch[pump_id]

        (x_tensors, y_tensors, t_tensors, true_dict) = self.unzip_train_dict(train_dict,pred_keys)
        pred_dict = self.forward(x_tensors, y_tensors, t_tensors, pump_id, keys=pred_keys)
        loss += loss_func(pred_dict, true_dict, pump_id, loss_weights)

      loss.backward(retain_graph=True)
      self.callback(loss.detach().numpy().squeeze())

      if np.remainder(len(self.loss_list),print_interval) == 1:
        print('Iter # %d, Loss: %.8f' % (len(self.loss_list), self.loss_list[-1]))
        print_loss = dict()
        for pid in pump_id_list:
          print_loss = "Pump "+ str(pid) + ": "
          for k in ['u','f','K','neum','pump','diri']:
            s = k+":"+str(self.loss_container[pid][k][-1])+"; "
            print_loss += s
          print(print_loss)

      optimizer.step()


  def callback(self, loss):
    self.loss_list.append(loss)

  def coor_shift(self, X, lbs, ubs):
    return 2.0*(X - lbs) / (ubs - lbs) - 1

  def data_loader(self, X, u, lbs, ubs):
              
    X = self.coor_shift(X, lbs, ubs)

    x_tensor = torch.tensor(X[:,0:1], requires_grad=True, dtype=torch.float32)
    y_tensor = torch.tensor(X[:,1:2], requires_grad=True, dtype=torch.float32)

    u_tensor = torch.tensor(u, dtype=torch.float32)
    
    return (x_tensor, y_tensor, u_tensor)

  def predict(self, X_input, t_input, pid=0, target='u'):
    x_tensor = torch.tensor(X_input[:,0:1], dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(X_input[:,1:2], dtype=torch.float32, requires_grad=True)
    t_tensor = torch.ones_like(x_tensor) * t_input
    pred = None
    if target == 'u':
      pred = self.net_u(x_tensor, y_tensor, t_tensor, pid).detach().numpy().squeeze()
    elif target == 'du':
      dudx, dudy = self.net_du(x_tensor, y_tensor, t_tensor, pid)
      return dudx.detach().numpy().squeeze(), dudy.detach().numpy().squeeze()
    elif target == 'f':
      pred = self.net_f(x_tensor, y_tensor, t_tensor, pid).detach().numpy().squeeze()
    elif target == 'K':
      pred = self.net_K(x_tensor, y_tensor).detach().numpy().squeeze()

    return pred
