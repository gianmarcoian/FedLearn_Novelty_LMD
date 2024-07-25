import abc
import torch
import numpy as np
import torch.optim as optim
import logging
import os
from .sde import subVPSDE,VESDE,VPSDE
logger = logging.getLogger(__name__)




def get_optimizer(config, params):
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def get_model_ema(config, model_path):
  score_model = create_model(config)

  optimizer = get_optimizer(config, score_model.parameters())

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  state = restore_model(model_path, state, device=config.device)
  state['ema'].copy_to(score_model.parameters())

  return score_model


def get_model_ema_current_model(config, loaded_state):
    score_model = create_model(config)
    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    loaded_state['ema'].copy_to(score_model.parameters())

    return score_model



def get_model_fn(model, train=False):
  def model_fn(x, labels):

    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def create_model(config):
  model_name = config.model.name

  score_model = get_model(model_name)(config)

  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model

_MODELS = {}

def get_model(name):
  return _MODELS[name]


def register_model(cls=None, *, name=None):
  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

def restore_model(model_dir, state, device):
    loaded_state = torch.load(os.getcwd()+model_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


  

class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):

    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):

    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):

    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):

    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):

    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']

def get_sde(config):
  if config.training.sde.lower() == 'vpsde':
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  return sde
