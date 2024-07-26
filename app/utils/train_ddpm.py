import torch
import torch.optim as optim
import numpy as np
from .sde import subVPSDE, VESDE, VPSDE
from .recon_utils import ExponentialMovingAverage, get_optimizer, get_model, create_model
from .datasets import CustomDatasetWithLabelsList,get_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets
from .sampling import get_sampling_fn
import logging
import tqdm



def train_ddpm(config, directory_train, directory_eval, labels_list):
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    initial_step = int(state['step'])

    train_ds, eval_ds, _ = get_dataset(config, labels_list, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)
    sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

    optimize_fn = loss_optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = loss_get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
    eval_step_fn = loss_get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))

    num_train_examples = sum(1 for _ in train_ds)

    batch_size = config.training.batch_size
    steps_per_epoch = num_train_examples // batch_size
    logging.info(f"Number of examples: {num_train_examples}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")

    for step in tqdm.tqdm(range(initial_step, num_train_steps + 1)):
        try:
            batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        except StopIteration:
            print(f"Step {step}: End of dataset reached. Restarting iterator.")
            train_iter = iter(train_ds)
            batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()

        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        if step % config.training.eval_freq == 0:
            try:
                eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
            except StopIteration:
                print(f"Step {step}: End of eval dataset reached. Restarting iterator.")
                eval_iter = iter(eval_ds)
                eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()

            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    sample, n = sampling_fn(model)
    ema.restore(model.parameters())

    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)

    return state

def loss_optimization_manager(config):
  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def loss_get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):

  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)

  def step_fn(state, batch):

    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn

def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):

    score_fn = get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_score_fn(sde, model, train=False, continuous=False):

  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE) or True:
    def score_fn(x, t):
      if continuous or isinstance(sde, subVPSDE) or True:

        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_model_fn(model, train=False):
  def model_fn(x, labels):
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)
  return model_fn

def get_data_scaler(config):
    if config.data.centered:
        return lambda x: x * 2. - 1.     # Rescale to [-1, 1]
    else:
        return lambda x: x

def get_data_inverse_scaler(config):
    if config.data.centered:
        return lambda x: (x + 1.) / 2. #from [-1, 1] to [0, 1]
    else:
        return lambda x: x
