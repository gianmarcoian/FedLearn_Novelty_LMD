import ml_collections
import torch

def get_default_configs():
  config = ml_collections.ConfigDict()
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 4
  training.n_iters = 15000
  training.snapshot_freq = 10000
  training.log_freq = 50
  training.eval_freq = 60
  training.snapshot_freq_for_preemption = 5000
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.sde = 'subvpsde'
  training.continuous = True
  training.reduce_mean = True

  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.recon_ckpt = 25
  evaluate.n_samples = 50

  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST_IN'
  data.image_size = 32
  data.random_flip = False
  data.centered = True
  data.uniform_dequantization = False
  data.num_channels = 3

  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 8
  model.ch_mult = (1, 1, 1, 1)
  model.num_res_blocks = 1
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  
  #config.mask_type= "checkerboard"
  return config

