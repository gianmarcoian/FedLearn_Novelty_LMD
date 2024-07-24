import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from .recon_utils import get_model_ema, get_sde
from .inpainting_utils import get_pc_inpainter
import os
import numpy as np
import torch
from  .mask_utils import get_mask
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import abc
from .sampling import get_corrector, get_predictor

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

_PREDICTORS = {}

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt_path", None, "checkpoint path to load")
flags.DEFINE_integer("reps_per_image", 1, 'number of recons to do for each image')

# Data related args
flags.DEFINE_integer("num_images_in", 10, "number of in-domain images to reconstruct")
flags.DEFINE_integer("num_images_out", 10, "number of out-of-domain images to reconstruct")
flags.DEFINE_string("in_domain", None, 'in-domain dataset')
flags.DEFINE_string("out_of_domain", None, 'out of domain dataset')
flags.DEFINE_boolean('id_center_crop', True, 'apply center crop to in domain images')
flags.DEFINE_boolean('ood_center_crop', True, 'apply center crop to out of domain images')

# Mask related args
flags.DEFINE_string("mask_type", 'center', 'mask type; center, random, checkerboard, checkerboard_alt')
flags.DEFINE_boolean("save_mask", False, "save mask or not")
flags.DEFINE_integer("mask_num_blocks", 4, 'number of blocks per edge for checkerboard mask; image_size must be divisible by it')
flags.DEFINE_integer("mask_patch_size", 4, 'patch size, if mask type is random')
flags.DEFINE_float("mask_ratio", 0.5, 'mask ratio, if mask type is random')
flags.DEFINE_string("mask_file_path", None, "file path to load mask from")

flags.DEFINE_string("mask_save_dir", None, "directory to save mask")
flags.DEFINE_integer("mask_identifier", 0, "mask identifier")

flags.mark_flags_as_required(["workdir", "config", "ckpt_path", "in_domain", "out_of_domain"])


'''configs'''
sigma_min = 0.01
sigma_max = 50
num_scales = 1000
beta_min = 0.1
beta_max = 20.

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



class Detector(object):
    def __init__(self, config, model_path): 
   
        self.sde = get_sde(config)
        self.eps = 1e-3   
        self.config=config
        self.model = get_model_ema(config, model_path)

        self.model.eval()
        self.shape = (config.data.num_channels,
                      config.data.image_size, config.data.image_size)

        self.scaler = get_data_scaler(config)
        self.inverse_scaler = get_data_inverse_scaler(config)

        self.predictor = get_predictor("euler_maruyama")

        self.corrector = get_corrector("none")
        logger.debug(f"self.sde= {self.sde}, predictor={self.predictor},corrector= {self.corrector}, inverse_scaler={self.inverse_scaler},snr={self.config.sampling.snr} ")
		
        logger.debug(f"n_steps= {self.config.sampling.n_steps_each}, probability_flow={self.config.sampling.probability_flow},continuous= {self.config.training.continuous}, denoise={self.config.sampling.noise_removal},eps={self.eps} ")

        self.inpainter = get_pc_inpainter(
            sde=self.sde, 
            predictor=self.predictor, 
            corrector=self.corrector, 
            inverse_scaler=self.inverse_scaler, 
            snr=self.config.sampling.snr,
            n_steps=self.config.sampling.n_steps_each, 
            probability_flow=self.config.sampling.probability_flow, 
            continuous=self.config.training.continuous,
            denoise=self.config.sampling.noise_removal, 
            eps=self.eps)
    
    def recon(self, image, mask_info_dict=None, mode="pos"):
        logger.debug(f"Starting recon with mode {mode}")
        mask = get_mask(**mask_info_dict).unsqueeze(0).cuda()
        image_masked = image * mask
        image_inpainted = self.inpainter(self.model, self.scaler(image.cuda()), mask.cuda())
        logger.debug(f"image inpainted yes")
        return image_masked.detach().cpu(), image_inpainted.detach().cpu()

def get_mask_info_dict():
    return {
        "mask_type": "checkerboard_alt", 
        "image_size": 32,
        "num_channels": 3,
        "mask_file_path": None,
        "checkerboard_num_blocks": 4,
        "rand_patch_size": 4,
        "rand_mask_ratio": 0.5,
        "maskgen": None,
        "maskgen_offset": 0
    }



def run_recon_pretained_model(image_in,image_out, config):
    
    detector = Detector(config, "/ddpm_models/732K/checkpoint_10.pth")
    mask_info_dict = get_mask_info_dict()


    with torch.no_grad():
        image_out=  image_out.unsqueeze(0).cuda()
        image_in=  image_in.unsqueeze(0).cuda()

        save_dict_in = {"orig": image_in.detach().cpu(), "masked": [], "recon": []}
        save_dict_out = {"orig": image_out.detach().cpu(), "masked": [], "recon": []}
        
        for j in range(10):

            mask_info_dict['maskgen_offset'] = j
            masked, recon = detector.recon(image=image_in, mask_info_dict=mask_info_dict, mode="pos")

            save_dict_in["masked"].append(masked)
            save_dict_in["recon"].append(recon)

        torch.save(save_dict_in, "image_in.pth")
        del save_dict_in
        
        for j in range(10):
            mask_info_dict['maskgen_offset'] = j
            masked, recon = detector.recon(image=image_out, mask_info_dict=mask_info_dict, mode="neg")
            save_dict_out["masked"].append(masked)
            save_dict_out["recon"].append(recon)

        torch.save(save_dict_out, "image_out.pth")
        del save_dict_out
        
    return 





