import random
import numpy as np
import os
import torch
import torchvision.datasets
from flask import Flask, request, jsonify
from torchvision import transforms
import io
from PIL import Image
import copy
from torch.utils.data import DataLoader, Dataset
import logging



from utils.datasets import CustomDatasetWithLabelsList, CustomDataset
from utils.network import CNN, DDPM
from utils.utils import run_model, run_monte_carlo, train_model, save_model, \
    validate_model, train_model_with_fedprox, model_to_hex, federated_average, \
    save_and_load_hex_model, image_transf, image_transf_cl
from utils.configs import get_default_configs
from utils.recon import  run_recon_pretrained_model, run_recon_current_model
from utils.detect import detect
from utils.train_ddpm import train_ddpm

config = get_default_configs()


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

class FederatedLearningServer(Flask):
    def __init__(self, *args, **kwargs):
        super(FederatedLearningServer, self).__init__(*args, **kwargs)
        self.model = CNN()
        self.model_ddpm = DDPM(config)
        if os.path.exists("current_model.pth"):
            self.model.load_state_dict(torch.load("current_model.pth"))
        '''
        if os.path.exists("checkpoint_10.pth"):
            checkpoint = torch.load("checkpoint_10.pth")
            state_dict = remove_module_prefix(checkpoint['model'])
            #print("Keys in state_dict after removing prefix:", state_dict.keys())  
            try:
                self.model_novelty.load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                logger.error(f'Errore durante il caricamento del modello DDPM: {e}')
                raise
        '''    

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FederatedLearningServer(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    logger.info('Ricevuta richiesta per /infer')
    try:
        image_in_p = request.files['image_in'].read()
        image_out_p = request.files['image_out'].read()

        image_in,image_out= image_transf(image_in_p,image_out_p)
        image_out_class= image_transf_cl(image_out_p)
        #classification
        prediction = run_model(image_out_class, app.model)
        #detection
        #run_recon_pretrained_model(image_in,image_out, config)
        run_recon_current_model(image_in,image_out, config, app.ddpm_state)
        is_ood = bool(detect())
        return jsonify({'prediction': prediction[1], 'is_ood': is_ood})
    
    except Exception as e:
        logger.error(f'error occured during inference: {e}')
        return jsonify({'error': str(e)}), 500



@app.route('/train-worker-on-labels-list', methods=['POST'])
def train_worker_directory_on_labels_list():
    directory_train = request.json['directory_train']
    directory_eval=request.json['directory_eval']
    labels_list = request.json['labels_list']
    transform1 = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1), 
    torchvision.transforms.Resize((32, 32)), 
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  ])
    # Load dataset from directory
    dataset = CustomDatasetWithLabelsList(
        directory_train,
        transform=transform1,
        labels_list=labels_list
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    for images, labels in data_loader:
        print(f"Shape of images batch: {images.shape}")  
        break 
    #trained_model = train_model(app.model, data_loader)
    #save_model(trained_model, "current_model")
    #app.model = trained_model
    #loss, accuracy = validate_model(app.model, labels_list=labels_list)
    ddpm_state= train_ddpm(config,directory_train , directory_eval, labels_list)
    app.ddpm_state=ddpm_state
    app.model_ddpm=ddpm_state['model']
    '''
    return jsonify(
        {
            'loss classification model': loss,
            'accuracy classification model': accuracy
        }
    )
    '''
    return jsonify(
        {
            'loss classification model': 0,
            'accuracy classification model': 100
        }
    )

@app.route('/fed-prox-train-worker-on-labels-list', methods=['POST'])
def fed_prox_train_worker_directory_on_labels_list():
    directory = request.json['directory']
    labels_list = request.json['labels_list']
    global_model = request.json.get('global_model', None)

    if global_model is None:
        global_model = copy.deepcopy(app.model)
        global_model = global_model.state_dict()
    else:
        global_model = torch.load(io.BytesIO(bytes.fromhex(global_model)))

    dataset = CustomDatasetWithLabelsList(
        directory,
        transform= torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1), 
            torchvision.transforms.Resize((32, 32)), 
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.1307,), (0.3081,))  ]),
        labels_list=labels_list
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    temp_model = app.model
    trained_model = train_model_with_fedprox(
        temp_model,
        global_model,
        data_loader
    )
    loss, accuracy = validate_model(temp_model, labels_list=labels_list)
    hex_model = model_to_hex(trained_model)
    return jsonify(
        {
            'model': hex_model,
            'loss': loss,
            'accuracy': accuracy
        }
    )


@app.route('/validate-worker-on-labels-list', methods=['POST'])
def validate_worker_directory_on_labels_list():
    labels_list = request.json['labels_list']
    loss, accuracy = validate_model(app.model, labels_list=labels_list)
    return jsonify(
        {
            'loss': loss,
            'accuracy': accuracy
        }
    )

'''
@app.route('/fed-avg', methods=['POST'])
def federated_average_route():
    data = request.json
    models_state_dict = [save_and_load_hex_model(model_hex, f"model_{i}")
                         for i, model_hex in enumerate(data['models'])]
    weights = data['weights']
    avg_state_dict = federated_average(models_state_dict, weights)
    new_model = CNN()
    new_model.load_state_dict(avg_state_dict)
    save_model(new_model, "current_model")
    app.model = new_model
    hex_model = model_to_hex(new_model)
    return jsonify({'model': hex_model})


'''
@app.route('/set-model', methods=['POST'])
def set_model():
    model = request.json['model']
    app.model.load_state_dict(
        save_and_load_hex_model(model, "current_model")
    )
    return jsonify({'status': 'success'})

@app.route('/get-model', methods=['GET'])
def get_model():
    hex_model = model_to_hex(app.model)
    return jsonify({'model': hex_model})


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    app.run(host='0.0.0.0', port=8000,debug=True)

