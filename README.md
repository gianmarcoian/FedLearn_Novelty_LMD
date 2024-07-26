# Federated learning with Detector and Classifier 
## Intro:
In this project you will find a Federated Learning cluster with 3 nodes:
1 server and 2 clients/workers, operating in this current version on the following ports:
* server    '9000:8000'
* client 1  '8800:8000'
* client 2  '8000:8000'

In this current implementation there are 2 models:
* Classifier (CNN)
* Detector - Diffusion Model (DDPM)

The main.py offers some services, called by the test.py. The current test.py can train both the ddpm and the cnn with a label_list, that contains the MNIST digits (the classes) recognizible by the classifier. The ddpm is trained to understand the dataset in_domain (its manifold), so it can understand if an MNIST digit has surely a digit not seen in training phase.
After the test has trained both the CNN and the DDPM, there is an inference call: Picked a random image from a subfolder (0-9 MNIST digits subfolders), there is a phase of reconstruction of this image (from 5 to 10 replicas of the image), followed by a detection phase. The idea is to inpaint the images with a mask that follows the in_domain manifold, so that the median of the inpainted reconstructions of an OOD (Out Of Domain) image should be further than the threshold established by the median of the distances between the same number of inpainted images from an in-domain image.

## References 
This work follows the LMD paper : @misc{liu2023unsupervisedoutofdistributiondetectiondiffusion,
      title={Unsupervised Out-of-Distribution Detection with Diffusion Inpainting}, 
      author={Zhenzhen Liu and Jin Peng Zhou and Yufan Wang and Kilian Q. Weinberger},
      year={2023},
      eprint={2302.10326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2302.10326}, 
} 

## Requirements

The compose.yaml generates the 3 container running the command ```docker-compose up```
The requirements are written in app/requirements.txt

## Getting Started (how to run it)
Currently if you run the main.py (e.g. running it in your editor) and then you run test.py, it will generates some models using API '/train-worker-on-labels-list', so you can choose the MNIST classes by your convenience. Then it will automatically use the ddpm trained by the label_list chosen and test the inference picking a random image from the in_domain and from the out_domain. There is also the possibility to use pre-trained ddpm models, calling the run_recon_pretrained_model() function from recon.py, instead of recon.run_recon_current_model() (These 2 function are already in the inference service ('/infer'), but the run_recon_pretrained_model() call is currently commented).


## Results expactation
The results are the accuracy of the CNN model and the result of the inference. Since the current version picks a random image from a ood subfolder, the is_ood boolean returned should alwys be true (if you keep the current setting). Moreover the prediction from the CNN is on the same ood_image, that the CNN has not seen during the training, so e.g. if you choose a random image from the subfolder 5 and the Classifier predict it as a 2, you should not be worried (it just does not know about the 5 class existence).

## Possible Future Directions
The future prespectives are to train also the ddpm in the federated learning execution, so averaging the parameters by various ddpm (trained by different labels_lists) and merging them.

## Note

