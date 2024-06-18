# fine-grained-image-recognition-Intro2ML
## Overview
This project focuses on developing a fine-grained image recognition system capable of identifying subtle differences between images of similar objects. The goal is to create a robust model that can classify objects into their specific categories with high accuracy. This can be particularly useful in fields such as biodiversity studies, retail, and security systems.
## Installation
To get started, clone the repository and install the required dependencies:
```
git clone https://github.com/mattia-rampazzo/fine-grained-image-recognition-Intro2ML.git
cd fine-grained-image-recognition
pip install -r requirements.txt
```


## Pre-Trained models
To run using a pre-trained model follow these steps.

### Training
Train the model on the training data :
```
python main.py --config config.yaml --run_name first_run 
```

### Evaluation
Evaluate the trained model on a test dataset:
```
python test.py --config config.yaml --run_name first_run 
```

### Configuration
You can customize the training and evaluation parameters in the *config.yaml* file. This includes the pre-trained backbone model and the dataset.
Currently supported datasets:

 - *aircraft*: FGVC-Aircraft
 - *cars*: Stanford Cars Dataset
 - *cub2011*: CUB-200-2011
 - *dogs*: Stanford Dogs
 - *flowers*: Oxford 102 Flower
 
Currently supported models:

 - Any model from https://huggingface.co/timm
 - e.g: convnextv2_base.fcmae_ft_in22k_in1k

## Fully-Trained models
TODO
