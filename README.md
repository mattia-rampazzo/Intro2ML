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
 - *competition_data*: competition dataset
 
Currently supported models:
 - convnextv2_base.fcmae_ft_in22k_in1k
 - vit_base_patch16_clip_224.openai_ft_in12k_in1k
 - resnet50.a1_in1k
 - Any model from https://huggingface.co/timm

## Fully-Trained models
Download dataset (imagefolder) from: 

- Aircraft = ```git clone https://github.com/cyizhuo/FGVC-Aircraft-dataset /content/data/FGVC-Aircraft-dataset```
- Cars = ```git clone https://github.com/cyizhuo/Stanford-Cars-dataset /content/data/Stanford-Cars-dataset```
- CUB = ```git clone https://github.com/cyizhuo/CUB-200-2011-dataset /content/data/CUB-200-2011-dataset```

Then run with (substitute with name of dataset to run):
```
python fulltrain.py -d 'FGVC-Aircraft-dataset' -w 2 -b 32
```

## Competition

Download dataset folder, unzip it and place it in data folder. Then update config.yaml dataset_name.
```
mkdir data
curl 'https://drive.usercontent.google.com/download?id=1uTFLGixs4IFPQW5W5-O6nBmtBfSwHEal&export=download&authuser=0&confirm=t&uuid=e2b0a951-7db1-4d69-ad93-a52197c018a9&at=APZUnTVn6Pswx7FPTqczVzfut8eC:1717061137103' \
 -H 'authority: drive.usercontent.google.com' \
 -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
 -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' \
 -H 'cookie: SOCS=CAISHAgCEhJnd3NfMjAyNDA1MTYtMF9SQzEaAml0IAEaBgiAyK-yBg; SID=g.a000kAha7q-pQf3tdUiLnfmKlx7PNsxY3VNQPAscM35UbP_xudcuKOZTd4uRJlz0Gp57kernYQACgYKAbQSAQASFQHGX2MiElODhTlQixUjsi1DaWbL5BoVAUF8yKpLNg-8IqxNgDfCK1wpVUVp0076; __Secure-1PSID=g.a000kAha7q-pQf3tdUiLnfmKlx7PNsxY3VNQPAscM35UbP_xudcuNymNAqIFqS5uFNrXybY1QAACgYKAY4SAQASFQHGX2Mij4ACSUgjKR53pWUP6IUschoVAUF8yKpgMiZoszexyvx6l6J-pAu50076; __Secure-3PSID=g.a000kAha7q-pQf3tdUiLnfmKlx7PNsxY3VNQPAscM35UbP_xudcuykGXhxGgLix0kR0DxyoSdwACgYKAYcSAQASFQHGX2MiHCY-LsQd6ZQVRI3dwHlk3xoVAUF8yKpD0DxhOPNNMEwLf6TP6DRK0076; HSID=AdOqoAt5hD018Y15c; SSID=ADE-2hTBeYMFeqbcS; APISID=0uf7oqDtB5U5Byeq/Aq-IhAhAVT70aDqCR; SAPISID=WSjjdXyDgp4d7VrB/ABC81p7JOySvAYrI0; __Secure-1PAPISID=WSjjdXyDgp4d7VrB/ABC81p7JOySvAYrI0; __Secure-3PAPISID=WSjjdXyDgp4d7VrB/ABC81p7JOySvAYrI0; AEC=AQTF6Hx792OE8HaRjcQmdPpqSKso8zpgjT7QknMnufiWqWZ84nIYUNaxgA; 1P_JAR=2024-5-30-8; NID=514=ip1YECGgLJfYF6L-0US-iA39PB5tPMKJRX7WLlJqyjhpfOJWY7S-Hd85cZF8GaXT8bAJH5gh9FT8BjCtL9Gr7WuMzipH4fIJZz72oHRjMyV-CrdPUMepShGqxHu7IP6A3y49m0t7hV0o4-n_1O7ukofCGndimLM8CBY5tOGshYC9D1f3Y4ci0fmXDx0GF1KXy0Yvr6vGSkuxnhSCw2aKCwwrfaY163xq0F7XLcqcqeknOjBZsbdG9BPH2bd9scVnhiIToI3HJH6lXUbneWZCQ_uXnetP88sVHcqjPbtux80LKfrhM6AYCRT0HJbOTmN5xD9CvQ; __Secure-1PSIDTS=sidts-CjEBLwcBXC8wfO-jfsmcJ53wud7jVxkUKcz_A4WO1fAsS957tqF12I7J-1apb6Rp52eIEAA; __Secure-3PSIDTS=sidts-CjEBLwcBXC8wfO-jfsmcJ53wud7jVxkUKcz_A4WO1fAsS957tqF12I7J-1apb6Rp52eIEAA; SIDCC=AKEyXzUdLcGGrkMUDWlRjEFjhWg6X2bYTezMTs1H9QcJkWOiwzbG7Bws4HV_NJ0Tdpsvg-2hlm0; __Secure-1PSIDCC=AKEyXzWjYsSBIfO7hLzZMHd0-y3NmjdsD2t-kUZyXCOuAK84vu5kMHLmf1FfXz6TU1Bv2np_USU; __Secure-3PSIDCC=AKEyXzVFxxyVKsQSqhT0AjIbR9h7lETLQrnSuVFX0YdvryoLEFl_pW_VQhnvHnm9xraTd6ClvqLW' \
 -H 'sec-ch-ua: "Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"' \
 -H 'sec-ch-ua-mobile: ?0' \
 -H 'sec-ch-ua-platform: "Linux"' \
 -H 'sec-fetch-dest: iframe' \
 -H 'sec-fetch-mode: navigate' \
 -H 'sec-fetch-site: same-site' \
 -H 'upgrade-insecure-requests: 1' \
 -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36' \
 -H 'x-client-data: CI22yQEIpbbJAQipncoBCOzyygEIlaHLAQjcmM0BCIegzQEIucrNAQ==' \
 --compressed --output data/competition_data.tar.gz 
tar -xvzf data/competition_data.tar.gz
```