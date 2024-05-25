import torch
import torchvision
import torch.nn as nn
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools
from dogs import Dogs
from cub2011 import Cub2011


dataset_num_classes = {
    'aircraft': 102,
    'cars': 196,
    'cub2011': 200,
    'dogs': 120,
    'food': 101,
    'flowers': 102
    #'inat2017': None, # too big
    #'nabirds': 400, # no longer publicly accessible
    #'tiny_imagenet': 200 # too big + not a good benchmark
}

# Mapping dataset names to their corresponding classes
dataset_map = {
    'aircraft': torchvision.datasets.FGVCAircraft,
    'cub2011': Cub2011,
    'dogs': Dogs,
    'food': torchvision.datasets.Food101,
    'flowers': torchvision.datasets.Flowers102
    #'inat2017': INat2017,
    #'nabirds': NABirds,
    #'tiny_imagenet': TinyImageNet 
}

def get_num_classes(dataset_name):
    num_classes = 1000

    # Load dataset
    if dataset_name in dataset_num_classes:
        num_classes = dataset_num_classes[dataset_name]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return num_classes

def set_to_finetune_mode(model, do_summary=False):

    classifier = model.default_cfg['classifier']
    # print(classifier)

    # Freeze all parameters except those in the classifier
    for name, param in model.named_parameters():
        if name.startswith(classifier):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if do_summary:
        from torchinfo import summary
        summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        ) 

    return model

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.01 / (1.0 + 10 * p) ** 0.75

    return optimizer

def get_optimizer(model):
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    return optimizer

def get_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function

def save_model(weights, save_folder, run_name):
    print("Saving model")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(
        weights,
        os.path.join(save_folder, run_name)
    )

def set_model_mode(mode="train", models=None):
    for model in models:
        if mode == "train":
            model.train()
        else:
            model.eval()

def plot_embedding(X, y, d, training_mode):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        else:
            colors = (1.0, 0.0, 0.0, 1.0)
        plt.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            color=colors,
            fontdict={"weight": "bold", "size": 9},
        )

    plt.xticks([]), plt.yticks([])

    save_folder = "saved_plot"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_name = "saved_plot/" + str(training_mode) + ".png"
    plt.savefig(fig_name)
    print("{} has been successfully saved!".format(fig_name))

def visualize(encoder, training_mode, source_test_loader, target_test_loader):
  
    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= 16:  # to get only 512 samples
            break
        img, label = test_data
        label = label.numpy()
        img = img.cuda()
        img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
        source_label_list.append(label)
        source_img_list.append(img)

    source_img_list = torch.stack(source_img_list)
    source_img_list = source_img_list.view(-1, 3, 28, 28)

    # Get target_test samples
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(target_test_loader):
        if i >= 16:
            break
        img, label = test_data
        label = label.numpy()
        img = img.cuda()
        target_label_list.append(label)
        target_img_list.append(img)

    target_img_list = torch.stack(target_img_list)
    target_img_list = target_img_list.view(-1, 3, 28, 28)

    # Stack source_list + target_list
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    combined_img_list = torch.cat((source_img_list, target_img_list), 0)

    source_domain_list = torch.zeros(512).type(torch.LongTensor)
    target_domain_list = torch.ones(512).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cuda()

    print("Extracting features to draw t-SNE plot...")
    combined_feature = encoder(combined_img_list)  # combined_feature : 1024,2352

    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=3000)
    dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())

    print("Drawing t-SNE plot ...")
    plot_embedding(dann_tsne, combined_label_list, combined_domain_list, training_mode)