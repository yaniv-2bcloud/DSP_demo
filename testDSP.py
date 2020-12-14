# Test DSP
"""
This script runs a simple example of training a neural network.
"""

from typing import Tuple, List
import os
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from azureml.core import Run
from model import Net
from azureml.core.model import Model

# +

from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.dataset import Dataset


# -

run = Run.get_context()

EPOCHS = 2
BATCH_SIZE = 4
CPU_COUNT = os.cpu_count()
MODEL_PATH = './cifar_net.pth'
PRINT_INTERVAL = 2000

# main

# +
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for SGD'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD'
    )
    parser.add_argument('--output_dir', type=str, help='output directory')

    args = parser.parse_args()

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")
    print("Output dir : " + args.output_dir)

    # prepare DataLoader for CIFAR10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(
        root= args.data_path,
        train=True,
        download=False,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=CPU_COUNT
    )

    # load model from workspace
    print(Model.get_model_path(model_name="pytorch_model_1",_workspace="DSP_ML_DEMO",version=None))
    
      # run model on test
    model = Net()
    #model = torch.load('azureml-models/DSP_demo/7/DSP_demo.pt') 
    model.load_state_dict(torch.load('azureml-models/pytorch_model_1/2/DSP_demo.pt') )
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


    print('Finished Test')

   



# +
#model = torch.load('azureml-models/DSP_demo/7/DSP_demo.pt')  
# -




