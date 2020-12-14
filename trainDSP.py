# Train DSP
"""
This script runs a simple example of training a neural network.
"""

# +
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
from datetime import date

today = date.today()
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
    today = date.today()
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
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=False,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=CPU_COUNT
    )

    # define convolutional network
    model = Net()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #train(loss, model, optimizer, trainloader)
     # set up pytorch loss /  optimizer
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )
    
        
#def train(loss, model, optimizer, trainloader):
    # train the network
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # unpack the data
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                loss = running_loss / 2000
                run.log('loss', loss)  # log loss metric to AML
                print(f'epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}')
                running_loss = 0.0
    
    #add parameters for tag
    accuracy = 0.8 
    
    #Save model storage
    os.makedirs('./outputs', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./outputs', 'DSP_demo.pt'))
   
    #save model to workspace
    run.upload_file('DSP_demo.pt', './outputs/DSP_demo.pt')
    #print(run.get_file_names())
    model = run.register_model(model_name='pytorch_model_1', model_path='DSP_demo.pt',
                               tags={'area': "experiment", 'type': "torch",'Accuracy' : accuracy})
    # model_framework_version=torch.__version__,
    # description='torch model for DSP demo.',

    print("Finished Training")
   


