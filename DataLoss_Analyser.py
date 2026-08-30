#Import required modules for constructing and training the model.
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json


import ast

# Device and system imports 
import sys
import random
import argparse
import os
from pathlib import Path
import re
import numpy as np

#imports required for data logging
import json
import wandb
from datetime import datetime 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Required argument to load in JSON file of results
    parser.add_argument('-d', 
                        '--data-path', 
                        dest='data_path', 
                        required = True, 
                        type = str, 
                        nargs = "+",
                        help='Input')
    args = parser.parse_args()
   
    
    



def process_training_log(dataset_paths):

    all_training_err = []


    for path in dataset_paths:
        

        with open(path, "r") as file:
            data = json.load(file)

        all_training_err.append([epoch['train_err'] for epoch in data])
            

        
    #set up line chart to display data
    x_axis = np.arange(0, len(all_training_err[0]))
        
    plt.plot(x_axis, all_training_err[0], color="red", label="fc-legendre")
    plt.plot(x_axis, all_training_err[1], color = "green", label ="zero-padding")
    plt.ylabel("training error")
    plt.xlabel("epoch")

    plt.legend()
    plt.show()
        


if len(args.data_path) == 2:
    process_training_log(args.data_path)

