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
    

    parser.add_argument('-m',
                        '--metric',
                        dest='metric_path',
                        required=True,
                        type=str,
                        choices={'train_err', 'phys_loss', 'mse', 'rmse'})

    args = parser.parse_args()
   
    
    



def process_training_log(dataset_paths, metric):

    valid_metrics = ["train_err", "phys_loss", "mse", "rmse"]
    if metric not in valid_metrics:
        print(f'${metric} is not in valid metrics')
        return
    
    
    key_map = {
        "mse" : "(101, 1.0)_mse",
        "rmse" : "(101, 1.0)_rmse",
        "phys_loss" : "(101, 1.0)_phys_loss"
    }

    metric = key_map[metric] or metric

    print(f"DISPLAYING ${metric.capitalize()}")

    all_errors = []


    for path in dataset_paths:
        
        with open(path, "r") as file:
            data = json.load(file)

        all_errors.append([epoch[metric] for epoch in data])
            

        
    #set up line chart to display data
    x_axis = np.arange(0, len(all_errors[0]))
        
    plt.plot(x_axis, all_errors[0], color="red", label="fc-legendre")
    plt.plot(x_axis, all_errors[1], color = "green", label ="finite-diff-fft")
    plt.ylabel(metric)
    plt.xlabel("epoch")

    plt.legend()
    plt.show()
        


if len(args.data_path) == 2:
    process_training_log(args.data_path, args.metric_path)
else:
    print("unavailble")
