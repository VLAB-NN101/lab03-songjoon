import Problem 
import torch
import csv
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
PATH = "/home/runner/Testcase/Lab03/Problem2/"

def getAddress(x):
    return PATH + "ex" + str(x) + ".csv"
if __name__ == "__main__":
    ## need to implement grading code
    ## test run example : ./Test.py 1
    train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('./data/', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()])),
      batch_size=32
    )
    for batch in train_loader:
        ## process torch.Tensor and make comparsion
        ## Answer should be < 1%    relative error.
        train_data = batch[:-3]
        test_data = batch[-3:]
        train_tensor = train_data
        x_train = train_tensor[:,0]
        train_label = train_tensor[:,1]
        w, b = Problem.train(x_train,train_label)
            
        test_tensor = test_data
        x_test = test_tensor[:,0]
        test_label = test_tensor[:,1]
        y = torch.softmax(w*x_test+b)
            
        error = 1e-1 * test_label
        
        assert(all(abs(test_label - y) < error)) 