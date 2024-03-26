import Problem 
import torch
import csv
import sys
import numpy as np
import torch
PATH = "/home/runner/Testcase/Lab03/Problem2/"

def getAddress(x):
    return PATH + "ex" + str(x) + ".csv"

def onehot(n):
   a = torch.zeros(10)
   a[int(n)]= 1
   return a
if __name__ == "__main__":
    ## need to implement grading code
    ## test run example : ./Test.py 1
    
    with open(getAddress(1), "r") as f:
        rdr = csv.reader(f)
        arr = [line for line in rdr][1:]
        arr = list(map(lambda x: list(map(float, x)), arr))
        arr = torch.Tensor(arr)
        data = arr[:, 1:]
        label = arr[:,0].T
        label = (torch.concatenate(list(map(onehot, label))).reshape(len(label), 10))
      
      
        ## process torch.Tensor and make comparsion
        ## Answer should be < 1%    relative error.
        x_train = data[:30000]
        train_label = label[:30000]
        w, b = Problem.train(x_train,train_label)
            
        x_test = data[30000:]
        test_label = label[30000:]
        y = torch.softmax(w*x_test+b)
        error = 1e-1 * test_label
    
        assert(all(abs(test_label - y) < error)) 