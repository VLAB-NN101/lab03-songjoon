import Problem 
import torch
import csv
import sys
import numpy as np

PATH = "/home/runner/Testcase/Lab02/Problem1/"

def getAddress(x):
    return PATH + "ex" + str(x) + ".csv"

if __name__ == "__main__":
    ## need to implement grading code
    ## test run example : ./Test.py 1
    num = sys.argv[1]
    with open(getAddress(num), "r") as f:
        rdr = csv.reader(f)
        arr = [line for line in rdr][1:]

        train_data = arr[:-3]
        test_data = arr[-3:]
            
        ## process torch.Tensor and make comparsion
        ## Answer should be < 1% relative error.
            
        train_tensor = torch.Tensor(train_data)
        x_train = train_tensor[:,2:]
        train_label = train_tensor[:,1]
        w, b = Problem.train(x_train,train_label)
            
        test_tensor = torch.Tensor(test_data)
        x_test = test_tensor[:,2:]
        test_label = test_tensor[:,1]
        y = torch.sigmoid(w*x_test+b)
            
        error = 1e-1 * test_label

        assert(all(abs(test_label - y) < error)) 