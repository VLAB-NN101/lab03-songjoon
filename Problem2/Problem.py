from collections.abc import Callable
import numpy as np
from typing import List  
import torch
import csv

"""
Problem2 MNIST Handwriting Softmax

You are given a information about handwritten number from 0 to 9.
Determine what number a image indicates.

Here's structure of information!

[Label]
10 one-hot vectors
*if answer is one, then [0,1,0,0,0,0,...,0]

[Input]
784 vectors(28 * 28 image)

[Tensor structure]
x_train = Tensor(n, 784)
label = Tensor(n, 10)
"""
def train(x_train, label):
    # Write your implementation here.
   pass


def onehot(n):
   a = torch.zeros(10)
   a[int(n)]= 1
   return a
if __name__ == '__main__':
   with open("./ex1.csv", "r") as f:
      #Data processing
      rdr = csv.reader(f)
      arr = [line for line in rdr][1:]
      arr = list(map(lambda x: list(map(float, x)), arr))
      arr = torch.Tensor(arr)
      label = arr[:, 0].T
      data = arr[:, 1:]
      label = (torch.concatenate(list(map(onehot, label))).reshape(len(label), 10))
      x_test = data[:3]
      x_train = data

      #Test
      w,b = train(x_train, label) 
      print("weight : " + str(w) + ", bias : " + str(b))
      
      y = torch.softmax(torch.matmul(x_test,w) + b)
      print("Predicted value : " + str(y))
      
      answer = label[:3]
      print("Actual value : " + str(answer))