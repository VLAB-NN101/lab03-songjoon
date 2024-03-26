from collections.abc import Callable
import numpy as np
from typing import List  
import csv
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
#
# Logistic Problem 1
#
# Determine whether a passenger would survive or not using the given data.
# Caution : You should not use library
#


# Data is given as followings:

# Survival : Does a man survive? => It would be your output.

# Followings will be your input.
# pclass : room class the passenger is in
# Sex : gender of a passenger
# Age : age of a passenger
# sibsp : # of siblings/spouses abroad *타이타닉 밖 형제 명수
# parch : # of parents / children abroad. *타이타닉 밖에 부모, 아이들의 수
# fare : Passenger fare
# S, C, Q implies departure city.
def train(x_train, label):
    # Write your imple  mentation here.
   pass


def dataprocess(path):
   with open(path, "r") as f:
      rdr = csv.reader(f)
      arr = [line for line in rdr][1:]
      #print(list(map(lambda x: list(map(float, x)), arr)))
      ten = torch.Tensor(list(map(lambda x: list(map(float, x)), arr)))
      label = ten[:,1].T.flatten(0)
      
      train = ten[:,2:]
      
      #print(train)
      dataloader = DataLoader(TensorDataset(train, label), batch_size=int(label.size()[0]), shuffle=True)
      for i in dataloader:
         return i
if __name__ == '__main__':
   loader = dataprocess('./ex1.csv')
   x_train, label = loader
    
    # y = x_0 + 2*x_1 + 1 # Note that not all test cases give clear line.
   x_test = torch.Tensor([3,-0.5,26.0,1,0,14.4542,0,1,0], [3,0.5,21.0,0,0,7.65,1,0,0],[3,-0.5,38.0,0,0,7.8958,1,0,0])
   w,b = train(x_train, label) 
   print("weight : " + str(w) + ", bias : " + str(b))
    
   y = torch.matmul(x_test,w) + b
   print("Predicted value : " + str(y))
   
   answer = torch.tensor([0, 1, 0])
   print("Actual value : " + str(answer))