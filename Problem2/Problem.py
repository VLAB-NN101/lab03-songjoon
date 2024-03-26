from collections.abc import Callable
import numpy as np
from typing import List  
import pandas as pd
import torch
from torchvision import datasets, transforms
#
# Logistic Problem 1
#
# Determine whether a passenger would survive or not using the given data.
# Caution : You should not use library
#


# Data is given as followings:

# Survival : Does a man survive? => It would be your output.

# Followings will be your input.
# passengerId : Id of a passenger
# pclass : Classes the passenger is in
# Age 
# sibsp : # of siblings/spouses abroad *타이타닉 밖 형제 명수
# parch : # of parents / children abroad. *타이타닉 밖에 부모, 아이들의 수
# fare : Passenger fare
#
def train(x_train, label):
    # Write your imple  mentation here.
   pass



if __name__ == '__main__':
   train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('./data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()])),
      batch_size=32
   )
   for batch in train_loader:
      # y = x_0 + 2*x_1 + 1 # Note that not all test cases give clear line.
      x_test = batch[:3, 0]
      x_train = batch[:,0]
      label = batch[:,1]
      w,b = train(x_train, label) 
      print("weight : " + str(w) + ", bias : " + str(b))
      
      y = torch.softmax(torch.matmul(x_test,w) + b)
      print("Predicted value : " + str(y))
      
      answer = batch[:3, 1]
      print("Actual value : " + str(answer))