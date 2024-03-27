from collections.abc import Callable
import numpy as np
from typing import List  
import torch
import csv
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


def onehot(n):
   a = torch.zeros(10)
   a[int(n)]= 1
   return a
if __name__ == '__main__':
   with open("./ex1.csv", "r") as f:
      rdr = csv.reader(f)
      arr = [line for line in rdr][1:]
      arr = list(map(lambda x: list(map(float, x)), arr))
      arr = torch.Tensor(arr)
      label = arr[:, 0].T
      data = arr[:, 1:]
      label = (torch.concatenate(list(map(onehot, label))).reshape(len(label), 10))
      #print(label)

      # y = x_0 + 2*x_1 + 1 # Note that not all test cases give clear line.
      x_test = data[:3]
      x_train = data
      w,b = train(x_train, label) 
      print("weight : " + str(w) + ", bias : " + str(b))
      
      y = torch.softmax(torch.matmul(x_test,w) + b)
      print("Predicted value : " + str(y))
      
      answer = label[:3]
      print("Actual value : " + str(answer))