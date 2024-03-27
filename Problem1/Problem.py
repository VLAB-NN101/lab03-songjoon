from collections.abc import Callable
import numpy as np
from typing import List  
import csv
import torch
"""
Problem 1 (Titanic Logistic)
You are given a information about passengers in Titanic when the accident occurs.
Determine which person would survive or not.
123
Here's structure of information!asdasdasdasd

[Labelasd
Survival : Does a man survive? => It would be your output.

[Input]
pclass : Classes the passenger is in. 1 -> first class, 2 -> second class, 3-> third class
Age : Age of person.
Sex : gender of person.
sibsp : # of siblings/spouses abroad *타이타닉에 탑승한 형제 명수
parch : # of parents / children abroad. *타이타닉에 탑승한 부모, 아이들의 수
fare : the money he or she is paid for ship.
C : whether he or she is from Cherbourg.
Q : whether he or she is from Queenstown 
S : whether he or she is from Southampton
*C,Q,S is given by 0 or 1

Tensor structure
x_train = Tensor(n, 9)
label = Tensor(n, 1)
"""
def train(x_train, label):
    # Write your implementation here.
   pass

if __name__ == '__main__':
   with open("./ex1.csv", "r") as f:
      #Data Processing
      rdr = csv.reader(f)
      arr = [line for line in rdr][1:]
      arr = list(map(lambda x: list(map(float, x)), arr))
      arr = torch.Tensor(arr)
      label = arr[:, 1]
      x_train = arr[:, 2:]
      x_test = torch.Tensor([[3.,-0.5,26.0,1.,0.,14.4542,0.,1.,0.], [3.,0.5,21.0,0.,0.,7.65,1.,0.,0.],[3.,-0.5,38.0,0.,0.,7.8958,1.,0.,0.]])
      
      #Test
      w,b = train(x_train, label) 
      print("weight : " + str(w) + ", bias : " + str(b))
      
      y = torch.matmul(x_test,w) + b
      print("Predicted value : " + str(y))
      
      answer = torch.tensor([0., 1., 0.])
      print("Actual value : " + str(answer))
