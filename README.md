# Lab03
Logistic, Softmax
# Problem 1 (Titanic Logistic)
You are given a information about passengers in Titanic when the accident occurs.  
Determine which person would survive or not.

Here's structure of information!

[Label]  
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

# Problem 2 (MNIST Softmax)
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