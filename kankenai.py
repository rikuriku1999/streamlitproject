import random
import math
import matplotlib.pyplot as plt
y_list = []
x_list = []
y_list1 = []
x_list1 = []
y_list2 = []
x_list2 = []
n = 0
max = 100
z = 0
B = random.randrange(40,50)
C = random.randrange(6,8)
def wavegenerate(n, A, s, theta):
    def show():
        plt.cla()
        plt.xlim(0,100)
        plt.ylim(-70,70)
        print(x_list1)
        print(x_list2)
        if n < max :
            plt.plot(x_list,y_list,color = "blue",label = "test")
        else:
            plt.plot(x_list1,y_list1, color = "blue", label = "test")
            plt.plot(x_list2,y_list2, color = "blue", label = "test")
        plt.pause(.05)
    def listing(n,y):
        if n < max :
            x_list.append(n)
            y_list.append(y)
            y_list1 = []
            x_list1 = []
            y_list2 = []
            x_list2 = []
        else:
            m = n%100 -1
            y_list[m] = y
            if 0 <= m < max - 5:
                x_list1 = x_list[0:m]
                y_list1 = y_list[0:m]
                x_list2 = x_list[m+5:max]
                y_list2 = y_list[m+5:max]
            else:
                x_list1 = []
                y_list1 = []
                x_list2 = x_list[5-max+m:m]
                y_list2 = y_list[5-max+m:m]
        return x_list1, x_list2, y_list1,y_list2
    for t in range(s):
        n += 1
        y = A * math.sin(math.radians(theta*t))
        x_list1, x_list2, y_list1,y_list2 = listing(n,y)
        show()
    return n
for i in range(10000000):
    p = random.random()
    if p > 0.4 :
        A = 1
    else :
        A = 0.4
    n = wavegenerate(n,A,6,30)
    n = wavegenerate(n,0,3,30)
    n = wavegenerate(n,A*B,6,50)
    n = wavegenerate(n,0,6,30)
    n = wavegenerate(n,A*C,5,36)
    n = wavegenerate(n,0,3,30)
    n = wavegenerate(n,A*C,6,30)
    n = wavegenerate(n,0,3,30)