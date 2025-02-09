import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("first_linear_regression_data.txt", delimiter=",")

x_train = data[:, 0] 
y_train = data[:, 1]

#computer error by using least squares 
def compute_error(x,y,w,b):
    sum=0
    m=x_train.shape[0]
    for i in range(m):
        sum+=(((x[i]*w+b)-y[i])**2)
    return sum/(2*m)

# find new bias and weight by gradient descent 
def compute_gradient(x,y,w,b):
    dj_dw=0
    dj_db=0
    m=len(x)
    y_pred=w*x+b
    error=y_pred-y
        
    dj_dw=np.dot(error,x)
    dj_db=np.sum(error)
    return dj_dw/m,dj_db/m

def gradient_descent(x,y,w,b,alpha,iter):

    for i in range(iter):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db

    return w, b


def compute_alpha():
    return 0.0000001
        
    
    
    

w_final, b_final = gradient_descent(x_train ,y_train,0,0,compute_alpha(),100000)


y_pred=w_final*x_train+b_final  


plt.scatter(x_train,y_train,c="r")
plt.title("First regression model")
plt.xlabel("size")
plt.ylabel("price")
plt.plot(x_train,y_pred,c="b")
plt.show()

