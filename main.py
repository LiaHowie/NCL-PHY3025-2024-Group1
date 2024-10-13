import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def lin_lstsq(x,y):
    # Function for a linear least square fit -- Lia
    # maths for a and b provided by Jason
    N = len(x)
    a = (np.sum(x*y)-(1/N)*np.sum(x)*np.sum(y))/(np.sum(x**2)-(1/N)*(np.sum(x)**2))
    b = np.mean(y) - a*np.mean(x)
    return [a,b]

def nonlinpoly_lstsq(x,y,m):
    # Function for a non-linear polynomial least square fit -- Lia
    # maths provided by Jason
    x = np.array(x)
    y = np.array(y)
    A = np.zeros((m+1,m+1)) # Create an mxm matrix of 0s
    for i in range(0,m+1):   
        # Fill the mxm matrix with the appropriate values
        for j in range(0,m+1):
            A[i][j] = np.sum(x**(m-(m-j)+i))
    A_i = np.linalg.inv(A) # Invert the matrix for use in calculation
    
    v = np.zeros(m+1) # create an m-dimensional vector of 0s
    for i in range(0,m+1):
        # Fill the vector with appropriate values
        v[i] = np.sum(y* x**i)
    
    a = np.matmul(A_i,v) # Matrix Multiplication of the Inverse mxm Matrix and the m-dimensional vector
    """ a = []
    for i in range(0,m+1):
        # Matrix Multiplication of the Inverse mxm Matrix and the m-dimensional vector
        temp = []
        for j in range(0,m+1):
            temp.append(A_i[i][j]*v[j])
        temp_sum = round(np.sum(temp),5)
        a.append(temp_sum) """
    return a

# Open compiled dataset created in dataset_processing.py as a pandas dataframe
df = pd.read_csv("Compiled_AGN_dataset.csv",sep=",",index_col=0)


























""" #%%
# Test Functions

# Non-linear fit
x = np.linspace(1,4,50)
#y = [1,8,27,64,125,216,343,512]
y = 2*x**3 -17*x**2 + 47*x -42
#y = 5*x**2 - 6*x + 1
m = 3

print(nonlinpoly_lstsq(x,y,m))


fig, ax = plt.subplots() #set up figure

ax.grid(color='lightgrey', linestyle='--', linewidth=0.5,alpha=0.6)

ax.scatter(x, y, color="slategrey",marker='x', label="Test Data")

fit = nonlinpoly_lstsq(x,y,m)

#fitx = np.linspace(np.min(x),np.max(x))
fitx = x
fity = fit[0] + fit[1]*fitx + fit[2]*fitx**2 + fit[3]*fitx**3
ax.plot(fitx,fity, color='firebrick', linewidth=1, label="Fit Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Non-linear Polynomial Fit test using $y = 2x^3 - 17x^2 + 47x - 42$")
plt.legend()

# Linear Fit
data = []
for i in range(1,6):
    data.append(i)
data = np.array(data,dtype="float64")
x = data
y = data*2 """

""" x = np.array([2,3,3,3,4,4,5,5,5,6])
y = np.array([28.7,24.8,26.0,30.5,23.8,24.6,23.8,20.4,21.6,22.1]) """
""" fit = lin_lstsq(x,y)


print(fit)

fig, ax = plt.subplots() #set up figure
ax.scatter(x, y, color="slategrey",marker='x', label="Test Data")

fitx = np.linspace(0,np.max(x))
fity = fit[0]*fitx + fit[1]
ax.plot(fitx,fity, color='firebrick', linewidth=1, label="Fit Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Fit test using $y = 2x$")
plt.legend()

plt.show() """