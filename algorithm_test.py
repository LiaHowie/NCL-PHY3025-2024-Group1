# Algorithm test code -- Lia
import numpy as np
import scipy.stats
import scienceplots
import matplotlib.pyplot as plt
import matplotlib as mpl

#%%
# Defining functions

def lin_lstsq(x,y):
    # Function for a linear least square fit
    # maths for a and b provided by Jason
    N = len(x)
    a = (np.sum(x*y)-(1/N)*np.sum(x)*np.sum(y))/(np.sum(x**2)-(1/N)*(np.sum(x)**2))
    b = np.mean(y) - a*np.mean(x)
    # where y = a*x + b
    return [a,b]

def nonlinpoly_lstsq(x,y,m):
    # Function for a non-linear polynomial least square fit
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
    
    # a = np.matmul(A_i,v) # Matrix Multiplication of the Inverse mxm Matrix and the m-dimensional vector
    a = []
    for i in range(0,m+1):
        # Matrix Multiplication of the Inverse mxm Matrix and the m-dimensional vector
        temp = []
        for j in range(0,m+1):
            temp.append(A_i[i][j]*v[j])
        temp_sum = round(np.sum(temp),5)
        a.append(temp_sum)
    # where y = a[m]*x**m + a[m-1]*x**(m-1) + ... + a[1]*x + a[0]
    return a

# Fabricate data for test function
def fab_data(f,lb,ub,var=0.5):
    x = np.linspace(lb,ub, 50)
    data = f(x)
    if var != 0:
        variation = np.random.randint(0,100,size=(50))/(var*100)
    else:
        variation = np.zeros(50)
    data += variation
    return data, x

# Setup Matplotlib Settings
mpl.rcParams['lines.markersize'] = 5
mpl.style.use(['science','grid'])
mpl.rcParams['figure.figsize'] = (8,7)
mpl.rcParams['font.size'] = 14


dmy_len = 1000000
#%%
# Linear Test 1
# y = 2x + 16

def f(x):
    return 2*x + 16

# Without Variation in y
y, x = fab_data(f,-10,20,0)

fit = lin_lstsq(x,y)
fitx = np.linspace(np.min(x),np.max(x),dmy_len)
fity = fit[0]*fitx + fit[1]
print(f"y = 2x + 16, var = 0 ==> f(x) = {fit[0]}x + {fit[1]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = 2x + 16$, without variation")
plt.legend()


# With Variation in y
y, x = fab_data(f,-10,20,0.25)

fit = lin_lstsq(x,y)
fity = fit[0]*fitx + fit[1]
print(f"y = 2x + 16, var = 0.25 ==> f(x) = {fit[0]}x + {fit[1]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = 2x + 16$, with variation")
plt.legend()


#%%
# Linear Test 2
# y = 7x - 29

def f(x):
    return 7*x - 29

# Without Variation in y
y, x = fab_data(f,-10,20,0)

fit = lin_lstsq(x,y)
fitx = np.linspace(np.min(x),np.max(x),dmy_len)
fity = fit[0]*fitx + fit[1]
print(f"y = 7x - 29, var = 0 ==> f(x) = {fit[0]}x + {fit[1]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = 7x - 29$, without variation")
plt.legend()


# With Variation in y
y, x = fab_data(f,-10,20,0.01)

fit = lin_lstsq(x,y)
fity = fit[0]*fitx + fit[1]
print(f"y = 7x - 29, var = 0.01 ==> f(x) = {fit[0]}x + {fit[1]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = 7x - 29$, with variation")
plt.legend()



#%%
# Polynomial Test 1
# y = x^2 -6x + 23

def f(x):
    return x**2 - 6*x + 23

# Without Variation in y
y, x = fab_data(f,-10,20,0)

fit = nonlinpoly_lstsq(x,y,m=2)
fitx = np.linspace(np.min(x),np.max(x),dmy_len)
fity = fit[2]*fitx**2 + fit[1]*fitx + fit[0]
print(f"y = x^2 -6x + 23, var = 0 ==> f(x) = {fit[2]}x^2 + {fit[1]}x + {fit[0]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = x^2 -6x + 23$, without variation")
plt.legend()


# With Variation in y
y, x = fab_data(f,-10,20,0.05)

fit = nonlinpoly_lstsq(x,y,m=2)
fity = fit[2]*fitx**2 + fit[1]*fitx + fit[0]
print(f"y = x^2 -6x + 23, var = 0.05 ==> f(x) = {fit[2]}x^2 + {fit[1]}x + {fit[0]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = x^2 -6x + 23$, with variation")
plt.legend()




#%%
# Polynomial Test 2
# y = x^4 + 2x^3 - 3x^2 - 4x + 4

def f(x):
    return x**4 + 2*x**3 - 3*x**2 - 4*x + 4

# Without Variation in y
y, x = fab_data(f,-3,2.4,0)

fit = nonlinpoly_lstsq(x,y,m=4)
fitx = np.linspace(np.min(x),np.max(x),dmy_len)
fity = fit[4]*fitx**4 + fit[3]*fitx**3 + fit[2]*fitx**2 + fit[1]*fitx + fit[0]
print(f"y = x^4 + 2x^3 - 3x^2 - 4x + 4, var = 0 ==> f(x) = {fit[4]}x^4 + {fit[3]}x^3 + {fit[2]}x^2 + {fit[1]}x + {fit[0]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = x^4 + 2x^3 - 3x^2 - 4x + 4$, without variation")
plt.legend()


# With Variation in y
y, x = fab_data(f,-3,2.4,0.25)

fit = nonlinpoly_lstsq(x,y,m=4)
fity = fit[4]*fitx**4 + fit[3]*fitx**3 + fit[2]*fitx**2 + fit[1]*fitx + fit[0]
print(f"y = x^4 + 2x^3 - 3x^2 - 4x + 4, var = 0.25 ==> f(x) = {fit[4]}x^4 + {fit[3]}x^3 + {fit[2]}x^2 + {fit[1]}x + {fit[0]}")

plt.figure()
plt.scatter(x, y, marker='x', label="Data")
plt.plot(fitx, fity, label="Least Squares Best Fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Test Plot for $y = x^4 + 2x^3 - 3x^2 - 4x + 4$, with variation")
plt.legend()



plt.show()