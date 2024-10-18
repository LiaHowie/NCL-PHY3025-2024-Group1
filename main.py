# Main code for data analysis -- Lia
import pandas as pd
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy import coordinates as coord
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9

L_sol = 3.846* 1e26

def lin_lstsq(x,y):
    # Function for a linear least square fit
    # maths for a and b provided by Jason
    N = len(x)
    a = (np.sum(x*y)-(1/N)*np.sum(x)*np.sum(y))/(np.sum(x**2)-(1/N)*(np.sum(x)**2))
    b = np.mean(y) - a*np.mean(x)
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
    return a


def lum(f,d):
    # Luminosity formula
    return f*4*np.pi*(d**2)

def SFR(L):
    # Star formation rate formula (Kennicutt 1998)
    return (L)/(5.8*10**(9) *(L_sol))

# Open compiled dataset created in dataset_processing.py as a pandas dataframe
df = pd.read_csv("Compiled_AGN_dataset.csv",sep=",",index_col=0)
# Remove any rows that have NaN in columns of interest
df.dropna(subset=['XRAY:REDSHIFT'],inplace=True)
df.dropna(subset=['IR:FNU_12'],inplace=True)
df.dropna(subset=['XRAY:LUM'],inplace=True)

# Calculate distance to galaxies using redshift
z = df['XRAY:REDSHIFT'].to_numpy() * cu.redshift
d = z.to(u.meter, cu.redshift_distance(WMAP9, kind="comoving"))

# Using the distance, now find the IR luminosity (converting from Jy to W m^-2 Hz^-1 first)
lum_fir = lum(df['IR:FNU_12']*1e-26,d)
# Converting luminosity to units of L_sol
lum_fir = lum_fir / L_sol

# Calaculate Star Formation Rates in units of M_sol per year
sfr = SFR(lum_fir)

# Converting luminosity to units of L_sol
lum_xray = (10**df['XRAY:LUM'])/ L_sol

""" # Remove outliers
outlier = list(lum_xray).index(np.max(lum_xray))
del sfr[outlier]
del lum_xray[outlier] """

# Run linear least square algorithm on the base 10 log of the SFR and luminosity
fit = lin_lstsq(np.log10(lum_xray),np.log10(sfr))
# Create a best fit line using the linear least square output
fitx = np.geomspace(np.min(np.log10(lum_xray)),np.max(np.log10(lum_xray)),10000)
fity = fit[0]*fitx + fit[1]



#%%
# Plot graph:
mpl.rcParams['lines.markersize'] = 5
mpl.style.use(['science','grid'])
mpl.rcParams['figure.figsize'] = (8,7)
mpl.rcParams['font.size'] = 14

plt.figure()

#ax.grid(color='lightgrey', linestyle='--', linewidth=0.5,alpha=0.6)

plt.scatter(np.log10(lum_xray), np.log10(sfr), color="slategrey", marker='x', label="Data")

plt.plot(fitx,fity, color='firebrick', linewidth=1, label="Fit Line")

plt.xlabel("$\log($AGN Luminosity, $L_{\odot})$")
plt.ylabel("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")
#plt.xlim(np.min(lum_xray),1.5e11)
#plt.ylim(0,2e15)
plt.legend()

plt.show()