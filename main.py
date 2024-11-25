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


def R2(y, yfit):
    # R^2 method for testing how good of a fit our functions are
    r = y - yfit
    a = y - np.mean(y)
    r2 = 1- ((np.sum(r**2))/(np.sum(a**2)))
    return r2


def lum(f,d):
    # Luminosity formula
    return np.array(f*4*np.pi*(d**2))

def SFR(L):
    # Star formation rate formula, requiring luminosity in erg s^-1 (Kennicutt, 1998 [pg 13 (202)])
    return np.array((4.5*1e-44)*L)



#%%
# Processing data so it is analysis ready

# Open compiled dataset created in dataset_processing.py as a pandas dataframe
df = pd.read_csv("Compiled_AGN_dataset.csv",sep=",",index_col=0)
# Remove any rows that have NaN in columns of interest
df.dropna(subset=['XRAY:REDSHIFT'],inplace=True)
df.dropna(subset=['IR:FNU_100'],inplace=True)
df.dropna(subset=['IR:FNU_60'],inplace=True)
df.dropna(subset=['XRAY:FLUX'],inplace=True)

# Remove redshifts below 0.05 to avoid K-corrections
df.drop(df[df['XRAY:REDSHIFT'] > 0.05].index, inplace=True)

print(df)

# Calculate distance (in metres) to galaxies using redshift
z = df['XRAY:REDSHIFT'].to_numpy() * cu.redshift
d = z.to(u.centimeter, cu.redshift_distance(WMAP9, kind="comoving"))

# Converting from specific (nominal) fluxes [Jy] to normal flux [W m^-2] (Helou, 1988 [pg 21 (171)])
f100 = df['IR:FNU_100']*(1.00*10**(-14))
f60 = df['IR:FNU_60']*(2.58*10**(-14))
# Calculate total FIR flux (Helou, 1988 [pg 19 (169)])
FIR = 2.16*(f100 + f60)
# Convert flux from W m^-2 to erg s^-1 cm^-2 (1 W = 10^7 erg s^-1, 1m^2 = 10^4 cm^2)
FIR = FIR*1e7 * 1e-4
# Using the distance, now find the IR luminosity
lum_fir = lum(FIR,d)
# Calaculate Star Formation Rates in units of M_sol per year
sfr = SFR(lum_fir)

# Extract X-Ray flux from SWIFT BAT dataset, units of erg s^-1 cm^-2, and calculate the luminosity in erg s^-1
lum_xray = lum(df['XRAY:FLUX'],d)
# Defining L_sol (luminosity of the sun) in erg s^-1
L_sol = 3.846* 1e26 *1e7
# Putting our calculated X-Ray luminosities in terms of L_sol
lum_xray = lum_xray/L_sol

# Logging SFR, X-Ray luminosity, and redshift as we will frequrntly be using logged values going forward
log_sfr = np.log10(sfr)
log_lum_xray = np.log10(lum_xray)
log_z = np.log10(z)



#%%
# Setup Matplotlib Settings
mpl.rcParams['lines.markersize'] = 5
mpl.style.use(['science','grid'])
mpl.rcParams['figure.figsize'] = (8,7)
mpl.rcParams['font.size'] = 14

# Defining custom colour gradient for redshift
clrs = mpl.colors.LinearSegmentedColormap.from_list("custom", ["orange","red","firebrick"])

# Defining length of dummy arrays to be used when plotting best fit lines, simply ensures they are all the same length index-wise
dmy_len = 1000000


#%%
# Create figure - log(SFR) vs log(lum_xray)
fig, ax = plt.subplots()

# Plot scatter plot of x=log(lum_xray), y=log(sfr) with redshift determining point colour
scatter = ax.scatter(log_lum_xray, log_sfr, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter)
clrbar.set_label("Redshift")

# Run linear least square algorithm on the base 10 log of the SFR and luminosity
fit_sfr_Lagn = lin_lstsq(log_lum_xray,log_sfr)
# Create an array of dummy x-axis values
fitx = np.geomspace(np.min(log_lum_xray),np.max(log_lum_xray),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(lum_xray)
fity_sfr_Lagn = fit_sfr_Lagn[0]*fitx + fit_sfr_Lagn[1]
# Print the best fit equation to console
print(f"log(SFR) = {fit_sfr_Lagn[0]}*log(L_agn) + {fit_sfr_Lagn[1]}")

# Plot the best fit line
ax.plot(fitx,fity_sfr_Lagn, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("$\log($AGN X-Ray Luminosity, $L_{\odot})$")
plt.ylabel("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")
plt.legend()

print(f"R^2 = {R2(y=log_sfr, yfit=fit_sfr_Lagn[0]*log_lum_xray + fit_sfr_Lagn[1])}")


#%%
# Create figure - log(lum_xray) vs log(z)
fig, ax = plt.subplots()

# Plot scatter plot of x=log(z), y=log(lum_xray) with log(SFR) determining point colour
scatter = ax.scatter(log_z, log_lum_xray, marker='x', label="Data", c=log_sfr, cmap="viridis")

# Plot a colourbar to show how the datapoint colour changes depending on log(SFR)
clrbar = plt.colorbar(scatter)
clrbar.set_label("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")

# Run linear least square algorithm on the base 10 log of the luminosity and redshift
fit_Lagn_z = lin_lstsq(log_z, log_lum_xray)
# Create an array of dummy x-axis values
fitx = np.linspace(min(log_z),max(log_z),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(z)
fity_Lagn_z = fit_Lagn_z[0]*fitx + fit_Lagn_z[1]
# Print the best fit equation to console
print(f"log(L_agn) = {fit_Lagn_z[0]}*log(z) + {fit_Lagn_z[1]}")

# Plot the best fit line
ax.plot(fitx,fity_Lagn_z, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("$\log($Redshift$)$")
plt.ylabel("$\log($AGN X-Ray Luminosity, $L_{\odot})$")
plt.legend(loc="lower right")

print(f"R^2 = {R2(y=log_lum_xray, yfit=fit_Lagn_z[0]*log_z + fit_Lagn_z[1])}")


#%%
# Create figure - log(SFR) vs log(z)
fig, ax = plt.subplots()

# Plot scatter plot of x=log(z), y=log(SFR) with log(lum_xray) determining point colour
scatter = ax.scatter(log_z, log_sfr, marker='x', label="Data", c=log_lum_xray, cmap="viridis")

# Plot a colourbar to show how the datapoint colour changes depending on log(lum_xray)
clrbar = plt.colorbar(scatter)
clrbar.set_label("$\log($AGN X-Ray Luminosity, $L_{\odot})$")

# Run linear least square algorithm on the base 10 log of the SFR and redshift
fit_sfr_z = lin_lstsq(log_z, log_sfr)
# Create an array of dummy x-axis values
fitx = np.linspace(min(log_z),max(log_z),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(z)
fity_sfr_z = fit_sfr_z[0]*fitx + fit_sfr_z[1]
# Print the best fit equation to console
print(f"log(SFR) = {fit_sfr_z[0]}*log(z) + {fit_sfr_z[1]}")

# Plot the best fit line
ax.plot(fitx,fity_sfr_z, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("$\log($Redshift$)$")
plt.ylabel("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")
plt.legend(loc="lower right")

print(f"R^2 = {R2(y=log_sfr, yfit=fit_sfr_z[0]*log_z + fit_sfr_z[1])}")


#%%
# Create figure - SFR vs lum_xray
fig, ax = plt.subplots()

# Plot scatter plot of lum_xray, sfr with redshift determining point colour
scatter = ax.scatter(lum_xray, sfr, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter)
clrbar.set_label("Redshift")

# Create an array of dummy x-axis values
fitx = np.linspace(np.min(lum_xray),np.max(lum_xray),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(lum_xray)
fity_sfr_Lagn_lin = (fitx**(fit_sfr_Lagn[0]))*(10**(fit_sfr_Lagn[1]))

# Plot the best fit line
ax.plot(fitx,fity_sfr_Lagn_lin, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("AGN X-Ray Luminosity, $L_{\odot}$")
plt.ylabel("SFR, $M_{\odot}\cdot$yr$^{-1}$")
plt.legend()

# Create figure - SFR vs lum_xray (y-limited)
fig, ax = plt.subplots()

# Plot scatter plot of lum_xray, sfr with redshift determining point colour
scatter = ax.scatter(lum_xray, sfr, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter)
clrbar.set_label("Redshift")

# Create an array of dummy x-axis values
fitx = np.linspace(np.min(lum_xray),np.max(lum_xray),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(lum_xray)
fity_sfr_Lagn_lin = (fitx**(fit_sfr_Lagn[0]))*(10**(fit_sfr_Lagn[1]))

# Plot the best fit line
ax.plot(fitx,fity_sfr_Lagn_lin, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("AGN X-Ray Luminosity, $L_{\odot}$")
plt.ylabel("SFR, $M_{\odot}\cdot$yr$^{-1}$")
plt.ylim(0,36)
plt.legend()



#%%

""" 
NOTE: This section is useless for the actual project but whatever

To fix our 3D stuff do Gradient Descent Method
"""
print("===================================================")

plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)

# Create figure - 3D plot of all three variables
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Plot scatter plot of x=log(lum_xray), y=log(SFR), z=log(z) with redshift determining point colour
scatter = ax.scatter(log_lum_xray, log_sfr, log_z, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter,location='left')
clrbar.set_label("Redshift")

# Dummy arrays for plotting continuous line
fitx = np.linspace(min(log_lum_xray),max(log_lum_xray),dmy_len)
fity = np.linspace(min(log_sfr),max(log_sfr),dmy_len)

# This is the LS line of best fit estimated ustilising http://dx.doi.org/10.18642/ijamml_7100121818
a = fit_Lagn_z[0]
b = fit_Lagn_z[1]
c = fit_sfr_z[0]
d = fit_sfr_z[1]
x0 = np.mean(log_lum_xray)
y0 = np.mean(log_sfr)
z0 = np.mean(log_z)
fitz = ((fitx - x0)/(2*a)) + ((fity - y0)/(2*c)) + z0
# Plot the best LS fit line
ax.plot(fitx, fity, fitz, color='purple', linewidth=1, label="LS Fit Line")
# Print the best fit equation
print(f"log(z) = (log(lum_xray)-{x0})/({2*a}) + (log(SFR)-{y0})/({2*c}) + {z0}")

# Here is our modified version of the above fit:
def sim_solve_3d(a,b,c,d,e,f):
    a1 = (1/(e - a*c))
    a2 = (-a/(e - a*c))
    a3 =(a*d - f)/(e - a*c) 
    return [a1,a2,a3]

fit = sim_solve_3d(a=a, b=b, c=c, d=d, e=fit_sfr_Lagn[0], f=fit_sfr_Lagn[1])
fitz1 = ((fitx - x0)*fit[0]) + ((fity - y0)*fit[1]) + z0 #+ fit[2]
# Plot the modified LS fit
ax.plot(fitx, fity, fitz1, color='blue', linewidth=1, label="Modified LS Fit Line")
# Print the best fit equation
print(f"log(z) = (log(lum_xray)-{x0})*({fit[0]}) + (log(SFR)-{y0})*({fit[1]}) + {z0} (+ {fit[2]} --> we removed this factor)")


"""
# This is code for plotting the computers best estimate for a line of best fit
# This plot is so bad I'm keeping it comented forever
from skspatial.objects import Line, Points

points_array = []
for i in range(len(log_lum_xray)):
    points_array.append([log_lum_xray[i],log_sfr[i],log_z[i]])

points = Points(points_array)

line_fit = Line.best_fit(points)
points_projected = line_fit.project_points(points)
points_projected.plot_2d(ax, color='green', s=50, zorder=3) """

# The next 2 comments worth of code is to draw a line between the two extremes of the scatter plot - just for funzies
# Create a matrix of coordinates of the datapoints in 3d space
pts = []
for i in range(len(log_lum_xray)):
    pts.append([log_lum_xray[i],log_sfr[i],log_z[i]])
pts = np.array(pts)
# Find the indices of the 2 points with the largest distance between them
max_dist = [0,0,0]
max_dist_idx = []
for i in range(len(pts)):
    for j in range(len(pts)):
        if abs(pts[i][0]-pts[j][0]) > abs(max_dist[0]) and abs(pts[i][1]-pts[j][1]) > abs(max_dist[1]) and abs(pts[i][2]-pts[j][2]) > abs(max_dist[2]):
            max_dist = pts[i]-pts[j]
            max_dist_idx = [i, j]
i = max_dist_idx[0]
j = max_dist_idx[1]
# Drawing a line between the two extreme points using these indices
ax.plot([log_lum_xray[i],log_lum_xray[j]],[log_sfr[i],log_sfr[j]],[log_z[i],log_z[j]], linestyle='--', color='green', label='Line between extremes')


# Plotting the centra point of the point cloud
ax.scatter(x0,y0,z0, color="green", label="Point Cloud Origin")

# Label axes and plot legend
ax.set_xlabel("$\log($AGN X-Ray Luminosity, $L_{\odot})$", labelpad=10)
ax.set_ylabel("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$", labelpad=10)
ax.set_zlabel("$\log($Redshift$)$", labelpad=10)
plt.legend()



# Show figures - this stalls the code so has to be ran last
plt.show()






#%%
"""
# Re-plot figures, but dark mode

plt.style.use('dark_background')


#%%
# Create figure - log(SFR) vs log(lum_xray)
fig, ax = plt.subplots()

# Plot scatter plot of x=log(lum_xray), y=log(sfr) with redshift determining point colour
scatter = ax.scatter(log_lum_xray, log_sfr, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter)
clrbar.set_label("Redshift")

# Run linear least square algorithm on the base 10 log of the SFR and luminosity
fit_sfr_Lagn = lin_lstsq(log_lum_xray,log_sfr)
# Create an array of dummy x-axis values
fitx = np.geomspace(np.min(log_lum_xray),np.max(log_lum_xray),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(lum_xray)
fity_sfr_Lagn = fit_sfr_Lagn[0]*fitx + fit_sfr_Lagn[1]
# Print the best fit equation to console
print(f"log(SFR) = {fit_sfr_Lagn[0]}*log(L_agn) + {fit_sfr_Lagn[1]}")

# Plot the best fit line
ax.plot(fitx,fity_sfr_Lagn, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("$\log($AGN X-Ray Luminosity, $L_{\odot})$")
plt.ylabel("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")
plt.legend()

print(f"R^2 = {R2(y=log_sfr, yfit=fit_sfr_Lagn[0]*log_lum_xray + fit_sfr_Lagn[1])}")


#%%
# Create figure - log(lum_xray) vs log(z)
fig, ax = plt.subplots()

# Plot scatter plot of x=log(z), y=log(lum_xray) with log(SFR) determining point colour
scatter = ax.scatter(log_z, log_lum_xray, marker='x', label="Data", c=log_sfr, cmap="viridis")

# Plot a colourbar to show how the datapoint colour changes depending on log(SFR)
clrbar = plt.colorbar(scatter)
clrbar.set_label("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")

# Run linear least square algorithm on the base 10 log of the luminosity and redshift
fit_Lagn_z = lin_lstsq(log_z, log_lum_xray)
# Create an array of dummy x-axis values
fitx = np.linspace(min(log_z),max(log_z),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(z)
fity_Lagn_z = fit_Lagn_z[0]*fitx + fit_Lagn_z[1]
# Print the best fit equation to console
print(f"log(L_agn) = {fit_Lagn_z[0]}*log(z) + {fit_Lagn_z[1]}")

# Plot the best fit line
ax.plot(fitx,fity_Lagn_z, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("$\log($Redshift$)$")
plt.ylabel("$\log($AGN X-Ray Luminosity, $L_{\odot})$")
plt.legend(loc="lower right")

print(f"R^2 = {R2(y=log_lum_xray, yfit=fit_Lagn_z[0]*log_z + fit_Lagn_z[1])}")


#%%
# Create figure - log(SFR) vs log(z)
fig, ax = plt.subplots()

# Plot scatter plot of x=log(z), y=log(SFR) with log(lum_xray) determining point colour
scatter = ax.scatter(log_z, log_sfr, marker='x', label="Data", c=log_lum_xray, cmap="viridis")

# Plot a colourbar to show how the datapoint colour changes depending on log(lum_xray)
clrbar = plt.colorbar(scatter)
clrbar.set_label("$\log($AGN X-Ray Luminosity, $L_{\odot})$")

# Run linear least square algorithm on the base 10 log of the SFR and redshift
fit_sfr_z = lin_lstsq(log_z, log_sfr)
# Create an array of dummy x-axis values
fitx = np.linspace(min(log_z),max(log_z),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(z)
fity_sfr_z = fit_sfr_z[0]*fitx + fit_sfr_z[1]
# Print the best fit equation to console
print(f"log(SFR) = {fit_sfr_z[0]}*log(z) + {fit_sfr_z[1]}")

# Plot the best fit line
ax.plot(fitx,fity_sfr_z, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("$\log($Redshift$)$")
plt.ylabel("$\log($SFR, $M_{\odot}\cdot$yr$^{-1})$")
plt.legend(loc="lower right")

print(f"R^2 = {R2(y=log_sfr, yfit=fit_sfr_z[0]*log_z + fit_sfr_z[1])}")


#%%
# Create figure - SFR vs lum_xray
fig, ax = plt.subplots()

# Plot scatter plot of lum_xray, sfr with redshift determining point colour
scatter = ax.scatter(lum_xray, sfr, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter)
clrbar.set_label("Redshift")

# Create an array of dummy x-axis values
fitx = np.linspace(np.min(lum_xray),np.max(lum_xray),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(lum_xray)
fity_sfr_Lagn_lin = (fitx**(fit_sfr_Lagn[0]))*(10**(fit_sfr_Lagn[1]))

# Plot the best fit line
ax.plot(fitx,fity_sfr_Lagn_lin, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("AGN X-Ray Luminosity, $L_{\odot}$")
plt.ylabel("SFR, $M_{\odot}\cdot$yr$^{-1}$")
plt.legend()

# Create figure - SFR vs lum_xray (y-limited)
fig, ax = plt.subplots()

# Plot scatter plot of lum_xray, sfr with redshift determining point colour
scatter = ax.scatter(lum_xray, sfr, marker='x', label="Data", c=z, cmap=clrs)

# Plot a colourbar to show how the datapoint colour changes depending on redshift
clrbar = plt.colorbar(scatter)
clrbar.set_label("Redshift")

# Create an array of dummy x-axis values
fitx = np.linspace(np.min(lum_xray),np.max(lum_xray),dmy_len)
# Calculate the the best fit line using the linear least square outputted parameters and the continuous value of x=log(lum_xray)
fity_sfr_Lagn_lin = (fitx**(fit_sfr_Lagn[0]))*(10**(fit_sfr_Lagn[1]))

# Plot the best fit line
ax.plot(fitx,fity_sfr_Lagn_lin, color='firebrick', linewidth=1, label="Fit Line")

# Label axes and plot legend
plt.xlabel("AGN X-Ray Luminosity, $L_{\odot}$")
plt.ylabel("SFR, $M_{\odot}\cdot$yr$^{-1}$")
plt.ylim(0,36)
plt.legend()


plt.show()
 """