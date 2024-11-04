#Import packages
import pandas as pd #for data frames
from astropy.cosmology import FlatLambdaCDM 
from astropy.cosmology import WMAP9
from astropy import coordinates as coord
from astropy import units as u
import astropy.cosmology.units as cu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import seaborn as sns

cosmo = WMAP9

def AbsMag(m,d):
    # Function to calculate absolute magnitude
    return m - 5*np.log10(d*(1e6)/10)

# Reading the SDSS dataset and our compiled dataset
df = pd.read_csv('SDSS_data.csv')
df_comp = pd.read_csv('Compiled_AGN_dataset.csv')



# The below section of code matches coordinates from the SDSS data and our compiled dataset
# See "dataset_processing.py" for more detailed commenting on how this works
# I don't actually know what equinox the SDSS dataset it using, I just guessed J2000
sdss_coords = coord.SkyCoord(ra=df['RA']*u.deg, dec=df['DEC']*u.deg, unit=u.deg, equinox="J2000")
agn_coords = coord.SkyCoord(ra=df_comp['XRAY:RA']*u.deg, dec=df_comp['XRAY:DEC']*u.deg, unit=u.deg, equinox="J2000")
idx, sep2d, dist3d = coord.match_coordinates_sky(agn_coords, sdss_coords)
df_match = df.iloc[idx,:]
df_match = df_match.reset_index(drop=True)
# Here the tolerance value is different to that in our dataset compiling.
# There were very few matches so we made the tolerance less limiting
tolerance = coord.Angle("1d1m00s", unit=u.deg)
filt_idx = []
for i in range(len(sep2d)):
    if sep2d[i] <= tolerance:
        filt_idx.append(i)
df_filt = df_match.iloc[filt_idx,:]
df_filt = df_filt.reset_index(drop=True)
df_comp_filt = df_comp.iloc[filt_idx,:]
df_comp_filt = df_comp_filt.reset_index(drop=True)
df_final = pd.concat([df_filt,df_comp_filt],axis=1)





# Grabbing colours green and red from SDSS data for plotting a colour-magnitude plot
galaxies_g = df["g"]
galaxies_r = df["r"]
# Calculating the luminosity distance for use in absolute magnitude calculations
galaxies_d = cosmo.luminosity_distance(df['redshift'])
# Calculating the colour, g-r, of the SDSS galaxies
g_r = np.array(galaxies_g - galaxies_r)
# calculating the absolute magnitude of the SDSS galaxies
galaxies_mg = np.array(AbsMag(galaxies_g,galaxies_d.value))

# This next esction of code separates the SDSS dataset into three sections based on two lines on the plot
# Section 1 (below the bottom line) is the Blue Cloud of star forming galaxies
# Section 2 (in between the two lines) is the Green Valley
# Section 3 (above the top line) is the Red Sequence of post-star forming galaxies
blue_cloud_g_r = []
blue_cloud_g = []
blue_cloud_d = []
red_sequence_g_r = []
red_sequence_g = []
red_sequence_d = []
green_valley_g_r = []
green_valley_g = []
green_valley_d = []
for i in range(len(g_r)):
    if g_r[i] < -0.0571*(np.array(galaxies_mg[i])+24)+0.9 - 0.05:
        blue_cloud_g_r.append(g_r[i])
        blue_cloud_g.append(galaxies_g[i])
        blue_cloud_d.append(galaxies_d[i].value)
    elif g_r[i] > -0.0571*(np.array(galaxies_mg[i])+24)+0.9 + 0.05:
        red_sequence_g_r.append(g_r[i])
        red_sequence_g.append(galaxies_g[i])
        red_sequence_d.append(galaxies_d[i].value)
    else:
        green_valley_g_r.append(g_r[i])
        green_valley_g.append(galaxies_g[i])
        green_valley_d.append(galaxies_d[i].value)

# Calcuate the absolute magnitudes of each of these three sections individually
blue_Mg = AbsMag(np.array(blue_cloud_g),np.array(blue_cloud_d))
red_Mg = AbsMag(np.array(red_sequence_g),np.array(red_sequence_d))
green_Mg = AbsMag(np.array(green_valley_g),np.array(green_valley_d))

# Calculate the colour and absolute magnitude of any matched AGN from our compiled dataset
agn_d = cosmo.luminosity_distance(df_final['redshift'])
agn_g_r = df_final["g"] - df_final["r"]
agn_mg = np.array(AbsMag(df_final["g"],agn_d.value))


# Setup Matplotlib Settings
mpl.rcParams['lines.markersize'] = 5
mpl.style.use(['science','grid'])
mpl.rcParams['figure.figsize'] = (8,7)
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.markersize'] = 1.2
props = dict(boxstyle='round', facecolor='white', edgecolor='darkgrey', alpha=0.5)


#set up figure
fig, ax = plt.subplots() 
ax.grid(color='lightgrey', linestyle='--', linewidth=0.5,alpha=0.6)

# Plot the three SDSS dataset sections
ax.scatter(blue_Mg, blue_cloud_g_r, color="slateblue")
ax.scatter(red_Mg, red_sequence_g_r, color="indianred")
ax.scatter(green_Mg, green_valley_g_r, color="green")
# Label Axes
plt.xlabel('$M_g$')
plt.ylabel('Colour (g-r)')

# Overlay a contour polot and 2D histogram so the density of data, and as such the Green Valley, is more visible
gist_gray = mpl.colormaps['gist_gray'].resampled(256)
colour_2dhist = gist_gray(np.linspace(0.3,1,256))
cmap_2dhist = mpl.colors.LinearSegmentedColormap.from_list("",colour_2dhist)
ax.hist2d(galaxies_mg, g_r, bins=100, cmap = cmap_2dhist, cmin = 2.5, alpha = 0.5)
sns.kdeplot(data=df,x=galaxies_mg,y=g_r,color= (0.7, 0.7, 0.7, 0.5),linewidths=0.8, levels=20)

# Calculating the SFR and L_AGN of the matched galaxies from our compiled dataset
# See "main.py" for more detailed commenting on how this works
z = df_final['XRAY:REDSHIFT'].to_numpy() * cu.redshift
d = z.to(u.centimeter, cu.redshift_distance(WMAP9, kind="comoving"))
f100 = df_final['IR:FNU_100']*(1.00*10**(-14))
f60 = df_final['IR:FNU_60']*(2.58*10**(-14))
FIR = 2.16*(f100 + f60)
FIR = FIR*1e7 * 1e-4
lum_fir = np.array(FIR*4*np.pi*(d**2))
sfr = np.array((4.5*1e-44)*lum_fir)
lum_xray = np.array(df_final['XRAY:FLUX']*4*np.pi*(d**2))
L_sol = 3.846* 1e26 *1e7
lum_xray = lum_xray/L_sol

# Plot all the matched galaxies with labels displaying the SFR and L_AGN values
fontsize = 11
x_shift = -0.15
for i in range(len(agn_g_r)):
    if i == 0:
        ax.plot(agn_mg[i],agn_g_r[i],"v",markersize=7,color="Gold",label="Galaxies of Interest",markeredgecolor='black')
    else:
        ax.plot(agn_mg[i],agn_g_r[i],"v",markersize=7,color="Gold",markeredgecolor='black')
    plt.text(agn_mg[i]-x_shift,agn_g_r[i],
             '''SFR = '''+str(np.format_float_positional(sfr[i],precision=4,unique=False,fractional=False, trim='k'))+''' $M_{\odot}\cdot$yr$^{-1}$
             $L_{AGN}$ = '''+str(np.format_float_positional((lum_xray[i])/(1e22),precision=4,unique=False,fractional=False, trim='k'))+'''$\\times 10^{22}$ $L_{\odot}$''',
             color='black', bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkgrey', alpha=0.8), fontsize=fontsize, ha='right', va='bottom')

# Plot text labelling the Blue Cloud, Green Valley, and Red Sequence sections of the plot
plt.text(-13.4, 0.14, 'Blue Cloud', color='royalblue', rotation=15, rotation_mode='anchor',bbox=props)
plt.text(-13.4, 0.275, 'Green Valley', color='green', rotation=15, rotation_mode='anchor',bbox=props)
plt.text(-13.4, 0.41, 'Red Sequence', color='firebrick', rotation=15, rotation_mode='anchor',bbox=props)

# Plot the two lines of separation used to estimate the location of the Green Valley
x = np.linspace(-30,-10,1000)
y = -0.0571*(x+24)+0.9 + 0.05
ax.plot(x,y,"--", color="Black")
y = -0.0571*(x+24)+0.9 - 0.05
ax.plot(x,y,"--", color="Black",label="Lines of Separation")

# Limit the axes for better looking plot
plt.xlim(-13,-23)
plt.ylim(-0.18,1.68)
# Plot Legend
plt.legend(loc="upper left")

# Show figures - this stalls the code so has to be ran last
plt.show()