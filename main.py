import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Open X-Ray dataset
df_xray = pd.read_csv("SWIFT X-Ray Catalogue.csv", delimiter="|", skiprows=1)
""" Columns: 
'BAT_NAME', 'RA', 'DEC', 'SNR', 'COUNTERPART_NAME', 'OTHER_NAME', 'CTPT_RA', 'CTPT_DEC', 'FLUX', 'FLUX_LO',
'FLUX_HI', 'CONTA', 'GAMMA', 'GAMM_LO', 'GAMM_HI', 'CHI_SQ_R', 'REDSHIFT', 'LUM', 'ASSOC_STREN', 'CL2', 'TYPE'
"""
print(df_xray)
print(df_xray.loc[1, 'RA'])
print(df_xray['RA'])

# Open Infrared dataset
df_ir = pd.read_csv("table_irsa_catalog_search_results.csv", delimiter=",", skiprows=0)
""" Columns:
galname', 'rah', 'ram', 'ras', 'decsign', 'decd', 'decm', 'decs', 'glat_orig', 'semimajor', 'semiminor', 
'posang', 'nhcon', 'fnu_12', 'fqual_12', 'fnu_25', 'fqual_25', 'fnu_60', 'fqual_60', 'fnu_100', 'fqual_100', 
'fir', 'fqfir', 'relunc_12', 'relunc_25', 'relunc_60', 'relunc_100', 'cc_12', 'cc_25', 'cc_60', 'cc_100', 
'cirr1', 'cirr2', 'confuse', 'pnearh', 'pnearw', 'hsdflag', 'ses1_12', 'ses1_25', 'ses1_60', 'ses1_100', 
'ses2', 'nsss', 'sssname', 'dissss', 'nongal', 'idngal', 'dsngal', 'nrecs', 'rat_12_25', 'err_12_25', 
'rat_25_60', 'err_25_60', 'rat_60_100', 'err_60_100', 'glon', 'glat', 'elon', 'elat', 'ra', 'dec', 'cra', 
'cdec', 'ra1950', 'dec1950', 'cra1950', 'cdec1950', 'ROW_IDX', 'ROW_NUM'
"""
print(df_ir)
print(df_ir.loc[1, 'ra'])
print(df_ir['ra'])


def lin_lstsq(x,y):
    # Function for a linear least square fit -- Lia
    # maths for a and b provided by Jason
    N = len(x)
    a = (np.sum(x*y)-(1/N)*np.sum(x)*np.sum(y))/(np.sum(x**2)-(1/N)*(np.sum(x)**2))
    b = np.mean(y) - a*np.mean(x)
    return [a,b]












# Test Data:
data = []
for i in range(1,6):
    data.append(i)
data = np.array(data,dtype="float64")
x = data
y = data*2

""" x = np.array([2,3,3,3,4,4,5,5,5,6])
y = np.array([28.7,24.8,26.0,30.5,23.8,24.6,23.8,20.4,21.6,22.1]) """
fit = lin_lstsq(x,y)
print(np.sum(x))
print(np.sum(y))
print(np.sum(x**2))
print(np.sum(y**2))
print(np.sum(x*y))

print(fit)

plt.figure()
plt.scatter(x,y)

fitx = np.linspace(0,np.max(x))
fity = fit[0]*fitx + fit[1]
plt.plot(fitx,fity)

plt.show()