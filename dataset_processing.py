# Dataset processing -- Lia
# Combining the SWIFT BAT dataset and IRAS dataset by filtering out non-AGN sources in SWIFT BAT,
# then coressmatching RA and DEC values with the IRAS dataset
import pandas as pd
from astropy import coordinates as coord
from astropy import units as u

# Open X-Ray dataset as a dataframe
df_xray = pd.read_csv("SWIFT X-Ray Catalogue.csv", delimiter="|", skiprows=1)
""" Columns: 
'BAT_NAME', 'RA', 'DEC', 'SNR', 'COUNTERPART_NAME', 'OTHER_NAME', 'CTPT_RA', 'CTPT_DEC', 'FLUX', 'FLUX_LO',
'FLUX_HI', 'CONTA', 'GAMMA', 'GAMM_LO', 'GAMM_HI', 'CHI_SQ_R', 'REDSHIFT', 'LUM', 'ASSOC_STREN', 'CL2', 'TYPE'
"""
# Filter out non-AGN sources by searching for strings "AGN" and "Sy"
df_xray_agn = pd.DataFrame(df_xray[df_xray['TYPE'].str.contains("AGN|Sy")]) 
# Reset the indices so it goes from 0 onwards
df_xray_agn = df_xray_agn.reset_index(drop=True)
# Convert X-Ray AGN RA and DEC coordinates to a common system
xray_agn_coords = coord.SkyCoord(ra=df_xray_agn['RA']*u.deg, dec=df_xray_agn['DEC']*u.deg, unit=u.deg, equinox="J2000")

# Open Infrared dataset as a dataframe
df_ir = pd.read_csv("table_irsa_catalog_search_results.csv", delimiter=",", skiprows=0)
""" Columns:
'pscname', 'rah', 'ram', 'ras', 'decsign', 'decd', 'decm', 'decs', 'semimajor', 'semiminor', 'posang', 'nhcon', 'fnu_12',
'fnu_25', 'fnu_60', 'fnu_100', 'fqual_12', 'fqual_25', 'fqual_60', 'fqual_100', 'nlrs', 'lrschar', 'relunc_12', 'relunc_25',
'relunc_60', 'relunc_100', 'tsnr_12', 'tsnr_25', 'tsnr_60', 'tsnr_100', 'cc_12', 'cc_25', 'cc_60', 'cc_100', 'var', 'disc',
'confuse', 'pnearh', 'pnearw', 'ses1_12', 'ses1_25', 'ses1_60', 'ses1_100', 'ses2_12', 'ses2_25', 'ses2_60', 'ses2_100',
'hsdflag', 'cirr1', 'cirr2', 'cirr3', 'nid', 'idtype', 'mhcon', 'fcor_12', 'fcor_25', 'fcor_60', 'fcor_100', 'rat_12_25',
'err_12_25', 'rat_25_60', 'err_25_60', 'rat_60_100', 'err_60_100', 'glon', 'glat', 'elon', 'elat', 'ra', 'dec', 'cra',
'cdec', 'ra1950', 'dec1950', 'cra1950', 'cdec1950'
"""
# Converts Infrared RA and DEC coordinates to a common system
ir_coords = coord.SkyCoord(ra=df_ir['ra']*u.deg, dec=df_ir['dec']*u.deg, unit=u.deg, equinox="B1950")

# finds closest matching infrared coordinates for each AGN in the X-Ray coordinates
idx, sep2d, dist3d = coord.match_coordinates_sky(xray_agn_coords, ir_coords)
# trim the infrared dataframe to only objects obtained in the coordinate matching
df_ir_agn = df_ir.iloc[idx,:]
# reset the indices of the infrared agn dataset so that they start from 0 and count up. This is needed as the previous step jumbles up the indices
df_ir_agn = df_ir_agn.reset_index(drop=True)

# set a tolerance value for angular separation of matching objects, this will filter out any none-matching objects.
# this step is important as the astropy matching above only finds the closest object across the two datasets
tolerance = coord.Angle("0d1m00s", unit=u.deg)

# Filter out any angular separations larger than the tolerance set above
filt_idx = []
for i in range(len(sep2d)):
    if sep2d[i] <= tolerance:
        filt_idx.append(i)

# create new dataframes using the filter array created abov e
df_xray_agn_filt = df_xray_agn.iloc[filt_idx,:]
df_xray_agn_filt = df_xray_agn_filt.reset_index(drop=True)
df_ir_agn_filt = df_ir_agn.iloc[filt_idx,:]
df_ir_agn_filt = df_ir_agn_filt.reset_index(drop=True)

# merge the two dataframes into one and remove unecessary columns
df_final = pd.concat([df_xray_agn_filt,df_ir_agn_filt],axis=1)
df_final = df_final.drop(['#','CTPT_RA','CTPT_DEC','rah','ram','ras','decsign','decd',
                          'decm','decs','semimajor','semiminor','posang','glon','glat',
                          'elon','elat','ra','dec','cra','cdec','ra1950','dec1950',
                          'cra1950','cdec1950'],
                         axis=1)

# Add an 'XRAY:' or 'IR:' prefix to column headings to indicate which dataset they are from
for item in df_final.columns:
    lower = False
    upper = False
    for char in item:
        if char.islower():
            lower = True
        elif char.isupper():
            upper = True
        break
    if lower:
        df_final.rename({item: 'IR:'+item},axis=1,inplace=True)
    elif upper:
        df_final.rename({item: 'XRAY:'+item},axis=1,inplace=True)

# set all column names to be entirely upper case to avoid confusion
df_final.columns = map(str.upper, df_final.columns)

# export combined dataset to .csv for use in data analysis in main.py
df_final.to_csv("Compiled_AGN_dataset.csv",sep=",")

