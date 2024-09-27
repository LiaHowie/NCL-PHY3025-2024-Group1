import pandas as pd


df_xray = pd.read_csv("SWIFT X-Ray Catalogue.csv", delimiter="|", skiprows=1) # Open X-Ray dataset
""" Columns: 
'BAT_NAME', 'RA', 'DEC', 'SNR', 'COUNTERPART_NAME', 'OTHER_NAME', 'CTPT_RA', 'CTPT_DEC', 'FLUX', 'FLUX_LO',
'FLUX_HI', 'CONTA', 'GAMMA', 'GAMM_LO', 'GAMM_HI', 'CHI_SQ_R', 'REDSHIFT', 'LUM', 'ASSOC_STREN', 'CL2', 'TYPE'
"""
print(df_xray)
print(list(df_xray))
print(df_xray.loc[1, 'RA'])

df_ir = pd.read_csv("table_irsa_catalog_search_results.csv", delimiter=",", skiprows=0) # Open Infrared dataset
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
print(list(df_ir))
print(df_ir.loc[1, 'ra'])

# Hi