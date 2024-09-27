import pandas as pd
df_xray = pd.read_csv("SWIFT X-Ray Catalogue.csv", delimiter="|", skiprows=1)
print(df_xray)
print(list(df_xray))
print(df_xray.loc[1, 'RA'])

df_ir = pd.read_csv("table_irsa_catalog_search_results.csv", delimiter=",", skiprows=0)
print(df_ir)
print(list(df_ir))
print(df_ir.loc[1, 'ra'])