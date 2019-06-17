# Databricks notebook source
import pandas as pd
import numpy as np
real_estate_sales=pd.read_csv('/dbfs/FileStore/tables/Real_Estate_Sales_2001_2016-9945c.csv', header=0)

# COMMAND ----------

real_estate_1=real_estate_sales\
  .drop(columns=['NonUseCode', 'Remarks'])\
  .dropna(subset=['PropertyType', 'AssessedValue', 'SaleAmount', 'DateRecorded', 'Address', 'SalesRatio'])
real_estate_1.describe().round(1)

# COMMAND ----------

real_estate_1['ResidentialType'].fillna('Land', inplace=True)
real_estate_1=real_estate_1[real_estate_1.PropertyType != '10 Mill Forest']

# COMMAND ----------

real_estate_1.isnull().sum().sum()

# COMMAND ----------

def display_pdf(a_pdf):
  display(spark.createDataFrame(a_pdf))

# COMMAND ----------

display_pdf(real_estate_1.round(2).sort_values(by='SalesRatio', ascending=False))

# COMMAND ----------

real_estate_1=real_estate_1.assign(SalesRatio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='SalesRatio', ascending=False)

# COMMAND ----------

df_new=real_estate_1[(real_estate_1.SalesRatio<=5) & (real_estate_1.SalesRatio>0)] 
df_new=df_new[((df_new.SaleAmount - df_new.SaleAmount.mean()) / df_new.SaleAmount.std()).abs() < 3]
df_new.info()

# COMMAND ----------

display_pdf(df_new.round(2).sort_values(by='SalesRatio', ascending=False))

# COMMAND ----------

real_estate_2=real_estate_sales\
  .drop(columns=['Remarks'])\
  .dropna(subset=['PropertyType', 'AssessedValue', 'SaleAmount', 'DateRecorded', 'Address', 'SalesRatio', 'NonUseCode'])
real_estate_2['ResidentialType'].fillna('Land', inplace=True)
real_estate_2=real_estate_2[real_estate_2.PropertyType != '10 Mill Forest']
real_estate_2.info()

# COMMAND ----------

real_estate_2=real_estate_2.assign(SalesRatio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='SalesRatio', ascending=False)
df2=real_estate_2[(real_estate_2.SalesRatio<=5) & (real_estate_2.SalesRatio>0)] 
df2=df2[((df2.SaleAmount - df2.SaleAmount.mean()) / df2.SaleAmount.std()).abs() < 3]
df2.info()

# COMMAND ----------

display_pdf(df2.round(2))

# COMMAND ----------

real_estate_2.isnull().sum().sum()

# COMMAND ----------

Propertytype_mean = df_new.pivot_table(values=["AssessedValue", "SaleAmount", "SalesRatio"], index=["PropertyType", 'ListYear'], aggfunc=np.mean)\
.assign(Mean_sale_ratio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='Mean_sale_ratio', ascending=True)
Propertytype_mean.round(2)

# COMMAND ----------

df1.to_csv('/dbfs/FileStore/tables/maddy.csv', index=False, header=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists real_2013;
# MAGIC create temporary table real_2013 
# MAGIC using CSV 
# MAGIC options(path="/FileStore/tables/maddy.csv", header=TRUE)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM real_2013

# COMMAND ----------

Propertytype_mean = df_new.pivot_table(values=["AssessedValue", "SaleAmount"], index=["ResidentialType"], aggfunc=np.mean)\
.assign(Mean_sale_ratio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='Mean_sale_ratio', ascending=True)

Propertytype_mean.round(2)

# COMMAND ----------

Propertytype_median = df_new.pivot_table(values=["AssessedValue", "SaleAmount"], index=["PropertyType", "ResidentialType"], aggfunc=np.median)\
.assign(Mean_sale_ratio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='Mean_sale_ratio', ascending=True)
Propertytype_median.round(2)

# COMMAND ----------

Propertytype_mean = df_new.pivot_table(values=["AssessedValue", "SaleAmount", "SalesRatio"], index=["PropertyType", 'ListYear'], aggfunc=np.mean)\
.assign(Mean_sale_ratio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='Mean_sale_ratio', ascending=True)
Propertytype_mean.round(2)

# COMMAND ----------

import matplotlib.pyplot as plt
Propertytype_mean.plot(kind="scatter", x="SaleAmount", y="AssessedValue", alpha=0.4,
    s=Propertytype_mean["Mean_sale_ratio"]*100, label="Mean_sale_ratio", figsize=(7,5),
)
plt.legend()
display()

# COMMAND ----------

import seaborn as sns
g = sns.FacetGrid(df_new, col='ListYear', hue='ListYear', col_wrap=4)
g = g.map(plt.scatter, 'SaleAmount', 'AssessedValue')
g = g.map(plt.fill_between, 'SaleAmount', 'AssessedValue', alpha=0.2).set_titles("{col_name} ListYear")

display()


# COMMAND ----------

Town_sales_mean = df_new.pivot_table(values=["AssessedValue", "SaleAmount"], index=["Town"], aggfunc=np.median)\
.assign(Mean_sale_ratio=lambda x: x.AssessedValue/x.SaleAmount).sort_values(by='Mean_sale_ratio', ascending=True)
Town_sales_mean.round(2).head()

# COMMAND ----------

Town_sales_mean.round(2).tail()

# COMMAND ----------

import matplotlib.pyplot as plt
df_new.hist(bins=10, figsize=(20,15))
display()

# COMMAND ----------

df_new.plot(kind="scatter", x="SaleAmount", y="AssessedValue", alpha=0.3)
display()

# COMMAND ----------

df_new.plot(kind="scatter", x="SaleAmount", y="AssessedValue", alpha=0.4,
    s=df_new["SalesRatio"]*100, label="SalesRatio", figsize=(10,7),
    c="ListYear", cmap=plt.get_cmap("jet"),  colorbar=True,
)
plt.legend()
display()

# COMMAND ----------

# MAGIC %md ### 3. AFFORDABLE HOUSING DATASET

# COMMAND ----------

afford_table=pd.read_csv('/dbfs/FileStore/tables/Affordable_Housing_by_Town_2011_Present-ccc88.csv', header=0)\
.rename(columns= lambda x: x.lower().replace(' ','_'), inplace=False)\
.dropna()
afford_table.info()

# COMMAND ----------

display(afford_table.head())

# COMMAND ----------

afford_table_sum = afford_table.groupby([ "year"]).sum().sort_values(by='year', ascending=False)\
.assign(percent_affordable=lambda x: x['total_assisted_units']/x['2010_census_units'])\
.reset_index()
display(afford_table_sum.round(5))


# COMMAND ----------

# MAGIC %md ###4. MILL RATES DATASET

# COMMAND ----------

mill_rates_df=pd.read_csv('/dbfs/FileStore/tables/Mill_Rates_for_2017_Fiscal_Year.csv', header=0)\
.rename(columns= lambda x: x.lower().replace(' ','_'), inplace=False)\
.rename(columns={'fy_2017_mill_rate_-_real_&_personal_property__________________(pa_16-3_s.189_may_ss)':'fy_2017_mill_rate_2'})\

mill_rates_df.info()

# COMMAND ----------

display(mill_rates_df)

# COMMAND ----------

mill_rates_df = mill_rates_df.replace(np.nan, 0)
display(mill_rates_df)

# COMMAND ----------

mill_rates_df=mill_rates_df.assign(fy_2017_mill_rate=lambda x: x['fy_2017_mill_rate']+x['fy_2017_mill_rate_2'])
display(mill_rates_df)

# COMMAND ----------

display(mill_rates_df)

# COMMAND ----------

grouped_estate_sum=df_new.groupby(['Town', 'ListYear']).sum().sort_values(by='SaleAmount', ascending=False)
grouped_estate_sum=grouped_estate_sum.reset_index()
display_pdf(grouped_estate_sum.round(2))

# COMMAND ----------

grouped_estate_mean=df_new.groupby(['Town', 'ListYear']).mean().sort_values(by='ListYear', ascending=False)
grouped_estate_mean=grouped_estate_mean.reset_index()
display_pdf(grouped_estate_mean.round(2))

# COMMAND ----------

new_df_new=df_new.groupby(['ListYear', 'Address']).max()
display(new_df_new.round(2))

# COMMAND ----------

new_df_new.info()

# COMMAND ----------

df_new.info()

# COMMAND ----------

