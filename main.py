# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport


# Input 16
data_path = 'input/Worlds_Best_50_Hotels.csv'
df = pd.read_csv(data_path, header=None)

headers = ['Rank','Name','Location','Overview','Total Rooms','Starting Rate in ($)','Dining Area','Drinking Area','Hotel Ammenties','Address','Number']

df.columns = headers
print("Output [16]\n", df.head(), "\n")

# Input 17
print("Output [17]\n", df.tail(5), "\n")

# Input 18
print("Output [18]\n", df['Total Rooms'].dtype, "\n")

# Input 19
print("Output [19]\n", df.dtypes, "\n")

# Input 20
df_rank = df.groupby(['Location'])

# Input 21
#df.replace("?", np.nan, inplace = True)
df['Number'].str.replace('+', '*', regex=False)
df['Starting Rate in ($)'] = df['Starting Rate in ($)'].astype('float')
df_sub = df[df['Starting Rate in ($)'] > 1000]
print("Output [21]\n", df_sub, "\n")

# Input 22
df_f = df[ df['Total Rooms'] < 25]
print("Output [22]\n", df_f, "\n")

# Input 23
print("Output [23]\n", df.describe(), "\n")

# Input 24
print("Output [24]\n", df['Location'].value_counts(), "\n")

# Input 25
sns.boxplot(x='Location', y='Starting Rate in ($)', data=df_f)
# plt.show()

# Input 26
df_gptest1 = df.dropna(subset=['Name', 'Location', 'Total Rooms', 'Number'])
df_gptest = df[['Name', 'Location', 'Total Rooms']]
grouped_test1 = df_gptest.groupby(['Name', 'Location'], as_index=False).mean()
print("Output [26]\n", grouped_test1, "\n")

# Input 27
# Shows too much data because I can not df_gptest.groupby(['Name', 'Location'] or somehow else
sns.boxplot(x='Name', y='Total Rooms', data=grouped_test1)

# Input 28
numeric_df = df.select_dtypes(include='number')
correlation_matrix = numeric_df.corr()
print("Output [28]\n", correlation_matrix, "\n")



profile = ProfileReport(df, title="World's Best 50 Hotels Dataset Profiling Report", explorative=True)

# Save the report to an HTML file
profile.to_file("hotel_data_profiling_report.html")

# Optionally, display the report in Jupyter Notebook (if using one)
# profile.to_notebook_iframe()