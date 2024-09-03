# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
 import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
df=pd.read_csv("/content/titanic_dataset.csv")
df     
```

![361677677-7e5e17a3-0ca7-465f-b728-d3c67a2a4f7d](https://github.com/user-attachments/assets/819a4d37-1de9-45b7-8a50-01861b7f36bb)

df.info()

![361677153-d603f787-94f1-4138-819d-418a86f5ddea](https://github.com/user-attachments/assets/1e8e35b2-723c-42a5-8d2d-dc4f6704ebd5)

df.shape

![361677766-4e2ffc5f-d198-4f87-bf17-e75162558d79](https://github.com/user-attachments/assets/26e0985a-0457-45fd-93ab-321509fccf9c)

df.describe()

![361677915-bc6d662d-2259-4ca7-b3bf-a0c3e612cd4b](https://github.com/user-attachments/assets/9bef02e3-2329-48d9-af65-b30bae34d5d8)

df.set_index("PassengerId",inplace=True)
df.describe()

![361679106-f1cb5106-e322-4800-a8ca-87f242eeb6ac](https://github.com/user-attachments/assets/9e524b01-5e97-4d27-b202-25425a8032b3)

df.nunique()

![361679375-16756e99-833c-412a-ba62-39defb8408b6](https://github.com/user-attachments/assets/586c9072-1e7e-4250-b4c5-ea42520df815)

df["Survived"].value_counts()


![362538980-f8cd1730-ac50-4e4b-8299-df5e2d4cbc84](https://github.com/user-attachments/assets/5e98968e-6d9b-46a3-9c8a-7e5706ca9bc4)

per=(df["Survived"].value_counts()/df.shape[0]*100).round(2) 
per

![361680002-ca17452e-4508-4d6e-a724-a4c79a8a4152](https://github.com/user-attachments/assets/38c80359-4150-4f38-838d-cbdef5948d0f)

sns.countplot(data=df,x="Survived")


![361680213-6edb9293-3408-432a-aff6-29073117693b](https://github.com/user-attachments/assets/4d64a203-0e49-43ab-856e-b5c991cbfb0c)

df.Pclass.unique()

![361680424-b5fae832-3ca6-4bc9-9b38-c25ba29f4503](https://github.com/user-attachments/assets/73344077-6032-4f64-8967-4353fc7a1586)

df.rename(columns= {'Sex':'Gender'}, inplace=True)
df

![361680672-45c69834-ef85-4b91-9a98-af73278edd81](https://github.com/user-attachments/assets/efe63af6-0c0a-42e7-a918-2a7da3a02537)

sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)


![361681291-1f4e823f-312e-4a23-baa4-4372015fe9da](https://github.com/user-attachments/assets/4715a144-5805-4a0b-98b3-73e589c9850f)

sns.catplot(x='Survived',hue="Gender",data=df,kind="count")

![361681492-d5c1e2c6-7932-40a9-8ed6-36b86738b0bd](https://github.com/user-attachments/assets/12e243db-2b8e-40ff-9053-a43185e052b5)

df.boxplot(column="Age",by="Survived")

![361681628-9ef48650-fc8d-4d2b-9705-6fd2c5dac1f7](https://github.com/user-attachments/assets/012ec926-e1c7-459a-96c7-1a47fa5c0b9d)

sns.scatterplot(x=df["Age"],y=df["Fare"])

![361681792-7a11bb7d-5ee8-402a-92c6-0753d8e5ac27](https://github.com/user-attachments/assets/ec70f2ce-1912-4e58-a8ab-e89e9fa6d46f)

sns.jointplot(x="Age",y="Fare",data=df)

![361682057-a280a685-5f5d-4842-b53b-62042d0df77b](https://github.com/user-attachments/assets/9acb45a2-0977-45dd-b80e-c3073b68f4c9)

fig,ax1 = plt.subplots(figsize=(8,5)) 

pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)

![361682306-b1f4d0ae-9300-427a-8f7f-753d61661708](https://github.com/user-attachments/assets/e1811d4b-3079-4401-b3c9-804ecf9f4a81)

sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="Count")

![362540761-dc5ae73b-a7a5-45a2-a1dc-7e4f0f0bfb52](https://github.com/user-attachments/assets/42c08481-bc33-47dd-a21e-56b3cdc1e180)

corr = df.corr()
sns.heatmap(corr,annot=True)

![362541150-1444abfb-ca59-4081-9842-698369d1670b](https://github.com/user-attachments/assets/699da49f-1461-48c3-9eb1-921cf68344f1)

sns.pairplot(df)

![361682568-8478ee30-f907-483b-8772-2693e57ada6e](https://github.com/user-attachments/assets/908188eb-cb32-4e6c-9c3d-57915c36ac9a)


# RESULT
Hence performing Exploratory Data Analysis on the given data set is successful
