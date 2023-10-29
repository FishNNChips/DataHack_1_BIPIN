import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from PIL import Image

st.header("AI SECTOR ANALYSIS")

df= pd.read_csv("AI sector.csv")
corr_matrix = df.corr()
st.write(corr_matrix)
img1=Image.open("download (1).png")
st.image(img1)
plt.matshow(corr_matrix)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.show()

X = df.iloc[:, 2:-1]
Y = df.iloc[:, 5]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
regressor = LinearRegression().fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# fig, ax = plt.subplots()
# st.scatter_chart(X_train.iloc[:, 2:4], Y_train, color="blue", label="Train set")
# st.scatter_chart(X_test.iloc[:, 2:4], Y_test, color="red", label="Test set")
X = pd.concat([X_train, X_test], axis=0)
y = np.concatenate((Y_train, Y_test))
regressor.fit(X.iloc[:, 2:3], y)
# ax.plot(X.iloc[:, 2], regressor.predict(X.iloc[:, 2:4]), color="green", label="Regression line")
# ax.title("Revenue v/s Funding for AI's")
# ax.xlabel("Funding Revieved")
# ax.ylabel("Revenue Predicted")
# ax.legend()
# fig.show()
st.subheader("Model")
img2=Image.open("download.png")
st.image(img2)



st.header("EV SECTOR ANALYSIS")
df= pd.read_csv("EV sector.csv")
corr_matrix = df.corr()
st.write(corr_matrix)

img3=Image.open("download2.png")
st.image(img3)

img4=Image.open("download (3).png")
st.image(img4)