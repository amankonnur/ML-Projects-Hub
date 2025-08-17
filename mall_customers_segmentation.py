# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
#
# # define the dataset
# df = pd.read_csv("Mall_Customers.csv")
#
# # define features and target Variables
# X = df[["CustomerID", "Gender","Age","Annual Income (k$)"]]
# # target / output variable
# y = df["Spending Score (1-100)"]
#
# # split the data into training and testing(0.2)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
# model = KMeans(n_clusters=10,random_state=0)
# model.fit(X_train)
#
# predictions = model.predict(X_test)
#
# print("Predictions : ",predictions)
# # print("Accuracy : ",accuracy_score(y_test["Age"],predictions["Age"]))


"""chatgpt method"""

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# # Load dataset
# data = pd.read_csv("Mall_Customers.csv")
#
# # Take only Annual Income and Spending Score
# X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
#
# # Use Elbow Method to find optimal clusters
# inertia = []
# K = range(1, 11)
#
# for k in K:
#     model = KMeans(n_clusters=k, random_state=0)
#     model.fit(X)
#     inertia.append(model.inertia_)
#
# plt.plot(K, inertia, 'bo-')
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia (WCSS)")
# plt.title("Elbow Method for Optimal k")
# plt.show()
#
# # Train with optimal k (let's say k=5)
# kmeans = KMeans(n_clusters=5, random_state=0)
# data['Cluster'] = kmeans.fit_predict(X)
#
# # Visualize clusters
# plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
#             c=data['Cluster'], cmap='rainbow')
# plt.xlabel("Annual Income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("Customer Segments")
# plt.show()


from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

df  = pd.read_csv("Mall_Customers.csv")
# convert gender male to 0 and Female to 1
# becuase KMeans works only on Numerical data
df['Gender'] = df['Gender'].map({"Male":0,"Female":1})

# assign the features to variable
X = df[["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]

Inertia = []
for k in range(1,11):
    model = KMeans(n_clusters=k,random_state=0)
    model.fit(X)
    Inertia.append(model.inertia_)

plt.plot(range(1,11), Inertia, 'bo-')

Kmeans = KMeans(n_clusters=5,random_state=0)

df["Cluster"] = Kmeans.fit_predict(X)

# plt.plot(range(1,11), Inertia, 'bo-')
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia (WCSS)")
# plt.title("Elbow Method for Optimal k")
# plt.show()
#
# # Train with optimal k (let's say k=5)
# kmeans = KMeans(n_clusters=5, random_state=0)
# df['Cluster'] = kmeans.fit_predict(X)
#
# # Visualize clusters
# plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
#             c=df['Cluster'], cmap='rainbow')
# plt.xlabel("Annual Income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("Customer Segments")
# plt.show()


# Create subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Elbow curve
ax[0].plot(range(1, 11), Inertia, 'bo-')
ax[0].set_xlabel("Number of Clusters")
ax[0].set_ylabel("Inertia (WCSS)")
ax[0].set_title("Elbow Method for Optimal k")

# Plot 2: Cluster visualization
scatter = ax[1].scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
                        c=df['Cluster'], cmap='rainbow')
ax[1].set_xlabel("Annual Income (k$)")
ax[1].set_ylabel("Spending Score (1-100)")
ax[1].set_title("Customer Segments")

plt.tight_layout()
plt.show()
