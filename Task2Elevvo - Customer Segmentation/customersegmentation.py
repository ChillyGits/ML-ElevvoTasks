import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
df = pd.read_csv("Mall_Customers.csv")

X=df[['Annual Income (k$)','Spending Score (1-100)']]#selecting the columns to cluster on

scaler = StandardScaler() #calling the scaler function
X_scaled = scaler.fit_transform(X)#makes selected column features have the same importance to perform clustering
inertia_result = []
#determining elbow point (suitable no of clusters) and plotting inertia points depending on no. of clusters
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, random_state = 67)
    kmeans.fit(X_scaled)
    inertia_result.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia_result, marker='o')#used to find the elbow point
plt.grid(True, which='both', linestyle='--', linewidth=0.7)#graph look
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

############################################################

kmeans = KMeans(n_clusters=5, random_state=67)#elbow(suitable no of clusters) point = 5
df['Cluster'] = kmeans.fit_predict(X_scaled)
cluster_labels = {
    0: "Average Customers",
    1: "Cautious Rich",
    2: "Budget Shoppers",
    3: "Luxury Spenders",
    4: "Impulsive Buyers"
}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

plt.figure(figsize=(8, 6))
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],c=df['Cluster'], cmap='rainbow', s=100, alpha=0.7)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)#graph look

centers = kmeans.cluster_centers_ #shows the center point of each cluster
centers = scaler.inverse_transform(centers)  # convert back to original scale
for i, (x, y) in enumerate(centers):
    plt.scatter(x, y, c='black', s=200, marker='X')  # mark center
    plt.text(x, y, cluster_labels[i], fontsize=10, ha='center', va='bottom', color='black')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation (using KMeans clustering)')
plt.show()



dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)#graph look
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['DBSCAN_Cluster'], cmap='plasma', s=100, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering')
plt.show()
