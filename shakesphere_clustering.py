'''
=========================================================================
This file clusters the Shakespeare play scripts and then creates a hierarcical
dendogram.

Based on code from:
https://pythonprogramminglanguage.com/kmeans-text-clustering/
The Indo-European Languages Dendogram which is based on:
https://doc.lagout.org/science/0_Computer%20Science/Computational%20Linguistics/
Statistics/johnson2008-quantitative_methods_in_linguistics/6historical.pdf
=========================================================================
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# files will need to be in same directory or paths updated
titles = 'shakespearePlayTitles.txt'
allText = 'shakespeare.txt'
textBoundary = '<<NEWTEXTSTARTSHERE>>' # play break

playTitles = open(titles, "r", encoding="utf8").read()
playTitles = playTitles.split("\n") # split by newline

playScripts = open(allText, "r", encoding="utf8").read()
playScripts = playScripts.split(textBoundary) # split by script

vectorizer = TfidfVectorizer(stop_words='english') # create Tf-idf vector
X = vectorizer.fit_transform(playScripts) # create matrix w/ weights

true_k = 10 # 10 clusters
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=15)
model.fit(X) # create model and fit, specified random seed for reproducibility

print("----------------------")
print("Top terms per cluster:")
print("----------------------")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

# print top 3 features per cluster
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
print("----------------------")

print("\n")
print("---------------------------------")
print("{:<25}{:<5}".format("Title","Cluster"))
print("---------------------------------")
# print play titles and which cluster they fit in
labels = model.labels_.tolist()
for idx in range(len(playTitles)):
    print("{:<25}{:<5}".format(playTitles[idx],str(labels[idx])))
print("---------------------------------")
print("\n\n")

print("-----------------------")
print("Prediction")
print("-----------------------")
# run predictions on the 2 follwing lines
Y = vectorizer.transform(["battle and king"])
prediction = model.predict(Y)
print(prediction)
Y = vectorizer.transform(["wit and love"])
prediction = model.predict(Y)
print(prediction)
print("-----------------------")
print("\n\n")

# create dendogram
Z = linkage(X.todense(), 'ward')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance (Single)')
dendrogram(Z, labels=playTitles, orientation="left", truncate_mode='level', leaf_rotation=0, leaf_font_size=6)
# plot dendogram
plt.savefig('ie-dendrogram.pdf')
