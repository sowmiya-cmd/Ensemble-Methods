Importing Libraries
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
Load the data 
 data = pd.read_csv('../input/prediction/fer (1).csv')  
      
    # Printing the dataswet shape 
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset obseravtions 
print ("Dataset: ",data.head(10)) 
 
    # Separating the target variable 
X = data.values[:, 1:12] 
Y = data.values[:, 13]

#DECISION TREE CLASSIFIER (ensemble method)
clf = DecisionTreeClassifier(max_depth=20, min_samples_split=5,
    random_state=100)
scores = cross_val_score(clf, X, Y, cv=5)
print("accuracy :",scores.mean())

#RANDOM FOREST( (ensemble method)

clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=5, random_state=100)
scores = cross_val_score(clf, X, Y, cv=5)
print("accuracy :",scores.mean())
