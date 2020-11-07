from data import read_data
from sklearn import tree
from sklearn.model_selection import train_test_split

df = read_data()


X = df.loc[:,df.columns != 'species']
y = df.species

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Training od the data

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

# tree.plot_tree(clf)

y_pred = clf.predict(X_test)
y_test = y_test.to_numpy()

print("The predictions are :", y_pred)
print("Actual target value for these Iris :", y_test)

# Evaluation 

erreur = sum(abs(y_pred-y_test))/len(y_test) # on obient une erreur de 4.4%

breakpoint()