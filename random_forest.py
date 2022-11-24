import numpy as np
from Decision_tree import DecisionTreeClassifier

class randomforest:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.trees = []

    def bootstrap(self,X,Y):
        n_rows=X.shape[0]
        indexs =np.random.choice(n_rows, n_rows,replace=True)
        return X[indexs],Y[indexs]
        _

    def fit(self,X,Y):
        self.trees = []
        for i in range(self.n_estimators):
            dtree=DecisionTreeClassifier()
            X_rows,Y_rows = self.bootstrap(X,Y)
            dtree.fit(X_rows,Y_rows)
            self.trees.append(dtree)

    def mojor_vote(self,Y):
        vals,counts = np.unique(Y, return_counts=True)
        index = np.argmax(counts)
        return vals[index]

    def predict(self,X):
        tree_preds = np.swapaxes(np.array([dtree.predict(X) for dtree in self.trees]),0,1)
        predictions = np.array([self.mojor_vote(i) for i in tree_preds])
        return predictions



