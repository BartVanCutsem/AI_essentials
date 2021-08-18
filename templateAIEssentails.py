# pip install sklearn numpy pandas

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


##########################  KNN ##########################
def KNN(X, y):
    n_neighbors = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22]
    algorithm = ['auto']
    weights = ['uniform', 'distance']
    leaf_size = list(range(1, 50, 5))
    hyperparams = {
        'algorithm': algorithm,
        'weights': weights,
        'leaf_size': leaf_size,
        'n_neighbors': n_neighbors
    }
    # andere scroing methodes
    # ['accuracy', 'f1', 'balanced_accuracy', 'roc_auc']
    clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams, verbose=True,
                       cv=10, scoring='balanced_accuracy', n_jobs=-1)
    clf.fit(X=X, y=y)
    print(clf.best_score_)
    print(clf.best_estimator_)


##########################  Logestic Regression ##########################
def logicRegression(X, y):
    solver = ['lbfgs', 'sag', 'newton-cg']
    tol = [1e-4, 1e-3, 1e-2]
    C = [1e-2, 0.1, 1, 5, 10]
    hyperparams = {'solver': solver, 'tol': tol, 'C': C}
    clf = GridSearchCV(estimator=LogisticRegression(), param_grid=hyperparams, verbose=True, cv=10,
                       scoring='balanced_accuracy', n_jobs=-1)

    clf.fit(X=X, y=y)
    print(clf.best_score_)
    print(clf.best_estimator_)


##########################  Decision_tree  ##########################
def decision_tree(X, y):
    criterion = ['gini', 'entropy']
    max_depth = [None, 3, 5, 7, 10]
    min_samples_split = [2, 5, 10, 0.1]
    hyperparams = {'criterion': criterion, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                   }
    clf = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=hyperparams, verbose=True, cv=10,
                       scoring='balanced_accuracy', n_jobs=-1)

    clf.fit(X, y)
    print(clf.best_score_)
    print(clf.best_estimator_)
    print(clf.best_estimator_.feature_importances_)


##########################  Random Forest  ##########################
def randomForest(X, y):
    n_estimators = [2, 3, 5, 10, 25, 50, 100]
    criterion = ['gini', 'entropy']
    max_depth = [None, 3, 5, 7, 10]
    min_samples_split = [2, 5, 10, 0.1]
    min_impurity_split = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    hyperparams = {'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_impurity_split': min_impurity_split}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=hyperparams, verbose=True, cv=10,
                       scoring='balanced_accuracy', n_jobs=-1)

    clf.fit(X, y)
    print(clf.best_score_)
    print(clf.best_estimator_)
    print(clf.best_estimator_.feature_importances_)


if __name__ == '__main__':
    pass
    ### inladen in data
    df = pd.read_csv("Path naar bestand")

    ### waarden aanpassen
    # checken dat er lege waarden instaan
    print(df.isnull().values.any())
    # Lege velden vervangen door NaN
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    # Nan Values droppen
    df.dropna(inplace=True)

    # Checken dat kolomn zijn met dezelfde values
    cols = df.select_dtypes([np.number]).columns
    std = df[cols].std()
    cols_to_drop = std[std == 0].index
    df = df.drop(cols_to_drop, axis=1)

    # shufflen van de data
    df = df.sample(frac=1).reset_index(drop=True)

    # onderverdelen van de data in test en train 70-30
    Size = len(df.index)
    Size = int(Size * 0.7)
    train = df[:Size]
    test = df[Size:]

    # nodige kolommen uit de df halen + verwijderen uit de df (labels)
    label = df["kolom_naam"]
    df = df.drop(colomn=["kolom_naam", "kolom_naam"])

    # normaliseren van de data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dataframe)

    # modellen testen



