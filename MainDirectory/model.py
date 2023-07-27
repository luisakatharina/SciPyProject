from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC

# to chose correct model to run based on user input
def use_model(classifier_name, X_train, y_train, X_pred):
    if classifier_name == "Naive Bayes":
        return  train_and_predict_with_naivebayes(X_train, y_train, X_pred)
    
    if classifier_name == "k-nearest Neighbor":
        return  train_and_predict_with_knn(X_train, y_train, X_pred)
    
    if classifier_name == "Random Forest":
        return  #train_and_predict_with_naivebayes(X_train, y_train, X_pred)
    
    if classifier_name == "Support Vector Machine":
        return  train_and_predict_with_SVM(X_train, y_train, X_pred)
    

# Naive Bayes Model:
def train_and_predict_with_naivebayes(X_train, y_train, X_pred):
    '''
    Fit NB classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = corresponding target variable
    '''
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train, y_train)

    # Make predictions: 
    # X_pred =  feature matrix, y_pred = predicted labels
    y_pred = naive_bayes_classifier.predict(X_pred)

    return y_pred #add , best_params, best_score

# K-nearest Neighbors Model:
def train_and_predict_with_knn(X_train, y_train, X_pred):
    '''
    Fit KNN classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = corresponding target variable
    k = number of nearest neighbors
    '''

    X_pred = X_pred.reshape(1, -1)
    print("train and predict starts")
    params_grid = {'n_neighbors': [2, 3]}

    # Grid search for parameter tuning
    grid_search = GridSearchCV( KNeighborsClassifier(), params_grid, cv=5, scoring='accuracy')
    print("grid search is fitting")
    grid_search.fit(X_train, y_train)

    # prediction
    print("prediction starts")
    y_pred = grid_search.best_estimator_.predict(X_pred)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return y_pred, best_params, best_score


# Suport Vector Machine Model:
def train_and_predict_with_SVM(X_train, y_train, X_pred):
    '''
    Fit SVM classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = corresponding target variable
    '''

    print("train and predict starts")
    # Grid search for parameter tuning
    params_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid_search = GridSearchCV( KNeighborsClassifier(), params_grid, cv=5, scoring='accuracy', refit=True,verbose=2)
    print("grid search is fitting")
    grid_search.fit(X_train, y_train)

    # prediction
    print("prediction starts")
    y_pred = grid_search.best_estimator_.predict(X_pred)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return y_pred, best_params, best_score
