# download in your environment with: conda install scikit-learn-intelex -c conda-forge OR pip install scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from numpy import mean


# to chose correct model to run based on user input
def use_model(classifier_name, X_train, y_train, X_pred):
    '''
    Easy access to training & testing function through classifier name.
    This function faciliates working with the user input.
    classifier_name = name of the classifier to be used
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = target variables corresponding to X_train
    X_pred = feature matrix obtained from TF-IDF conversion for testing
    '''
    if classifier_name == "Naive Bayes":
        return train_and_predict_with_naivebayes(X_train, y_train, X_pred)
    
    if classifier_name == "k-nearest Neighbor":
        return train_and_predict_with_knn(X_train, y_train, X_pred)
    
    if classifier_name == "Random Forest":
        return train_and_predict_with_RF(X_train, y_train, X_pred)
    
    if classifier_name == "Support Vector Machine":
        return train_and_predict_with_SVM(X_train, y_train, X_pred)
    

# Naive Bayes Model:
def train_and_predict_with_naivebayes(X_train, y_train, X_pred):
    '''
    Fit NB classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = target variables corresponding to X_train
    X_pred = feature matrix obtained from TF-IDF conversion for testing
    '''
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train, y_train)

    # Make predictions:
    y_pred = naive_bayes_classifier.predict(X_pred)

    #no gridsearch because Naive Bayes has no hyper-parameters to tune
    best_params = None
    mean_score = mean(cross_val_score(naive_bayes_classifier, X_train, y_train, cv=5, scoring='accuracy'))

    return y_pred, best_params, mean_score, naive_bayes_classifier

# K-nearest Neighbors Model:
def train_and_predict_with_knn(X_train, y_train, X_pred):
    '''
    Fit KNN classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = target variables corresponding to X_train
    X_pred = feature matrix obtained from TF-IDF conversion for testing
    '''

    print("train and predict for knn starts")
    params_grid = {'n_neighbors': [2, 3]}

    # Grid search for hyperparameter tuning
    # We had to limit grid search to very few options, otherwise it would have taken too long
    grid_search = GridSearchCV( KNeighborsClassifier(), params_grid, cv=5, scoring='accuracy')
    print("grid search is fitting")
    grid_search.fit(X_train, y_train)

    # Prediction
    print("prediction starts")
    y_pred = grid_search.best_estimator_.predict(X_pred)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return y_pred, best_params, best_score, grid_search.best_estimator_


# Suport Vector Machine Model:
def train_and_predict_with_SVM(X_train, y_train, X_pred):
    '''
    Fit SVM classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = target variables corresponding to X_train
    X_pred = feature matrix obtained from TF-IDF conversion for testing
    '''

    print("train and predict for SVM starts")
    # Grid search for hyperparameter tuning
    # We had to limit the amount of options because it would have taken way too long otherwise
    params_grid = {'C': [0.1,1],
                   'kernel': ['rbf', 'sigmoid']}
    grid_search = GridSearchCV(SVC(), params_grid, cv=2, scoring='accuracy', refit=True,verbose=2)
    print("grid search is fitting")
    grid_search.fit(X_train, y_train)

    # prediction
    print("prediction starts")
    y_pred = grid_search.best_estimator_.predict(X_pred)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return y_pred, best_params, best_score, grid_search.best_estimator_

def train_and_predict_with_RF(X_train, y_train, X_pred):

    '''
    Fit Random Forest classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = target variables corresponding to X_train
    X_pred = feature matrix obtained from TF-IDF conversion for testing
    '''

    print("training for Random Forest starts")

    classifier = RandomForestClassifier(verbose=3)

    # grid search for hyperparameter tuning
    # we had to limit the amount of options because it would have taken way too long otherwise
    grid_space = {'max_depth': [50,100],
                  'n_estimators': [10, 100],
                  }

    grid_search = GridSearchCV(classifier, param_grid=grid_space, cv=3, scoring='accuracy')

    print("grid search is fitting")
    grid_search.fit(X_train, y_train)

    #prediction
    print("model is predicting")

    y_pred = grid_search.best_estimator_.predict(X_pred)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_


    return y_pred, best_params, best_score, grid_search.best_estimator_
