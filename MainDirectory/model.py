from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# Naive Bayes Model:
def train_and_predict_with_naivebayes(X_train, y_train, X_test):
    '''
    Fit NB classifier to training data
    X_train = training feature matrix obtained from TF-IDF conversion
    y_train = corresponding target variable
    '''
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train, y_train)

    # Make predictions: 
    # X_test = testing feature matrix, y_pred = predicted labels 
    y_pred = naive_bayes_classifier.predict(X_test)

    return y_pred

def evaluate_model_naivebayes(y_true, y_pred):
    '''
    Evaluating:
    confusion_matrix 
        -> computes confusion matrix
        -> provides summary of the classifier's performance
            --> true positives, true negatives, false positives, false negatives

    classification_report 
        --> generates detailed report 
            --> precision, recall, F1-score, support for each class
    '''
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))



# Model Selection, Parameter Tuning

# Classifiers with parameter grids for grid search
classifiers = [
    {
        'name': 'Naive Bayes',
        'classifier': MultinomialNB(),
        'params': {}
    },
    {
        'name': 'K-Nearest Neighbors',
        'classifier': KNeighborsClassifier(),
       # 'params': {} you have to add this here then
    },
    {
        'name': 'Random Forest',
        'classifier': RandomForestClassifier(),
       # 'params': {} you have to add this here then
    },
    {
        'name': 'Support Vector Machine',
        'classifier': SVC(),
       # 'params': {} you have to add this here then
    }
]

# Grid search with cross-validation for each classifier
def perform_grid_search(classifiers, X_train, y_train, X_test, y_test):
    for clf in classifiers:
        print(f"Grid Search for {clf['name']}...")

        # Define GridSearchCV object
        grid_search = GridSearchCV(clf['classifier'], param_grid=clf['params'], cv=5, scoring='accuracy')

        # Fit Model
        grid_search.fit(X_train, y_train)

        # Best parameters, Best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # Predictions using the best model
        y_pred = grid_search.best_estimator_.predict(X_test)

        # Evaluate the model
        print("Evaluation Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("----------------------------------------------")









