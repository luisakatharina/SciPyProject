from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

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







