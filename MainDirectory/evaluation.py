from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(classifier_name, best_params, score, y_true, y_pred):
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

    #cross validation?

    print("Evaluation of ", classifier_name," model")

    # Best parameters, Best score
    print("Best Parameters:", best_params)
    print("Score:", score)

    print("Evaluation Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("----------------------------------------------")

    return 0
