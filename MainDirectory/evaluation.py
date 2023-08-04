from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
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

    report = classification_report(y_true, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("----------------------------------------------")

    return report['accuracy']

def plot_roc(ax, X_test, model, y_test, model_name):
    """
    Function does not return anything.

    pred_prob => predict probabilities of chosen model for roc curve.
    calculates False Positive Rate (fpr) and True Positive Rate (tpr) and threshold of roc curves of the models.
    generates also the neccessary values (p_fpr, p_tpr) for a diagonal of the plot (tpr=fpr).
    Calcualtes the AUC score for the model.
    adds values as a function into the given axes of a plot.
    """

    print("predicting probabilities of model")
    pred_prob = model.predict_proba(X_test)[:, 1]

    # roc curve for models
    fpr, tpr, thresh = roc_curve(y_test, pred_prob, pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    # auc score f model
    auc_score = roc_auc_score(y_test, pred_prob)

    print("Preparing Roc curve plot")

    # color of plots:
    if model_name == "Naive Bayes":
        color = 'orange'
    if model_name == "k-nearest Neighbor":
        color = 'red'
    if model_name == "Random Forest":
        color = 'green'
    if model_name == "Support Vector Machine":
        color = 'darkviolet'

    # plot roc curves
    ax.plot(fpr, tpr, linestyle='--', color=color, label=model_name + " (AUC_score = " + str(auc_score))
    ax.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    ax.set_title('ROC curve')
    # x label
    ax.set_xlabel('False Positive Rate')
    # y label
    ax.set_ylabel('True Positive rate')

    ax.legend(loc='best')
