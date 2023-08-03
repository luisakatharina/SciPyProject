from evaluation import evaluate_model
from preprocessing import *
from model import *
from userInput import user_Input
import easygui as eg

while True:

    headline_input, chosen_Classifiers = user_Input()

    headline_pp = preprocess_text(headline_input)
    X_pred = expand_contractions(headline_pp)

    if chosen_Classifiers == None:
        title = "There seems to be a problem..."
        ask = "Maybe you did not select any classifiers? \nDo you wish to rerun the program?"
        if eg.ynbox(ask, title):
            message = "Rerunning program..."
            eg.msgbox(message)
            continue
        else:
            break

    # get the training data
    X_train, X_test, y_train, y_test = preprocess_and_split_data()

    # im confused what we are using for the evaluation, since X_pred is only a single sample
    y_true = 1

    """ throws error: "ValueError: Expected 2D array, got scalar array instead: array=florida man nice person.
        Reshape your data either using array.reshape(-1, 1) if your data has a single feature 
        or array.reshape(1, -1) if it contains a single sample." """

    """if i use X_test instead of X_pred, naive bayes works fine, knn throws an error, SVM takes AGES (i think its due
        to the large amount of samples? I cant remember if random forest worked """

    for classifier in chosen_Classifiers:
        y_pred, best_params, best_score = use_model(classifier, X_train, y_train, X_pred)
        evaluate_model(classifier,best_params,best_score,y_true,y_pred)


# evaluate_model still needs parameters (best_params, best_score, y_true, y_pred)
#for classifier in chosen_Classifiers: # @shabnam because chosen_classifiers are given in a list
 #   correct variable names = use_model(classifier) @shabnam bitte dein model auch bei use_model function (bei model file) adden :)
 #   evaluation
    break