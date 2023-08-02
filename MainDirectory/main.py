from evaluation import evaluate_model
from preprocessing import *
from model import *
from userInput import user_Input
import easygui as eg

while True:

    headline_input, chosen_Classifiers = user_Input()

    headline_exp = expand_contractions(headline_input)
    X_pred = preprocess_text(headline_exp)

    #remove this if u want the shape error back lol
    X_pred = [X_pred]
    X_pred = vectorizeText(X_pred)
    # if i do this i get "ValueError: X has 2 features, but MultinomialNB is expecting 25852 features as input."




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

    #y_true = 1

    """ throws error: "ValueError: X has 2 features but [classifier] expects 25852 features"""

    """if we use X_test  for evaluation knn throws an error"""

    # evaluating the chosen models' performances on the test set
    for classifier in chosen_Classifiers:
        y_pred, best_params, best_score = use_model(classifier, X_train, y_train, X_test)
        evaluate_model(classifier,best_params,best_score,y_test,y_pred)


# evaluate_model still needs parameters (best_params, best_score, y_true, y_pred)
#for classifier in chosen_Classifiers: # @shabnam because chosen_classifiers are given in a list
 #   correct variable names = use_model(classifier) @shabnam bitte dein model auch bei use_model function (bei model file) adden :)
 #   evaluation
    break