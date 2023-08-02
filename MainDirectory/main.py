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
            message = "Rerunning programm..."
            eg.msgbox(message)
            continue
        else:
            quit()

# evaluate_model still needs parameters (best_params, best_score, y_true, y_pred)
#for classifier in chosen_Classifiers: # @shabnam because chosen_classifiers are given in a list
 #   correct variable names = use_model(classifier) @shabnam bitte dein model auch bei use_model function (bei model file) adden :)
 #   evaluation
