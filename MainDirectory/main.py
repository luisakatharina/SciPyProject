from evaluation import evaluate_model
from preprocessing import *
from model import *
from userInput import user_Input

headline_input, chosen_Classifiers = user_Input()

headline_pp = preprocess_text(headline_input)
X_pred = expand_contractions(headline_pp)

if chosen_Classifiers == None:
    print("There seems to be a problem... maybe you did not select any classifiers? Or you terminated the program early?")
    answer = input("Do you wish to retry the program? y/n")
    if answer == "y" or answer == "Y":
        #run main again --> Please shabnam can you implement? :)
        print("Rerun program")
    else:
        quit()

# evaluate_model still needs parameters (best_params, best_score, y_true, y_pred)
#for classifier in chosen_Classifiers: # @shabnam because chosen_classifiers are given in a list
 #   correct variable names = use_model(classifier) @shabnam bitte dein model auch bei use_model function (bei model file) adden :)
 #   evaluation
