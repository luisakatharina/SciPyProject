from evaluation import evaluate_model
from preprocessing import *
from model import *
from userInput import user_Input

headline_input, chosen_Classifiers = user_Input()

headline_pp = preprocess_text(headline_input)
X_pred = expand_contractions(headline_pp)

# evaluate_model still needs parameters (best_params, best_score, y_true, y_pred)
for classifier in chosen_Classifiers:
    evaluate_model(classifier) # because chosen_classifiers are given in a list
