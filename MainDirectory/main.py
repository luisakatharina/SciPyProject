
from evaluation import evaluate_model
from preprocessing import *
from model import *
from userInput import user_Input
import easygui as eg


while True:

    headline_input, chosen_Classifiers = user_Input()

    if chosen_Classifiers == None:
        title = "There seems to be a problem..."
        ask = "Maybe you did not select any classifiers? \nDo you wish to rerun the program?"
        if eg.ynbox(ask, title):
            message = "Rerunning program..."
            eg.msgbox(message)
            continue
        else:
            break

    print("Your headline: ", headline_input)
    headline_exp = expand_contractions(headline_input)
    headline_processed = preprocess_text(headline_exp)


    # get the training data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_and_split_data()
    X_input = vectorizer.transform([headline_processed]).toarray()

    # evaluating the chosen models' performances on the test set and get their classification of the input
    for classifier in chosen_Classifiers:
        y_pred, best_params, best_score, best_estimator = use_model(classifier, X_train, y_train, X_test)
        evaluate_model(classifier,best_params,best_score,y_test,y_pred)

        input_classification = best_estimator.predict(X_input)
        print(classifier, "classifies your headline as: ", convert(input_classification))

    break