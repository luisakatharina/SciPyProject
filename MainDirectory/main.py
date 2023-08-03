
from evaluation import evaluate_model, plot_roc
from preprocessing import *
from model import *
from userInput import user_Input
import easygui as eg
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(1)

    for classifier in chosen_Classifiers:
        y_pred, best_params, best_score, best_estimator = use_model(classifier, X_train, y_train, X_test)
        evaluate_model(classifier,best_params,best_score,y_test,y_pred)

        input_classification = best_estimator.predict(X_input)
        
        print(classifier, "classifies your headline as: ", convert(input_classification))
        print("----------------------------------------------------------------")
        plot_roc(ax=ax, fig=fig, X_test=X_test, model=best_estimator, y_test=y_test, model_name=classifier)
    
    print("plotting ROC curve...")
    plt.show()

    break