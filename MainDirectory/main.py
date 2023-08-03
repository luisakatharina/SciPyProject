from evaluation import evaluate_model, plot_roc
from preprocessing import *
from model import *
from userInput import user_Input
import easygui as eg
import matplotlib.pyplot as plt

while True:
    # get user input
    headline_input, chosen_Classifiers = user_Input()

    # in case no classifier was chosen
    if chosen_Classifiers == None:
        title = "There seems to be a problem..."
        ask = "Maybe you did not select any classifiers? \nDo you wish to rerun the program?"
        if eg.ynbox(ask, title):
            message = "Rerunning program..."
            eg.msgbox(message)
            continue
        else:
            break

    # preprocess headline input
    print("Your headline: ", headline_input)
    headline_exp = expand_contractions(headline_input)
    headline_processed = preprocess_text(headline_exp)


    # get the training data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_and_split_data()
    X_input = vectorizer.transform([headline_processed]).toarray()

    # evaluating the chosen models' performances on the test set and get their classification of the input

    fig, ax = plt.subplots(1) # for roc curve

    for classifier in chosen_Classifiers:
        # for each chosen classifier run its corresponding model and evaluate it
        y_pred, best_params, best_score, best_estimator = use_model(classifier, X_train, y_train, X_test)
        evaluate_model(classifier,best_params,best_score,y_test,y_pred)

        # classify inputed headline
        input_classification = best_estimator.predict(X_input)
        print(classifier, "classifies your headline as: ", convert(input_classification))
        print("----------------------------------------------------------------")

        # add classifier to the plot
        plot_roc(ax=ax, X_test=X_test, model=best_estimator, y_test=y_test, model_name=classifier)
    
    # show plot with all chosen classifiers
    print("plotting ROC curve...")
    plt.show()

    break