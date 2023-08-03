
from evaluation import evaluate_model, plot_roc
from preprocessing import *
from model import *
from userInput import *
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

    # get the training & test data and the vectorizer for the user input
    X_train, X_test, y_train, y_test, vectorizer = preprocess_and_split_data()

    headline_processed = preprocess_input(headline_input)
    X_user = vectorizer.transform([headline_processed]).toarray()

    # evaluating the chosen models' performances on the test set and get the "best" classification of the input
    max_acc = 0
    best_classifier = ""
    global best_classifier_estimator

    fig, ax = plt.subplots(1)  # for roc curve

    for classifier in chosen_Classifiers:
        y_pred, best_params, best_score, best_estimator = use_model(classifier, X_train, y_train, X_test)
        accuracy = evaluate_model(classifier,best_params,best_score,y_test,y_pred)

        # add classifier to the plot
        plot_roc(ax=ax, X_test=X_test, model=best_estimator, y_test=y_test, model_name=classifier)

        if accuracy > max_acc:
            max_acc = accuracy
            best_classifier = classifier
            best_classifier_estimator = best_estimator

    print("plotting ROC curve...")
    plt.show()
    print(best_classifier, "performed with the highest accuracy.")

    # classifier with the highest accuracy classifies the user input
    # the same classifier can be re-used for other headlines if the user wants to
    while True:
        print("Your headline: ", headline_input)
        input_classification = best_classifier_estimator.predict(X_user)
        print("\n It classifies your headline as: ",convert(input_classification))
        title = "another one?"
        ask = "Your headline has been classified. \nDo you want to try another one with the same classifier?"

        if eg.ynbox(ask, title):
            headline = user_headline()
            headline_processed = preprocess_input(headline_input)
            X_user = vectorizer.transform([headline_processed]).toarray()
            continue
        else:
            eg.msgbox("Goodbye!")
            break


    break