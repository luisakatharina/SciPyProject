from MainDirectory.evaluation import evaluate_model
from preprocessing import *
from model import *

#ask user for input
print("Hallo, this is a programm for you to write in a headline, and we will tell you according to different machine learning classifiers, whether it is most likeley sarcastic or not")

input_headline = input('Put in a headline, so we can tell you whether it is sarcastic or not.\n')
input_headline = preprocess_text(input_headline)
X_pred = expand_contractions(input_headline)

print("input:", X_pred)

print("you can choose between following options: ")
print("1: Naive Bayes ")
print("2: k-nearest kneighbors")
print("3: Random Forest")
print("4: Super Vector Machine ")
print("5: all and show which is the best")
print("note: we will always choose the best parameters and will give you an evaluation of the respective classifier/s ")

# get input
chosen_classifiers = input('Type in the number of all classifiers you want to use or 5 if you want to see all.\n')
print('okay thank you!')

#predict whether input sentence is sarcastic or not
X_train, X_test, y_train, y_test = preprocess_and_split_data()

#if chosen_classifier = 1 or 5 -> Naive bayes pred + evaluation
y_pred_naive_bayes = train_and_predict_with_naivebayes(X_train, y_train, X_pred)
evaluate_model("Naive Bayes", "none", "none", y_test, y_pred_naive_bayes)


#if chosen_classifier = 2 or 5 -> K-nearest Neighbors + evaluation
#y_pred_knn, best_params_knn, best_score_knn = train_and_predict_with_knn(X_train, y_train, X_pred)
#evaluate_model("K-nearest Neighbbours", best_params_knn, best_score_knn, y_test, y_pred_knn)


# do this for  3 and 4



