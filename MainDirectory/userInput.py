#from MainDirectory.evaluation import evaluate_model
#from preprocessing import *
#from model import *
import easygui as eg # installation of easygui needed

def user_Input():
    title = "What this program does"
    explain = "Hello, this is a programm where you can input a custom written news headline, and we will tell you (according to different machine learning classifiers), whether it is most likeley supposed to be sarcastic or not."
    if eg.msgbox(explain, title=title,ok_button="OK") == None:
        quit()

    title = "Please Confirm"
    ask = "Would you like to continue?"
    if eg.ccbox(ask, title):     # show a Continue/Cancel dialog
        pass  # user chose Continue
    else:  # user chose Cancel
        quit()

    title = "Input"
    satisfied_hl = False
    while satisfied_hl == False:
        headline_input = ""
        while headline_input == "":
            ask_input = "Please input your custom news headline below."
            headline_input = eg.enterbox(ask_input, title)
            if headline_input == None:
                quit()
        # preprocess headline_input 

        msg_verify = "Your Input:\n" + headline_input +"\n\nWould you like to continue or do you want to change your headline?"
        if eg.ccbox(msg_verify, title, ["Continue", "Change Headline"]):     # show a Continue/Cancel dialog
            satisfied_hl=True  # user chose Continue
        elif eg.ccbox(msg_verify, title, ["Continue", "Change Headline"]) == None: 
            quit()
        else:  # user chose Cancel
            satisfied_hl=False
    
    
    
    ask_class ="Now you can choose from which of the following ML classifiers you want a prediction from.\nNote: we will always choose the best parameters and will give you an evaluation of the respective classifier/s"
    choices = ["Naive Bayes", "k-nearest Neighbor", "Random Forest", "Support Vector Machine", "All"]
    choice = eg.choicebox(ask_class, title, choices)
    if choice == None:
        quit()

    return headline_input, choice


user_Input()

'''
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


# do this for  3 and 4'''