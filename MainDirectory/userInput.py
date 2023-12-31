import easygui as eg # installation of easygui needed


def user_Input():
    '''
    This function creates pop-up windows that briefly explain the usage of the program and ask the user for their custom
    headline and the classifiers they want to compare in order to use the best one for prediction of their own headline's label.
    We obtain the custom headline as string and the classiifer choice as an array.
    '''

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


    headline_input = user_headline()
    
    
    ask_class ="Now you can choose from which of the following ML classifiers you want a prediction from.\nNote: we will always choose the best parameters and will give you an evaluation of the respective classifier/s"
    choices = ["Naive Bayes", "k-nearest Neighbor", "Random Forest", "Support Vector Machine"]
    choice = eg.multchoicebox(ask_class, title, choices)

    return headline_input, choice

def user_headline():
    '''
    This function specifically asks the user for the custom headline.
    '''
    title = "Input"
    satisfied_hl = False
    while satisfied_hl == False:
        headline_input = ""
        while headline_input == "":
            ask_input = "Please input your custom news headline below."
            headline_input = eg.enterbox(ask_input, title)
            if headline_input == None:
                quit()

        msg_verify = "Your Input:\n" + headline_input +"\n\nWould you like to continue or do you want to change your headline?"
        if eg.ccbox(msg_verify, title, ["Continue", "Change Headline"]):     # show a Continue/Cancel dialog
            satisfied_hl = True  # user chose Continue
        else:  # user chose change headline
            satisfied_hl = False

        return headline_input
