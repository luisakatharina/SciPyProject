# SciPyProject
# Saracasm Detection
This project aimed to use Machine Learning to detect sarcasm in a custom made "News" Headline inputed by the user. 
Additionally the user is able to decide which of these four Classifiers they want to use to evaluate their custom headline: "Naive Bayes, k-nearest Neighbor, Random Forrest and Support Vector Machines. 
If you chose more than one the output of the better model will be taken.

## Libraries/Modules used
See requirements.txt for more information
- sklearn
- sklearnex (https://intel.github.io/scikit-learn-intelex/installation.html)
- numpy
- matplotlib
- pandas
- easygui (https://pypi.org/project/easygui/)
- json
- re
- nltk

## Data Preprocessing 
Luisa

## Classifiers + Evaluation

- Naive Bayes: Luisa
- KNN: Vanessa
- Random Forrest: Shabnam
- SVM: Isa

## userInput.py
The file has two functions: user_Input() and headline_Input()
With easygui the headline input and which classifier the user wants is saved.


## Plotting Evaluation + Dataframe


## Main

## Remarks on our Project
- Yields expected ouput [Sarcastic/not Sarcastic], but it has a very specific definition of what sarcastic is (most headlines, even very stupid ones, get classified as not sarcastic)
- this could be because the data used, whihc might train the models in a very particular way
- Due to data our definition of what is sarcastic is very narrow and limited. Sarcasm also depends a lot on context, tone of voice and expressions, which we cannot give.

