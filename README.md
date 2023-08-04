# SciPyProject
# Saracasm Detection
This project aimed to use Machine Learning to detect sarcasm in a custom made "News" Headline inputed by the user. 
Additionally the user is able to decide which of these four Classifiers they want to use to evaluate their custom headline: "Naive Bayes, k-nearest Neighbor, Random Forrest and Support Vector Machines. 
If you chose more than one the output of the better model will be taken.
This project uses the Sarcasm_Headlines_Dataset_v2.json data set (https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection). We split it 80/20 for train and test data set.
The evaluation is based on the test data set. 

## Intended Usage

- Make sure you have downloaded all packages required (see below or MainDirectory/requirements.txt)
- run main.py file
- follow the instructions on the pop-ups (If you use pycharm, make sure you are not in full-screen mode to see the pop-ups)
- After inputting a headline and choosing which Classifiers you want on the pop-ups, the evaluation (precision, recall, f-score, Confusion Matric and AUC-score) will appear in the terminal (If you chose SVM, this step might take a while).
- In the end the ROC-Curve will be plotted --> The farther away the graph is from the diagonal the better the classifier.
  
### Example

1. Input Headline: Astronaut Takes Blame for Gas in Spacecraft
2. Chosen Classifiers: Naive Bayes, k-nearest Neighbor
3. Evaluation:
      - Naive Bayes:
        - Precision= , Recall= ,F-score= , Confusion Matrix: , AUC_score=
        - Naive Bayes classifies the headline as [Sarcastic]
      - Knn:
        - Precision= , Recall= ,F-score= , Confusion Matrix: , AUC_score=
        - Knn classifies the headline as []
4. ROC_curve:
      - Naive Bayes gets closer to 1 than knn, therefore it performs better.
  
## Libraries/Modules used
See MainDirectory/requirements.txt for more information
- sklearn
- sklearnex 
- numpy
- matplotlib
- pandas
- easygui 
- json
- re
- nltk

## preprocessing.py

Cleans data and user input (e.g. everything to lower case, gets rid of stopwords, expands contractions), Vectorizes text --> creates feature Matrix, Splits data into train and test data set (80/20).


## model.py

Functions for all four classifiers (Naive Bayes, k-nearest Neighbor, Random Forest, Support Vector Machines): fits classifier, trains model, predicts test data.
Seperate function to get correct classifier function based on string name of classifier.

## userInput.py

The file has two functions: user_Input() and headline_Input()
With easygui the headline input and which classifier the user wants is saved in seperate variables and then ouputted. 
headline_Input() used by user_Input(), they are seperated to create a loop if user whishes to input another headline after evaluating previous headline.

## evaluation.py

Model evaluated through classification report and confusion matrix in function evaluate_model().
ROC_curve as well as AUC_score of model generated in function plot_roc() as further evaluation and comparison between models if more than one selected.

## main.py

Makes use of all previous functions. User runs programm by executing main.py.

## Remarks on our Project

- Yields expected ouput [Sarcastic/not Sarcastic], but it has a very specific definition of what sarcastic is (most headlines, even very stupid ones, get classified as not sarcastic)
- this could be because the data used, whihc might train the models in a very particular way
- Due to data our definition of what is sarcastic is very narrow and limited. Sarcasm also depends a lot on context, tone of voice and expressions, which we cannot give.
- Python might crash while executing code with SVM classifier IF YOU USE A MACBOOK. We did not encounter this problem with Windows11

