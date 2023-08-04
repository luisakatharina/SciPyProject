# SciPyProject
# Saracasm Detection
This project aimed to use Machine Learning to detect sarcasm in a custom made "News" Headline inputed by the user. 
Additionally the user is able to decide which of these four Classifiers they want to use to evaluate their custom headline: "Naive Bayes, k-nearest Neighbor, Random Forrest and Support Vector Machines. 
If you chose more than one the output of the better model will be taken.
This project uses the Sarcasm_Headlines_Dataset_v2.json data set (https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection). We split it 80/20 for train and test data set.
The evaluation is based on the test data set. 

## Intended Usage

- Make sure you have downloaded all packages required (see below or requirements.txt)
- run main.py file
- follow the instructions on the pop-ups (If you use pycharm, make sure you are not in full-screen mode to see the pop-ups)
- After inputting a headline and choosing which Classifiers you want on the pop-ups, the evaluation (precision, recall, f-score, Confusion Matric and AUC-score) will appear in the terminal (If you chose SVM, this step might take a while).
- In the end the ROC-Curve will be plotted --> The farther away the graph is from the diagonal the better the classifier.
  
### Example

- Input Headline: Astronaut Takes Blame for Gas in Spacecraft
- Chosen Classifiers: Naive Bayes, k-nearest Neighbor
  1. Evaluation:
      Naive Bayes:
        Precision= , Recall= ,F-score= , Confusion Matrix: , AUC_score=
-   Knn:
-     Precision= , Recall= ,F-score= , Confusion Matrix: , AUC_score=
-   ROC_curve:
-   Naive Bayes gets closer to 1 than knn, therefore it performs better.
  
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
ll

## Main
kk

## Remarks on our Project

- Yields expected ouput [Sarcastic/not Sarcastic], but it has a very specific definition of what sarcastic is (most headlines, even very stupid ones, get classified as not sarcastic)
- this could be because the data used, whihc might train the models in a very particular way
- Due to data our definition of what is sarcastic is very narrow and limited. Sarcasm also depends a lot on context, tone of voice and expressions, which we cannot give.
- Python might crash while executing code with SVM classifier IF YOU USE A MACBOOK. We did not encounter this problem with Windows11

