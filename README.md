# SciPyProject
# Saracasm Detection
This project aimed to use Machine Learning to detect sarcasm in a custom made "News" Headline inputed by the user. 
Additionally the user is able to decide which of these four Classifiers they want to use to evaluate their custom headline: "Naive Bayes, k-nearest Neighbor, Random Forrest and Support Vector Machines. 
If you chose more than one the output of the better model will be taken.
This project uses the Sarcasm_Headlines_Dataset_v2.json data set (https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection). We split it 80/20 for train and test data set.
The evaluation is based on the test data set. 

### Dataset citation

1. Misra, Rishabh and Prahal Arora. "Sarcasm Detection using News Headlines Dataset." AI Open (2023).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

The source of this data set: https://rishabhmisra.github.io/publications/ 

## Intended Usage

- Make sure you have downloaded all packages required (see below or MainDirectory/requirements.txt)
- run main.py file
- follow the instructions on the pop-ups (If you use pycharm, make sure you are not in full-screen mode to see the pop-ups)
- After inputting a headline and choosing which Classifiers you want on the pop-ups, the evaluation (precision, recall, f-score, Confusion Matric and AUC-score) will appear in the terminal (If you chose SVM, this step might take a while).
- In the end the ROC-Curve will be plotted --> The closer the graph to the point (0,1), the higher Sensitivity and Specificity and the more correct the classifier is at classifying all the Positive and Negative class points.
  
### Example

1. Input Headline: Astronaut Takes Blame for Gas in Spacecraft
2. Chosen Classifiers: Naive Bayes, k-nearest Neighbor
3. Evaluation:
      - Naive Bayes:
        - Best Parameter: None (from GridSearch --> Naive Bayes has no hyper-parameters to tune)
        - Score: 0.7953701681589866
        - Precision = 0.81
        - Recall = 0.75
        - F-score = 0.78
        - Accuracy = 0.8
        - Confusion Matrix: [[True Positives=1532, False Positives=469], [False Negatives= 678, True Negatives=2045]]
      - Knn:
        - Best Parameter: {'n_neighbors': 3} (from GridSearch, 3 neighbors best hyperparameter)
        - Score: 0.5709980345053505
        - Precision = 0.92
        - Recall = 0
        - F-score = 0.01
        - Accuracy = 0.53 
        - Confusion Matrix: [[True Positives=3000, False Positives=1], [False Negatives= 2711, True Negatives=12]]
4. ROC_curve:
   
  ![Figure_Scipy_Example_knn-NB](https://github.com/luisakatharina/SciPyProject/assets/110250036/f0dbfcf1-a2e4-41a9-a903-62a29b8684a4)
  
  - It is obvious from the plot that the AUC for the Naive Bayes ROC curve is higher than that for the KNN ROC curve. Therefore, we can say that Naive Bayes did a better job of classifying the positive class in the dataset.
  
6. Classification of headline (Best model - according to accuracy - of the chosen classifiers is used for classification of the headline)
      - Naive Bayes classifies the headline as [Sarcastic]
  
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

- Our program yields expected ouput: It classifies a headline as [Sarcastic] or [not Sarcastic] and outputs the evaluation of the selected models. But it classifies a headline most frequently as [not Sarcastic]. We believe this could be due to a very specific definition of what sarcastic is because of the data we chose to use. Since it only uses official headlines from "The Onions" (for sarcastic headlines of current events) and "Huffpost" (for real and non-sarcastic headlines) and the dataset was built mainly for Sarcasm in news headlines as well as fake news detection, the models are built for a very narrow definition of sarcasm. Additionally, our model does not take into account any semantics of the words and so a headline like "Cows lose jobs" is classified as [not Sarcastic]. Sarcasm is very dependent on context, voice tonality and facial expressions, so naturally building an algorithm to detect sarcasm is extremely difficult and demanding.
- Python might crash while executing code with SVM classifier IF YOU USE A MACBOOK. We did not encounter this problem (or any other complications) running the program using Windows11. 

