# extortion
This repository contains the code for extortion detection in Spanish  peruvian  texts. It also contains a dataset of  more than 900 Spanish phrases and its gold-standard. The results wof the experiments with the source code are included as well.

## INSTALLATION STEPS

Install Miniconda and create the environment "extortion" with Python 3.9. Install the packages spacy, pandas, scikit-learn, and matplotlib in the environment following the steps: 

$conda create --name extortion python=3.9

$conda activate extortion

$conda install -c conda-forge spacy
$python -m spacy download en_core_web_sm
$conda install -c conda-forge pandas
$conda install -c conda-forge scikit-learn
$conda install -c conda-forge matplotlib
Add the environment to an IDE such as Visual Studio.

## INPUT AND PRODUCED FILES

Input files: 
      "datav2.xlsx": The dataset of extortion and non-extortion messages. 
      "gold_standard.xlsx": The gold-standard of the dataset.
      
Output Files: 
      "list_words.xlsx" : A list of words obtained from the messages in dataset. 
      "classified_dictionary.xlsx": The list of classified words as threats, demands or neutral
      "classified_sentences.xlsx": The list of classified messages from datav2 as extortion or non-extortion 
      "confusion_matrix.xlsx": The comparison between the results in classified_sentences and the gold-standard

## STEPS FOR REPRODUCE THE EXPERIMENTS 

After installing the dependencies and loading the extortion environment, load the Visual Studio Project and/or run the Python file "ExtortionREVIEW.py"  

