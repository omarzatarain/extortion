# extortion
This repository contains the code for extortion detection in Spanish  peruvian  texts. It also contains a dataset of  more than 900 Spanish phrases and its gold-standard. The results wof the experiments with the source code are included as well.

## DATASET INFORMATION 
The dataset included in this repository is located at  "datav2.xlsx". The dataset contains 986 extorsion and non-extortion messages. A gold-standard for the dataset is provided in the file  "gold_standard.xlsx" which contains the label as extortion or non-extortion message. Both files are used as input at the process described forward in INSTRUCTIONS. 

## CODE
The python code in this repository  is located in the Python file "ExtortionREVIEW.py", additional project files are included for running the project in Visual Studio 2022.

## REQUIREMENTS

Install Miniconda and create the environment "extortion" with Python 3.9. Install the packages spacy, pandas, scikit-learn, matplotlib  and pysentimiento in the environment following the steps: 

$conda create --name extortion python=3.9

$conda activate extortion

$conda install -c conda-forge spacy

$python -m spacy download en_core_web_sm

$conda install -c conda-forge pandas

$conda install -c conda-forge scikit-learn

$conda install -c conda-forge matplotlib

$conda install -c conda-forge openpyxl

THE FOLLOWING STEPS ARE FOR USING A JUPYTER NOTEBOOK

$conda install ipykernel  

$python -m ipykernel install --user --name=extortion --display-name="Python (extortion)"


Load the python file ExtortionREVIEW into a jupyter notebook or Add the environment to an IDE such as Visual Studio.
Load the files "datav2.xlsx" and  "gold_standard.xlsx" into the jupyter notebook folder or the visual studio project's folder.


## INPUT FILES  AND PRODUCED FILES

Input files: 

      "datav2.xlsx": The dataset of extortion and non-extortion messages. 
      
      "gold_standard.xlsx": The gold-standard of the dataset.
      
Output Files: 

      "list_words.xlsx" : A list of words obtained from the messages in dataset. 
      
      "classified_dictionary.xlsx": The list of classified words as threats, demands or neutral.
      
      "classified_sentences.xlsx": The list of classified messages from datav2 as extortion or non-extortion 
      
      "confusion_matrix.xlsx": The comparison between the results in classified_sentences and the gold-standard

The output files will be generated on  experiments reproduction, however, the output files are provided for the convenience of the readers of this repository.

## INSTRUCTIONS 

Open a Jupyter notebook or  Visual Studio Project and run the Python file "ExtortionREVIEW.py" and the input files  "datav2.xlsx" and "gold_standard.xlsx". The output files will be produced after the following steps 

The process contains the following steps:

First, the list of words that will form the basis of the data dictionary is created.  This list of words will be the historical cloud of words commonly used in extortion sentences.

Second, the data dictionary is generated based on the list generated in the previous step. This algorithm classifies words of the “demand” type and words of the “threat” type. This analysis is based on a database of words initially entered.

Third,  the algorithm classifies the phrases entered based on the created dictionary. There are four classes: No Extortion, Extortion Type 1, Type 2, and Type3.

Fourth,  the confusion matrix is obtained, where the algorithm's prediction is compared against a gold standard in gold_standard.xlsx with the real classification of the sentence (extortionate, non-extortionate).


Follow the REQUIREMENTS for installing a Miniconda or Conda environment with package dependencies.



