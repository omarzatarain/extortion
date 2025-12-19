
import spacy
from spacy.lang.es.examples import sentences 
import pandas as pd
import re
import unicodedata
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report,  accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def normalize(text):
    #Normalize text to lowercase, remove accents and special characters, except hyphens.
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

def build_diccionary(df):
   
   # Build a dictionary of unique, normalized words from the ‘sentences’ column of a DataFrame.
 
    dictionary = set()
    for sentence in df['sentences'].dropna():
        #Find words that contain letters, numbers and hyphens
        words = re.findall(r'\b[\w-]+\b', normalize(sentence))
        dictionary.update(words)
    return dictionary

def classify_word(word, demand_anchors, threat_anchors, nlp):
    #Classify a word as ‘demand’ or ‘threat’ using semantic similarity.
    
    if not isinstance(word, str):
        return None
    
    token = nlp(word.lower())
    
    # Exclude words that are not verbs or nouns, and single-letter words.
    if len(token) == 1 and token[0].pos_ not in ['VERB', 'NOUN'] and len(word) > 2:
        return None

    # Calculate the average similarity with anchor words
    try:
        demand_score = token.similarity(demand_anchors)
        threat_score = token.similarity(threat_anchors)
    except ValueError:
        # When the string has no sentence
        return None
        
    # Assign the word to the category with the highest score.
    if demand_score > threat_score:
        return 'Demand'
    elif threat_score > demand_score:
        return 'Threat'
    else:
        return 'Not defined'

# --- Classification Functions ---

def classify_sentence(sentence,demand_verbs, threat_verbs, nlp ):
    
    #Classify an extortion sentence into one of the three defined types.
    result = analyzer.predict(sentence)
    print(result.probas["NEG"])
    
    if not isinstance(sentence, str):
        return 'No Extortion'
    
    doc = nlp(sentence.lower())
    
    has_demand = False
    is_demand_imperative = False
    has_threat = False
    is_threat_passive = False
    
    # Demand Detection
    for token in doc:
        if token.lemma_ in demand_verbs or token.text in demand_verbs:
            has_demand = True
            if token.pos_ == "VERB" and "VerbForm=Fin" in token.morph and "Mood=Imp" in token.morph:
                is_demand_imperative = True
                
    # Threat Detection
    for token in doc:
        if token.lemma_ in threat_verbs:
            has_threat = True
            if any(child.dep_ == "nsubjpass" for child in token.children):
                is_threat_passive = True
                break
    
    # Classification Logic
    if not has_demand or not has_threat:
        return "No Extortion"

    if not is_demand_imperative and is_threat_passive:
        return "Type 01: demand in active form + threat in passive form"
    
    if is_demand_imperative and not is_threat_passive:
        return "Type 02: demand in imperative form + threat in active form"
    
    if not is_threat_passive and not is_demand_imperative:
        return "Type 03: threat in active form + demand in active form"
    
    return "Type unknown/blended"

if __name__ == "__main__":
   
   
   
    print(f" STARTING..")
   # Load the sentences to analyze 
    file_path = "datav2.xlsx"

    # INIT SECTION
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file was not found in the path: {file_path}")
        exit()

   
    # Create a list of words from the data without considering stopwords.
    diccionary_sentences = build_diccionary(df)
    
    # Create a DataFrame from the list and save it to an Excel file.
    df_diccionary = pd.DataFrame(sorted(list(diccionary_sentences)), columns=['Words'])
    output_path_diccionary = "list_words.xlsx"
    df_diccionary.to_excel(output_path_diccionary, index=False)
    print(f'The list has been generated and saved in: {output_path_diccionary}')

    # Load the model to find similar verbs for threats and demands 
    try:
       import spacy
       #os.system("python -m spacy download es_core_news_sm")
       nlp = spacy.load("es_core_news_sm")
    except OSError:
       print("The 'es_core_news_sm' model was not found. Downloading the 'lg' model....")
       os.system("python -m spacy download es_core_news_lg")
       nlp = spacy.load("es_core_news_lg")


# --- Anchor words for classification ---
# These words define the concepts of “demand” and “threat.”
demand_anchors = nlp("pagar depositar dinero billete cuota cupo colaboracion hacer deposito enviar cumplir")
threat_anchors = nlp("matar golpear lastimar herir sufrir dañar sangre raptar morir dinamitar perder") # daño


# Create the demand and threat columns.
df_clasified = pd.DataFrame(columns=['Demand Words', 'Threat Words'])
demand_list = []
threat_list = []

# Classify each word in the dictionary.
for word in df_diccionary['Words'].dropna().unique():
    category = classify_word(word, demand_anchors, threat_anchors, nlp)
    if category == 'Demand':
        demand_list.append(word)
    elif category == 'Threat':
        threat_list.append(word)

# Fill the new DataFrame with the sorted lists
max_len = max(len(demand_list), len(threat_list))
df_clasified['Demand Words'] = pd.Series(demand_list, dtype='object').reindex(range(max_len))
df_clasified['Threat Words'] = pd.Series(threat_list, dtype='object').reindex(range(max_len))

# Save the new classified dictionary in an Excel file.
output_file = "classified_dictionary.xlsx"
df_clasified.to_excel(output_file, index=False)

print("\nDictionary classification completed.")
print(f"The new dictionary has been saved in '{output_file}'.")

demand_verbs = set()
threat_verbs = set()

try:
    # Load the new classified dictionary
    df_diccionary = pd.read_excel(output_file)
    demand_verbs = set(df_diccionary['Demand Words'].dropna().str.lower().tolist())
    threat_verbs = set(df_diccionary['Threat Words'].dropna().str.lower().tolist())
    
    print("Verb dictionary successfully loaded.")
    print(f"Demand verbs: {len(demand_verbs)} found.")
    print(f"Threat verbs: {len(threat_verbs)} found.")

except FileNotFoundError:
    print(f"Error: The file '{output_file}' Not found. Make sure you have run the dictionary sorting script.")
    exit()
except KeyError:
    print("Error: The columns Demand Words  or Threat Words were not found in the classified dictionary.")
    exit()


# --- Process the main sentence file ---

sentences_file = "datav2.xlsx"

try:
    df = pd.read_excel(sentences_file, sheet_name='sentences')
    df['sentences'] = df['sentences'].astype(str)
    
except FileNotFoundError:
    print(f"Error: The sentence archive '{sentences_file}' Not found. Make sure it is in the same folder as the script.")
    exit()
except KeyError:
    print("Error: The spreadsheet 'e' or the column 'sentences' was not found in 'datav2.xlsx'.")
    exit()

# Create a new column for sorting

df['extortion_category'] = "NULL" 

data = pd.read_excel("gold_standard.xlsx", sheet_name="MATRIZ")

#Collect results for making the comparison
i = 0

   

    
for sentence in df['sentences']:
    x = classify_sentence(df['sentences'].iloc[i],demand_verbs, threat_verbs, nlp)
    print(x)
    print(sentence)
    df.iloc[i] ={'sentences': sentence, 'extortion_category': x}
    realv = data['REAL'].iloc[i]
    if x == 'No Extortion':
        data.iloc[i] ={'SENTENCES': sentence, 'REAL': realv, 'PREDICTED':'NO EXTORTION'}
    else:
        data.iloc[i] ={'SENTENCES': sentence, 'REAL': realv, 'PREDICTED':'EXTORTION'}
    i = i +1
print(df)

# Save the results in a new XLSX file
output_file = "classified_sentences.xlsx"
df.to_excel(output_file, index=False)
data_output_file = "confusion_matrix.xlsx" 
data.to_excel(data_output_file, index=False) 

print("\nAnalysis completed.")
print(f"The results have been saved in '{output_file}' in Excel format.")

data = pd.read_excel("confusion_matrix.xlsx", sheet_name="Sheet1")
cm=confusion_matrix(data['PREDICTED'], data['REAL'])
print(cm)

###Indicate the positive class, i.e., 1 is EXTORTION and 0 is NOT EXTORTION.
print ('Accuracy:', accuracy_score(data['REAL'], data['PREDICTED']))
print ('F1 score:', f1_score(data['REAL'], data['PREDICTED'], pos_label="EXTORTION"))
print ('Recall:', recall_score(data['REAL'], data['PREDICTED'], pos_label="EXTORTION"))
print ('Precision:', precision_score(data['REAL'], data['PREDICTED'], pos_label="EXTORTION"))
print ('\nClassification report:\n', classification_report(data['REAL'], data['PREDICTED']))


# Plot the confusion matrix
display_labels = ["EXTORTION","NO EXTORTION"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig("confusion.png", dpi=300, bbox_inches="tight")
plt.show()

