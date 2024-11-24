import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df['text'].apply(str.lower)
email_text = df['text']

scam_keyword_list = [
    'congratulations', 'urgent', 'claim', 
    'best price', 'cash', 'credit', 
    'act now', 'cheap', 'free',
    'winner', 'click', 'special',
    'instant payout', 'dont hesitate', 'order', 
    'exlusive'
]

scam_greeting_list = [
    'user', 'candidate', 'winner'
]

scam_extension_list = [
    '.exe', '.rtf', '.vbs', 
    '.scr', '.rar', 'zip'
]

scam_symbol_list = [
      '*', '!', '$', '£', '#'
]

if 'text' not in df.columns:
        raise ValueError("The CSV file must have a text column for email contents.")

def spam_keyword_identifier(email_text):
    for keyword in scam_keyword_list:
         if keyword in email_text:
              return "spam"
         return "ham"

def spam_greeting_identifier(email_text):
      for greeting in scam_greeting_list:
            if greeting in email_text:
                  return "spam"
            return "ham"
      
def spam_extension_identifier (email_text):
      for extention in scam_extension_list:
            if extention in email_text:
                  return "spam"
            return "ham"
      
def spam_symbol_identifier (email_text):
      for symbol in scam_symbol_list:
            if symbol in email_text:
                  return "spam"
            return "ham"


df['predicted_label'] = df['text'].apply(spam_keyword_identifier, spam_greeting_identifier)
print("Performance Metrics:")
accuracy = accuracy_score(df['label'], df['predicted_label'])
print("Accuracy = ", accuracy)
precision = precision_score(df['label'], df['predicted_label'], pos_label="spam")
print("Precision:", precision)
recall = recall_score(df['label'], df['predicted_label'], pos_label="spam")
print("Recall:", recall)
f1 = f1_score(df['label'], df['predicted_label'], pos_label="spam")
print("F1 Score:", f1)
conf_matrix = confusion_matrix(df['label'], df['predicted_label'], labels=["spam", "ham"])
print("\nConfusion Matrix:")
print(f"[[TP, FP],\n [FN, TN]]\n{conf_matrix}")