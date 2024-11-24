import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('spam_ham_dataset.csv')

df['text'] = df['text'].apply(str.lower)

scam_keyword_list = [
    'congratulations', 'urgent', 'claim',
    'best price', 'cash', 'credit',
    'act now', 'cheap', 'free',
    'winner', 'click', 'special',
    'instant payout', 'dont hesitate', 'order',
    'exclusive'
]

scam_greeting_list = ['user', 'candidate', 'winner']

scam_extension_list = ['.exe', '.rtf', '.vbs', '.scr', '.rar', 'zip']

scam_symbol_list = ['*', '!', '$', '£', '#']


def spam_point_reducer(email_text):
    
    spam_points = 4

    if not keyword_checker(email_text):
        spam_points -= 1
    if not suspiscious_greeting_checker(email_text):
        spam_points -= 1
    if not malicious_file_type_checker(email_text):
        spam_points -= 1
    if not non_professional_punctuation_checker(email_text):
        spam_points -= 1

    return "spam" if spam_points >= 2 else "ham"

def keyword_checker(email_text):
    return any(keyword in email_text for keyword in scam_keyword_list)

def suspiscious_greeting_checker(email_text):
    return any(greeting in email_text for greeting in scam_greeting_list)

def malicious_file_type_checker(email_text):
    return any(extension in email_text for extension in scam_extension_list)

def non_professional_punctuation_checker(email_text):
    return any(symbol in email_text for symbol in scam_symbol_list)


df['predicted_label'] = df['text'].apply(spam_point_reducer)
print("Performance Metrics:")
accuracy = accuracy_score(df['label'], df['predicted_label'])
print(f"Accuracy = {accuracy:.2f}")
precision = precision_score(df['label'], df['predicted_label'], pos_label="spam")
print(f"Precision = {precision:.2f}")
recall = recall_score(df['label'], df['predicted_label'], pos_label="spam")
print(f"Recall = {recall:.2f}")
f1 = f1_score(df['label'], df['predicted_label'], pos_label="spam")
print(f"F1 Score = {f1:.2f}")
conf_matrix = confusion_matrix(df['label'], df['predicted_label'], labels=["spam", "ham"])
print("\nConfusion Matrix:")
print(f"[[TP, FP],\n [FN, TN]]\n{conf_matrix}")