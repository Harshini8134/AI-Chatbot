import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("questions_answers.csv")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_question(question):
    """Preprocesses the input question to match stored processed questions."""
    tokens = word_tokenize(question.lower())  # Tokenization
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return "-".join(tokens)  # Join with hyphen

# Apply preprocessing to dataset
df['Processed_Question'] = df['Question'].apply(preprocess_question)

# Save processed data
df[['Processed_Question', 'Answer']].to_csv("processed_questions_answers.csv", index=False)

print("Data processing completed. You can now chat!")

# Load processed data
df_processed = pd.read_csv("processed_questions_answers.csv")

# Chatbot function
def chatbot():
    print("Welcome to the chatbot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")  # Get user input
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        processed_input = preprocess_question(user_input)  # Preprocess input
        response = df_processed[df_processed['Processed_Question'] == processed_input]
        
        if not response.empty:
            print("Chatbot:", response.iloc[0]['Answer'])
        else:
            print("Chatbot: Sorry, I don't have an answer for that.")

# Run chatbot
chatbot()
