import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
with open("xgm.pkl", 'rb') as file:
    x_g = pickle.load(file)

with open("vec.pkl", 'rb') as file:
    vec = pickle.load(file)

# Text preprocessing functions
def preprocess_text(text):
    text = text.lower()  
    text = remove_special_characters(text)  
    text = remove_stopwords(text)  
    return text

def remove_special_characters(text):
    return ''.join(char for char in text if char.isalnum() or char.isspace())

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Streamlit app
st.title("Cyberbullying Detection")

# Input text from the user
user_input = st.text_area("Enter text to analyze:", "")

if st.button("Analyze"):
    if user_input:
        # Preprocess the input text
        preprocessed_new_text = [preprocess_text(user_input)]
        
        # Transform the preprocessed text to tf-idf vectors
        tfidf_vectors_new = vec.transform(preprocessed_new_text)
        
        # Predict using the loaded model
        predictions = x_g.predict(tfidf_vectors_new)
        
        # Display the prediction
        if predictions[0] == 1:
            st.write("Prediction: Gender-based Cyberbullying!")
        elif predictions[0] == 2:
            st.write("Prediction: Religion-based Cyberbullying!")
        elif predictions[0] == 4:
            st.write("Prediction: Age-based Cyberbullying!")
        elif predictions[0] == 5:
            st.write("Prediction: Ethnicity-based Cyberbullying!")
        else:
            st.write("Prediction: Not Cyberbullying")
    else:
        st.write("Please enter some text to analyze.")
