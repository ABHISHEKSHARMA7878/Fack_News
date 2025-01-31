import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Text preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def apply_stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def apply_lemmatization(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    text = apply_lemmatization(text)
    return text

# Load models
@st.cache_resource
def load_models():
    # Load TF-IDF
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    
    # Load all models
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'Neural Network': 'neural_network_model.pkl'
    }
    
    for model_name, file_path in model_files.items():
        with open(file_path, 'rb') as file:
            models[model_name] = pickle.load(file)
    
    return tfidf, models

# Main function
def main():
    st.title("Fake News Detection System")
    st.write("This application uses machine learning to detect fake news articles.")
    
    # Load models
    try:
        tfidf, models = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Select model
    model_name = st.selectbox(
        "Select Model",
        ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Decision Tree', 'Neural Network']
    )
    
    # Text input
    news_text = st.text_area("Enter news text", height=200)
    
    if st.button("Predict"):
        if news_text:
            try:
                # Preprocess text
                processed_text = preprocess_text(news_text)
                
                # Transform text using TF-IDF
                text_tfidf = tfidf.transform([processed_text])
                
                # Get prediction
                model = models[model_name]
                prediction = model.predict(text_tfidf)[0]
                probability = model.predict_proba(text_tfidf)[0]
                
                # Display results
                st.write("---")
                st.subheader("Prediction Results")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("Prediction: Real News")
                    else:
                        st.error("Prediction: Fake News")
                
                with col2:
                    confidence = probability[1] if prediction == 1 else probability[0]
                    st.info(f"Confidence: {confidence:.2%}")
                
                # Display probability distribution
                st.write("Probability Distribution:")
                prob_df = pd.DataFrame({
                    'Category': ['Fake News', 'Real News'],
                    'Probability': [probability[0], probability[1]]
                })
                st.bar_chart(prob_df.set_index('Category'))
                
                # Display processing details
                with st.expander("See processing details"):
                    st.write("Preprocessed text:", processed_text)
                    st.write("Model used:", model_name)
                    st.write("Raw probabilities:", {
                        'Fake News': f"{probability[0]:.4f}",
                        'Real News': f"{probability[1]:.4f}"
                    })
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Please enter some text to analyze")
    
    # Add information about the models
    with st.expander("About the Models"):
        st.write("""
        This application uses five different machine learning models:
        
        1. **Logistic Regression**: A linear model that works well with text classification
        2. **Naive Bayes**: Particularly effective for text classification tasks
        3. **Random Forest**: An ensemble method that combines multiple decision trees
        4. **Decision Tree**: A simple but interpretable model
        5. **Neural Network**: A deep learning approach using MLPClassifier
        
        The text is preprocessed using:
        - Text cleaning (removing special characters, lowercase conversion)
        - Stopword removal
        - Stemming
        - Lemmatization
        - TF-IDF vectorization
        """)

if __name__ == "__main__":
    main()
# Add some information about the app
st.write("---")
st.write("This app uses Logistic Regression, Naive Bayes, and Random Forest models to classify news articles as real or fake based on the text content.")
st.write("Deployment with💓 using streamlit")
