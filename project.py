import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import os

# Step 1: Load and display the dataset
def load_uploaded_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Step 2: Preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    return text

# Step 3: Prepare data for training
def prepare_data(df):
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Encode target labels into integers
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['airline_sentiment'])
    
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=50, padding='post')
    
    return tokenizer, padded_sequences, np.array(df['sentiment_encoded']), label_encoder

# Step 4: Build the LSTM model
def build_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=32),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')  # 3 output classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def train_or_load_model(padded_sequences, labels):
    model_file = 'multi_class_sentiment_model.h5'
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = build_model()
        model.fit(padded_sequences, labels, epochs=3, batch_size=64, validation_split=0.2)
        model.save(model_file)
    return model

# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Classify tweets into sentiments (e.g., positive, negative, neutral).")
    
    # File upload section
    st.subheader("Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file with 'text' and 'sentiment' columns.", type=["csv"])
    
    if uploaded_file is not None:
        df = load_uploaded_dataset(uploaded_file)
        st.write("Dataset Sample:")
        st.write(df.head())
        
        # Step 2: Preprocess and Prepare Data
        st.subheader("Step 2: Preprocess Data")
        if 'text' in df.columns and 'airline_sentiment' in df.columns:
            tokenizer, padded_sequences, labels, label_encoder = prepare_data(df)
            st.write("Sample cleaned text:", df['cleaned_text'].iloc[0])
            
            # Step 3: Train or Load the Model
            st.subheader("Step 3: Train Model")
            if st.button("Train Model") or os.path.exists('multi_class_sentiment_model.h5'):
                with st.spinner("Training/loading the model..."):
                    model = train_or_load_model(padded_sequences, labels)
                    st.success("Model is ready!")
                
                # Step 4: Sentiment Prediction
                st.subheader("Step 4: Test the Model")
                user_input = st.text_input("Enter a tweet:")
                if user_input:
                    cleaned_input = preprocess_text(user_input)
                    input_sequence = tokenizer.texts_to_sequences([cleaned_input])
                    padded_input = pad_sequences(input_sequence, maxlen=50, padding='post')
                    prediction = model.predict(padded_input)[0]
                    predicted_class = np.argmax(prediction)
                    sentiment = label_encoder.inverse_transform([predicted_class])[0]
                    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {prediction[predicted_class]:.2f})")
        else:
            st.error("Dataset must have 'text' and 'sentiment' columns!")

if __name__ == "__main__":
    main()