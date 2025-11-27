import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ---------------------------
# PARAMETERS
# ---------------------------
VOCAB_SIZE = 10000
MAXLEN = 500
MODEL_FILENAME = "model_imdb_lstm.h5"

# ---------------------------
# 1. Load trained model
# ---------------------------
st.title("IMDB Movie Review Classifier by Anjum")
st.write("This app displays 5 sample IMDB test reviews with sentiment predictions (Positive/Negative).")

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_FILENAME)

model = load_lstm_model()
st.success("Model loaded successfully!")

# ---------------------------
# 2. Load IMDB dataset
# ---------------------------
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_test_padded = pad_sequences(x_test, maxlen=MAXLEN)
    return x_test, x_test_padded, y_test

x_test_raw, x_test, y_test = load_data()

# ---------------------------
# 3. Decode function
# ---------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "the"

def decode_review(seq):
    return " ".join([reverse_word_index.get(i, "?") for i in seq if i != 0])

# ---------------------------
# 4. Display 5 Reviews
# ---------------------------
st.header("Sample IMDB Reviews & Predictions")

if st.button("Show 5 Reviews"):
    for i in range(5):
        review = decode_review(x_test_raw[i])
        prob = model.predict(np.array([x_test[i]]), verbose=0)[0, 0]
        prediction = "Positive" if prob >= 0.5 else "Negative"
        actual = "Positive" if y_test[i] == 1 else "Negative"

        st.subheader(f"Review {i+1}")
        st.write(f"**Actual Sentiment:** {actual}")
        st.write(f"**Predicted Sentiment:** {prediction} (prob = {prob:.4f})")

        st.write("**Review Text (first 600 chars):**")
        st.write(review[:600] + "...")
        st.markdown("---")
