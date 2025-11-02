import streamlit as st
import numpy as np
import pickle
#from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Paste a news article or headline below to check if it is **Real** or **Fake**.")

# model = load_model("fake_news_model_converted.keras")

model = load_model("fake_news_model_new.keras", compile=False)
#model = load_model("fake_news_model_legacy.h5", compile=False)


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200

# User input
text_input = st.text_area("Enter news text here:")

if st.button("🔍 Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(padded)[0][0]
        label = "FAKE" if pred >= 0.5 else "REAL"
        st.write(f"### Prediction: **{label}** ({pred*100:.2f}% confidence)")
        




