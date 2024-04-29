import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as  st
import spacy
from sklearn.naive_bayes import MultinomialNB
nlp=spacy.load("en_core_web_sm")

intents=json.load(open('intents (1).json'))
tag=[]
patterns=[]
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tag.append(intent['tag'])

vector=TfidfVectorizer()
pattern_scaled=vector.fit_transform(patterns)

bot=LogisticRegression()
bot.fit(pattern_scaled,tag)

def chatbot(input_message):
    input_message=vector.transform([(input_message)])
    predict_tag=bot.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag']==predict_tag:
            response=random.choice(intent['responses'])
            return response
#print(preprocess('about ragging'))


st.title("SJC BOT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"ChatBot: "+ chatbot(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
