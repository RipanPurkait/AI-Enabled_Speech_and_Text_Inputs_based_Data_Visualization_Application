import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI, GooglePalm
import pandas as pd
import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3

load_dotenv()
# Load OpenAI API token from environment variables
openai_api_token = os.getenv("OPENAI_API_TOKEN")
palm_api_token = os.getenv("GOOGLE_API_KEY")
openai = OpenAI(api_token=openai_api_token)
palm = GooglePalm(api_key=palm_api_token)

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to speech
def speak_text(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def main():
    st.title("Data Visualization App by Speech: ")

    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Select box to choose the model
    model_selection = st.selectbox("Select Model", ["OpenAI", "GooglePALM"])

    if uploaded_file is not None:
        st.write("File Uploaded Successfully!")
        # Read the uploaded CSV file into a pandas dataframe
        df = pd.read_csv(uploaded_file)
        # Display the dataframe
        st.write(df)
        config = {"OpenAI": openai, "GooglePALM": palm}
        pandas_ai = SmartDataframe(df, config=config)

        # Button to generate visualization
        if st.button("Generate Visualization"):
            # Prompt user to speak their visualization prompt
            st.write("Speak your visualization prompt...")
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                prompt = r.recognize_google(audio)
                speak_text(prompt)
                st.write("You said:", prompt)

                # Choose the model based on user selection
                if model_selection == "OpenAI":
                    model = openai
                else:
                    model = palm

                # Use selected model to generate visualization
                pandas_ai.chat(prompt)
                speak_text("Visualization generated successfully")

if __name__ == "__main__":
    main()
