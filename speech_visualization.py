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
    print("Data Visualization App")
    
    # File upload section
    file_path = input("Enter the path of the CSV file: ")
    df = pd.read_csv(file_path)
    print("File Uploaded Successfully!")
    print("DataFrame:")
    print(df)
    
    # Select model
    model_selection = input("Select Model (OpenAI/GooglePALM): ").lower()
    
    config = {"OpenAI": openai, "GooglePALM": palm}
    pandas_ai = SmartDataframe(df, config=config)
    
    # Generate visualization
    prompt = input("Speak your visualization prompt- Press Enter: ")
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        prompt = r.recognize_google(audio)
        speak_text(prompt)
        print("You said:", prompt)
    
    if model_selection == "openai":
        model = openai
    else:
        model = palm
    
    # Use selected model to generate visualization
    pandas_ai.chat(prompt)
    speak_text("Visualization generated successfully")

if __name__ == "__main__":
    main()
