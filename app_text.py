import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI, GooglePalm
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
# Load OpenAI API token from environment variables
openai_api_token = os.getenv("OPENAI_API_TOKEN")
palm_api_token = os.getenv("GOOGLE_API_KEY")
openai = OpenAI(api_token=openai_api_token)
palm = GooglePalm(api_key=palm_api_token)



def main():
    st.title(" Data Visualization App using Text")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    
    # Select box to choose the model
    model_selection = st.selectbox("Select Model", ["OpenAI", "GooglePLAM"])
    
    if uploaded_file is not None:
        st.write("File Uploaded Successfully!")
        # Read the uploaded CSV file into a pandas dataframe
        df = pd.read_csv(uploaded_file)
        # Display the dataframe
        st.write(df)
        config = {"OpenAI": openai, "GooglePALM": palm}
        pandas_ai = SmartDataframe(df, config=config)
        
        # User prompt section
        prompt = st.text_input("Enter your visualization prompt:")
        
        # Button to generate visualization
        if st.button("Generate Visualization"):
            # Choose the model based on user selection
            if model_selection == "OpenAI":
                model = openai
            else:
                model = palm
            
            # Use selected model to generate visualization
            result=pandas_ai.chat(prompt)
           
            
            # Display generated visualization
            st.write("Generated Visualization/TextANswer Here")
            st.image(result)
            
if __name__ == "__main__":
    main()
