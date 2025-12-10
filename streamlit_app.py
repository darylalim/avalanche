# Import packages
from dotenv import load_dotenv
import anthropic
import os
import pandas as pd
import re
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic()

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path

# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to get sentiment using generative AI
# @st.cache_data
# def get_message(user_prompt, temperature):
#     message = client.messages.create(
#         model="claude-haiku-4-5",
#         max_tokens=100,
#         messages=[
#             {
#                 "role": "user",
#                 "content": user_prompt
#             }
#         ],
#         temperature=temperature
#     )
#     return message

st.title("Hello, GenAI!")
st.write("This is a data processing application powered by generative AI.")

# Add a text input box for the user prompt
# user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

# Add a slider for temperature
# temperature = st.slider(
#     "Model temperature:",
#     min_value=0.0,
#     max_value=1.0,
#     value=1.0,
#     step=0.01,
#     help="Amount of randomness injected into the response. Use temperature closer to 0.0 for analytical/multiple choice, and closer to 1.0 for creative and generative tasks."
#     )

# with st.spinner("AI is working..."):
#     message = get_message(user_prompt, temperature)
#     # print the response from Anthropic
#     st.write(message.content[0].text)

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    st.subheader("Sentiment Score by Product")
    grouped = st.session_state["df"].groupby(["PRODUCT"])["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)

