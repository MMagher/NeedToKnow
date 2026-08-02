import streamlit as st
import os
from openai import OpenAI

# --- Page Configuration ---
st.set_page_config(page_title="What I need To Know", page_icon="📜")
st.title("What I need To Know")
st.markdown("Ask about the laws and differences between places (e.g., 'I moved to Austin from New York...').")

# --- Get API Key from Streamlit Secrets ---
try:
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
except KeyError:
    st.error("NVIDIA API key not found. Please set it in your Streamlit Cloud secrets.")
    st.stop()

# --- Initialize NVIDIA NIM Client ---
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# --- Core Logic ---
def ask_llm(query):
    """Sends the user's query to the LLM and returns the response."""
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that explains laws, bylaws, and regulations in an easy-to-understand but detailed way. 
                
Format your response as follows:
- Location X requires your grass mowed and no higher than 4 centimeters tall
- Location X requires driveways, sidewalks, and approaches to your door to be snow and ice free

Make sure to cover:
- Noise ordinances
- Property maintenance
- Parking and vehicles
- Waste and recycling
- Seasonal requirements
- Pet laws
- Any other common or uncommon concerns

report the result in a table, with there being columns for what you are responsible for, the city is responsible for, exceptions, whether the law comes from the city or province/state. 
make sure to add details for details like construction, location, date and time, differences between authorities etc. 
Make sure details are relevant to a resident of the city. At the bottom of the table add a section about parks, festivals and general local entertainment"""},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# --- Streamlit User Interface ---
user_input1 = st.text_area("I Live In", height=40, placeholder="e.g., In Quebec")

if st.button("Get Explanation", type="primary"):
    if user_input1.strip() == "":
        st.warning("Please enter a location.")
    else:
        with st.spinner("Consulting the legal text..."):
            result = ask_llm("I Live in " + user_input1 + ". No need for a fancy introduction, just get into the explanation. Provide the result in a table.")
            st.success("### Response:")
            st.write(result)

# --- Example Questions ---
with st.expander("💡 Example Questions"):
    st.write("- I moved to Austin from New York, explain the key zoning and rental bylaws I need to know.")
    st.write("- What are the main differences in property taxes between Texas and California?")
    st.write("- How do noise ordinances compare between small towns and big cities?")