import streamlit as st
import os
import requests
from openai import OpenAI

# --- Page Configuration ---
st.set_page_config(page_title="What I need To Know", page_icon="📜")
st.title("What I need To Know")
st.markdown("Ask about the laws and differences between places (e.g., 'I moved to Austin from New York...').")

# --- Get API Keys from Streamlit Secrets ---
try:
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]  # Free search API
except KeyError as e:
    st.error(f"API key not found: {e}. Please set it in your Streamlit Cloud secrets.")
    st.stop()

# --- Initialize NVIDIA NIM Client ---
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# --- Search Function ---
def search_bylaws(location):
    """Search for bylaws and regulations for a specific location."""
    try:
        # Tavily search API (free tier: 1000 searches/month)
        search_url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": f"{location} municipal bylaws regulations property maintenance noise parking waste",
            "search_depth": "advanced",
            "max_results": 5
        }
        
        response = requests.post(search_url, json=payload)
        response.raise_for_status()
        results = response.json()
        
        # Extract and combine the search results
        search_context = ""
        for result in results.get("results", []):
            search_context += f"Source: {result.get('url')}\n"
            search_context += f"Title: {result.get('title')}\n"
            search_context += f"Content: {result.get('content')}\n\n"
        
        return search_context
    except Exception as e:
        return f"Search error: {e}"

# --- Core Logic ---
def ask_llm(query, search_results):
    """Sends the user's query and search results to the LLM."""
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that explains laws, bylaws, and regulations in an easy-to-understand but detailed way. 
                
You will be given search results from official municipal sources. Use these search results to answer the user's question. 
If the search results don't contain specific information about a bylaw, say "I couldn't find specific information about this in the search results" rather than making it up.

Format your response as follows:
- Location X requires your grass mowed and no higher than 4 centimeters tall
- Location X requires driveways, sidewalks, and approaches to your door to be snow and ice free

Make sure to cover:
- Noise ordinances
- Property maintenance
- Parking and vehicles
- Waste and recycling
- Seasonal requirements including snow, leaves, and any other bylaws covering winter, spring, summer and autumn.
- Pet laws
- Any other common or uncommon concerns

report the results in a table and make sure the bylaws are correct for the city, but they should be well organized the results should be easy to follow and read, but must be detailed and not missing any details. 
make sure to add details like construction, location, date and time, differences between authorities, who is responsible for what, limits on what you can/can't do, limits on what you can/can't own, limits on how much you can have,
limits in general, what the city vs resident is responsible for, exceptions to rules for residents, exceptions to rules for city, whether the law comes from the province, city bylaw, or the city law overrides provincial law etc. 
make sure to add numbers for limits, things like how tall/much grass/snow/leaves/etc you can have and for how long you can violate. make sure every section has the numbers necessary listed
Make sure details are relevant to a resident of the city. At the bottom of the table add a section about parks, festivals and general local entertainment. Add a summery at the end.

Cite your sources by including the URL of where you found the information. make sure to not use HTML and only use markdown"""},
                {"role": "user", "content": f"""Question: {query}

Search Results:
{search_results}

Please answer the question using the search results above."""}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# --- Streamlit User Interface ---
user_input1 = st.text_area("I Live In (City, Province)", height=40, placeholder="e.g., In Montreal Quebec ")

if st.button("Get Explanation", type="primary"):
    if user_input1.strip() == "":
        st.warning("Please enter a location.")
    else:
        with st.spinner("Searching for bylaws and regulations..."):
            # Step 1: Search for relevant bylaws
            search_results = search_bylaws(user_input1)
            
            if "Search error" in search_results:
                st.error(search_results)
                st.stop()
            
            # Step 2: Show sources found
            with st.expander("📚 Sources Found"):
                st.markdown(search_results)
            
            # Step 3: Get AI to process the search results
            with st.spinner("Analyzing the regulations..."):
                result = ask_llm(
                    "I Live in " + user_input1 + ". No need for a fancy introduction, just get into the explanation. Report results in a table.",
                    search_results
                )
                st.success("### Response:")
                st.markdown(result)