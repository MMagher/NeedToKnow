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

# --- Multi-Search Function ---
def search_bylaws(location):
    """Search for bylaws and regulations using multiple targeted queries."""
    
    # Define multiple search queries for different categories
    search_queries = [
        f"{location} municipal bylaws property maintenance grass cutting weeds",
        f"{location} noise bylaw construction hours quiet hours",
        f"{location} parking bylaws street parking overnight parking",
        f"{location} waste recycling garbage collection bylaws",
        f"{location} snow removal ice clearance winter maintenance bylaws",
        f"{location} pet bylaws dog cat licensing animal limits",
        f"{location} fence bylaws sight lines property boundaries",
        f"{location} parks festivals community events entertainment",
        f"{location} rental housing bylaws heating standards property standards",
        f"{location} boulevard maintenance sidewalk clearing bylaws",
    ]
    
    all_results = []
    search_context = ""
    total_sources = 0
    
    # Create a progress bar for multiple searches
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, query in enumerate(search_queries):
        status_text.text(f"Searching: {query[:60]}...")
        
        try:
            search_url = "https://api.tavily.com/search"
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 3,  # Fewer results per query to stay within limits
                "chunks_per_source": 2,
            }
            
            response = requests.post(search_url, json=payload)
            response.raise_for_status()
            results = response.json()
            
            # Add results to the combined context
            for result in results.get("results", []):
                # Avoid duplicate sources
                url = result.get('url', '')
                if url not in [r.get('url', '') for r in all_results]:
                    all_results.append(result)
                    search_context += f"Source: {url}\n"
                    search_context += f"Title: {result.get('title', '')}\n"
                    search_context += f"Content: {result.get('content', '')}\n\n"
                    total_sources += 1
            
            # Update progress
            progress_bar.progress((i + 1) / len(search_queries))
            
        except Exception as e:
            # Continue with other searches even if one fails
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Add a summary of what was found
    search_context = f"Total sources found: {total_sources}\n\n" + search_context
    
    return search_context

# --- Core Logic with Streaming ---
def ask_llm(query, search_results):
    """Sends the user's query and search results to the LLM with streaming."""
    try:
        # Create the streaming request
        stream = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that explains laws, bylaws, and regulations in an easy-to-understand but detailed way. 
                
You will be given search results from official municipal sources. Use these search results to answer the user's question. 
If the search results don't contain specific information about a bylaw, say "I couldn't find specific information about this in the search results" rather than making it up.

Format your response as follows:
- Location X requires your grass mowed and no higher than 4 centimeters tall
- Location X requires driveways, sidewalks, and approaches to your door to be snow and ice free

Make sure to cover ALL of these categories. If information is missing for a category, explicitly say so:
- Noise ordinances (construction hours, quiet hours, exemptions)
- Property maintenance (grass height, property standards, boulevard maintenance)
- Parking and vehicles (street parking, overnight restrictions)
- Waste and recycling (collection schedules, limits)
- Winter snow and ice requirements (sidewalk clearing, snow removal)
- Lawn requirements (grass height, weed control)
- Pet Limits and regulations (licensing, number limits)
- Fences and sight lines
- Rental housing standards (heating, maintenance)
- Any other common or uncommon concerns

Report the results in a table and make sure the bylaws are correct for the city. The table should be well organized and easy to follow.

Make sure to add details like construction, location, date and time, differences between authorities, who is responsible for what, limits on what you can/can't do, limits on what you can/can't own, limits on how much you can have,
limits in general, what the city vs resident is responsible for, exceptions to rules for residents, exceptions to rules for city, whether the law comes from the province, city bylaw, or the city law overrides provincial law etc. 

At the top, provide a link to the city's official bylaws page if found.

Make sure details are relevant to a resident of the city. Below the table, add a section about parks, festivals, local traditions and general local entertainment. Add a summary in point format at the end.

Cite your sources by including the URL of where you found the information. It is important that the source is linked to the URL it was found in.""",                {"role": "user", "content": f"""Question: {query}

Search Results:
{search_results}

Please answer the question using the search results above. Make sure to cover ALL categories. If you don't find information for a category, say "No information found in search results."""}
            ],
            temperature=0.3,
            stream=True,  # ← Enable streaming!
        )
        
        # Yield each chunk as it arrives
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"An error occurred: {e}"

# --- Streamlit User Interface ---
user_input1 = st.text_area("I Live In (City, Province)", height=40, placeholder="e.g., In Montreal Quebec ")

if st.button("Get Explanation", type="primary"):
    if user_input1.strip() == "":
        st.warning("Please enter a location.")
    else:
        # --- Use a status container for better feedback ---
        status = st.status("🔍 Getting information...", expanded=True)
        
        # Step 1: Search for relevant bylaws using multiple queries
        status.update(label="📡 Searching municipal databases (10 targeted searches)...", state="running")
        search_results = search_bylaws(user_input1)
        
        if "Search error" in search_results:
            status.update(label="❌ Search failed", state="error")
            st.error(search_results)
            st.stop()
        
        # Step 2: Show sources found
        # Extract the total count from the search results
        import re
        count_match = re.search(r'Total sources found: (\d+)', search_results)
        source_count = count_match.group(1) if count_match else "0"
        
        status.update(label=f"✅ Found {source_count} sources across all categories", state="running")
        with st.expander("📚 Sources Found"):
            st.markdown(search_results)
        
        # Step 3: Stream the response
        status.update(label="🤖 Analyzing regulations and generating response...", state="running")
        
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        
        # Stream the response
        response_stream = ask_llm(
            "I Live in " + user_input1 + ". No need for a fancy introduction, just get into the explanation. Report results in a table. Make sure to cover ALL categories.",
            search_results
        )
        
        # Collect the full response for display
        full_response = ""
        for chunk in response_stream:
            full_response += chunk
            # Update the placeholder with the accumulated response
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
        
        # Update status to complete
        status.update(label="✅ Complete!", state="complete")

# --- Optional: Add a note about streaming at the bottom ---
st.caption("Responses stream in real-time as they're generated.")