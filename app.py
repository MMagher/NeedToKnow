import streamlit as st
import os
import requests
import re
from openai import OpenAI

# --- Page Configuration ---
st.set_page_config(page_title="What I need To Know", page_icon="📜")
st.title("What I need To Know")
st.markdown("Ask about the laws and differences between places (e.g., 'I moved to Austin from New York...').")

# --- Get API Keys from Streamlit Secrets ---
try:
    NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"API key not found: {e}. Please set it in your Streamlit Cloud secrets.")
    st.stop()

# --- Initialize NVIDIA NIM Client ---
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# --- Multi-Search Function with Targeted Category ---
def search_bylaws(location, category=None):
    """Search for bylaws and regulations using multiple targeted queries.
    
    Args:
        location: The city/province to search for
        category: Optional specific category to focus on (e.g., "winter", "parking", "pets")
    """
    
    # If a specific category is provided, do targeted searches for that category
    if category:
        search_queries = [
            f"{location} {category} bylaws regulations municipal code",
            f"{location} {category} requirements rules ordinances",
            f"{location} {category} restrictions compliance enforcement",
            f"{location} {category} permits exemptions penalties",
            f"{location} {category} responsibilities residents property owners",
        ]
        category_label = f" (targeted: {category})"
    else:
        # Otherwise do the full comprehensive search
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
        category_label = ""
    
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
                "max_results": 4,  # Slightly more for targeted searches
                "chunks_per_source": 3,
            }
            
            response = requests.post(search_url, json=payload)
            response.raise_for_status()
            results = response.json()
            
            # Add results to the combined context
            for result in results.get("results", []):
                url = result.get('url', '')
                if url not in [r.get('url', '') for r in all_results]:
                    all_results.append(result)
                    search_context += f"Source: {url}\n"
                    search_context += f"Title: {result.get('title', '')}\n"
                    search_context += f"Content: {result.get('content', '')}\n\n"
                    total_sources += 1
            
            progress_bar.progress((i + 1) / len(search_queries))
            
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    search_context = f"Total sources found: {total_sources}{category_label}\n\n" + search_context
    
    return search_context

# --- Core Logic with Streaming ---
def ask_llm(query, search_results, category=None):
    """Sends the user's query and search results to the LLM with streaming."""
    try:
        # Build the system prompt based on whether we have a specific category
        if category:
            category_instruction = f"""
You are a helpful assistant that provides DETAILED information about {category} regulations, bylaws, and requirements for a specific location.

Focus your response EXCLUSIVELY on {category} regulations. Cover ALL aspects including:
- Specific requirements and rules
- Time limits and deadlines (e.g., how many hours/days to comply)
- Measurement limits (e.g., how high snow can be, how tall grass can be)
- Who is responsible (resident, landlord, city)
- Where the rules apply (property lines, sidewalks, boulevards)
- Exceptions and exemptions
- Penalties for non-compliance
- Permits required
- Seasonal/time-based requirements
- Differences between city and provincial rules

Make sure to include exact numbers, measurements, and timeframes whenever available.

Format your response with:
1. A brief introduction about {category} regulations in the location
2. A DETAILED table with columns: Category | Specific Rule | Details (including measurements, timeframes, responsible party) | Source
3. A summary of key points
4. Links to official sources

If you don't find specific information for {category} regulations, clearly state what is missing and suggest where the user might find it."""
        else:
            category_instruction = """
You are a helpful assistant that explains laws, bylaws, and regulations in an easy-to-understand but detailed way.

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

Cite your sources by including the URL of where you found the information. It is important that the source is linked to the URL it was found in."""

        stream = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=[
                {"role": "system", "content": category_instruction},
                {"role": "user", "content": f"""Question: {query}

Search Results:
{search_results}

Please answer the question using the search results above. Make sure to be detailed and specific. If you don't find information, say "No information found in search results."""}
            ],
            temperature=0.3,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"An error occurred: {e}"

# --- Streamlit User Interface ---
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 80px;
    }
    .help-text {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Two input fields: Location and optional category
col1, col2 = st.columns([2, 1])

with col1:
    user_input1 = st.text_area(
        "📍 I Live In",
        height=40,
        placeholder="e.g., Toronto, Ontario",
        help="Enter your city and province"
    )

with col2:
    user_input2 = st.text_input(
        "🎯 Focus on (optional)",
        placeholder="e.g., winter, parking, pets",
        help="Enter a specific category to focus on (leave blank for general overview)"
    )

if st.button("Get Explanation", type="primary", use_container_width=True):
    if user_input1.strip() == "":
        st.warning("Please enter a location.")
    else:
        # Determine if we have a specific category
        category = user_input2.strip().lower() if user_input2.strip() else None
        
        if category:
            status_text = f"🔍 Getting information about {category} in {user_input1}..."
        else:
            status_text = "🔍 Getting comprehensive information..."
        
        status = st.status(status_text, expanded=True)
        
        # Step 1: Search with targeted queries
        status.update(label=f"📡 Searching for {'{category} regulations' if category else 'all categories'}...", state="running")
        search_results = search_bylaws(user_input1, category=category)
        
        if "Search error" in search_results:
            status.update(label="❌ Search failed", state="error")
            st.error(search_results)
            st.stop()
        
        # Step 2: Show sources found
        count_match = re.search(r'Total sources found: (\d+)', search_results)
        source_count = count_match.group(1) if count_match else "0"
        
        status.update(label=f"✅ Found {source_count} sources", state="running")
        with st.expander("📚 Sources Found"):
            st.markdown(search_results)
        
        # Step 3: Stream the response without table jumping
        status.update(label="🤖 Analyzing and generating detailed response...", state="running")

        # Create a container with fixed dimensions
        response_container = st.container()
        with response_container:
            # Use a placeholder with fixed height
            response_placeholder = st.empty()

        # Build the query
        if category:
            query = f"I Live in {user_input1}. Provide DETAILED information specifically about {category} regulations. No introduction needed. Include specific rules with measurements and timeframes. Report results in a detailed table."
        else:
            query = f"I Live in {user_input1}. No need for a fancy introduction, just get into the explanation. Report results in a table. Make sure to cover ALL categories."

        response_stream = ask_llm(query, search_results, category=category)

        full_response = ""
        for chunk in response_stream:
            full_response += chunk
            # Wrap in a container with fixed width and overflow handling
            response_placeholder.markdown(
                f"""
                <div style="overflow-x: auto; width: 100%;">
                    {full_response}
                </div>
                """,
                unsafe_allow_html=True
            )

        status.update(label="✅ Complete!", state="complete")

st.caption("💡 Tip: Use the 'Focus on' field to get detailed information about a specific category like winter, parking, or pets.")