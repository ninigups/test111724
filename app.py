import streamlit as st
import openai
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load your API Key
my_secret_key = st.secrets["MyOpenAIKey"]
openai.api_key = my_secret_key

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.7, openai_api_key=my_secret_key)

# Function to get response from GPT-4o Mini
def get_gpt4o_mini_response(input_text, no_words, blog_style):
    try:
        # Construct the prompt
        prompt = f"Write a blog for a {blog_style} job profile on the topic '{input_text}'. Limit the content to approximately {no_words} words."
        
        # Make a call to OpenAI's GPT-4o Mini model
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract and return the response content
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit UI configuration
st.set_page_config(
    page_title="Travel Assistant",
    page_icon="🛫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
branch = st.sidebar.selectbox("Select a branch", ["Generate Blogs", "Pre-travel", "Post-travel"])

# Generate Blogs Branch
if branch == "Generate Blogs":
    st.header("Generate Blogs 🤖")
    
    # User inputs
    input_text = st.text_input("Enter the Blog Topic")
    col1, col2 = st.columns([5, 5])
    
    with col1:
        no_words = st.text_input("No of Words")
    with col2:
        blog_style = st.selectbox("Writing the blog for", ("Researchers", "Data Scientist", "Common People"), index=0)
    
    # Generate blog button
    submit = st.button("Generate")
    
    # Display the generated blog content
    if submit:
        blog_content = get_gpt4o_mini_response(input_text, no_words, blog_style)
        if blog_content:
            st.write(blog_content)

# Pre-travel Branch
elif branch == "Pre-travel":
    st.header("Pre-travel: Itinerary Generation")
    
    # User inputs
    destination = st.text_input("Enter Destination")
    duration = st.number_input("Enter Duration (in days)", min_value=1, max_value=30, value=5)
    interests = st.text_input("Enter your interests (comma-separated)")
    budget = st.selectbox("Select your budget level", ["Low", "Medium", "High"])
    travel_dates = st.date_input("Select your travel dates", [])
    
    # Generate itinerary button
    generate_itinerary = st.button("Generate Itinerary")
    
    if generate_itinerary:
        # Create a prompt template
        prompt_template = """You are a travel assistant. Create a {duration}-day itinerary for a trip to {destination}. The user is interested in {interests}. The budget level is {budget}. The travel dates are {travel_dates}. Provide a detailed plan for each day."""
        
        prompt = PromptTemplate(
            input_variables=["duration", "destination", "interests", "budget", "travel_dates"],
            template=prompt_template,
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Format travel dates
        if isinstance(travel_dates, list) and travel_dates:
            travel_dates_str = ', '.join([date.strftime("%Y-%m-%d") for date in travel_dates])
        elif not isinstance(travel_dates, list) and travel_dates:
            travel_dates_str = travel_dates.strftime("%Y-%m-%d")
        else:
            travel_dates_str = "Not specified"
        
        itinerary = chain.run({
            "duration": duration,
            "destination": destination,
            "interests": interests,
            "budget": budget,
            "travel_dates": travel_dates_str
        })
        
        st.subheader("Generated Itinerary:")
        st.write(itinerary)

# Post-travel Branch
elif branch == "Post-travel":
    st.header("Post-travel: Data Classification and Summary")
    
    # Allow user to upload an Excel file
    uploaded_file = st.file_uploader("Upload your travel data (Excel file)", type=["xlsx"])
    
    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Data Preview:")
        st.write(df.head())
        
        # Check if required columns exist
        if 'Description' in df.columns and 'Amount' in df.columns:
            # Add a 'Category' column to the DataFrame
            def classify_expense(description):
                prompt = f"Classify the following expense description into categories like 'Food', 'Transport', 'Accommodation', 'Entertainment', 'Miscellaneous':\n\n'{description}'\n\nCategory:"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=10,
                    n=1,
                    stop=None,
                    temperature=0
                )
                category = response.choices[0].text.strip()
                return category
            
            df['Category'] = df['Description'].apply(classify_expense)
            
            st.subheader("Classified Data:")
            st.write(df)
            
            # Generate summary
            total_spent = df['Amount'].sum()
            summary_prompt = f"Provide a quick summary of the travel expenses based on the following data:\n\n{df.to_string()}\n\nSummary:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=summary_prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5
            )
            summary = response.choices[0].text.strip()
            
            st.subheader("Summary:")
            st.write(summary)
            
            st.subheader(f"Total Spent: ${total_spent:.2f}")
        else:
            st.error("The uploaded Excel file must contain 'Description' and 'Amount' columns.")
