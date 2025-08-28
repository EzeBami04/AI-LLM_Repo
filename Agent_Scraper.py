from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import openai
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
import os

## Load Environment Variablie
load_dotenv()

## initialize environment 
openai_api_key= os.getenv("api_key")

# Define URLs of target websites
URLS = {
    "Hugging Face": "https://huggingface.co/models",
    "PromptBase": "https://promptbase.com/best-ai-prompts",
    "FlowGPT": "https://flowgpt.com/?tags=AI+Tools"
}

# Function to scrape Hugging Face models
def scrape_huggingface():
    response = requests.get(URLS["Hugging Face"])
    soup = BeautifulSoup(response.text, "html.parser")
    
    models = []
    for model in soup.find_all("div", class_="truncate"):
        title = model.text.strip()
        models.append(["Hugging Face", "Models", title])
    
    return models
# Function to scrape PromptBase prompts
def scrape_promptbase():
    response = requests.get(URLS["PromptBase"])
    soup = BeautifulSoup(response.text, "html.parser")
    
    prompts = []
    for prompt in soup.find_all("div", class_="tile-title"): 
        title = prompt.find("href").text if prompt.find("href") else "Unknown"
        prompts.append(["PromptBase", "Prompts", title])
    
    return prompts

# Function to scrape FlowGPT prompts
def scrape_flowgpt():
    response = requests.get(URLS["FlowGPT"])
    soup = BeautifulSoup(response.text, "html.parser")
    
    prompts = []
    for item in soup.find_all("div", class_="line-clamp-2"):
        title = item.find("h2").text if item.find("h2") else "Unknown"
        prompts.append(["FlowGPT", "Prompts", title])
    
    return prompts

# Collect all scraped data
def scrape_all():
    data = []
    data.extend(scrape_huggingface())
    data.extend(scrape_promptbase())
    data.extend(scrape_flowgpt())
    
    return data

# Define tools for LangChain agent
scrape_tool = Tool(
    name="Web Scraper",
    func=scrape_all,
    description="Scrape AI model and prompt data from Hugging Face, PromptBase, and FlowGPT."
)

# Initialize LangChain agent
openai.api_key=openai_api_key
prompt = f"Extract key insights from this agent data:{scrape_all}"
llm = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content":prompt}],
        max_tokens=300
    )

agent = initialize_agent(
    tools=[scrape_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Load existing CSV structure
file_path = "./data/categories_subcategories.csv"  # Ensure correct path
if os.path.exists(file_path):
    existing_df = pd.read_csv(file_path)
else:
    existing_df = pd.DataFrame(columns=["Category", "Subcategory", "Title"])
    
# Run the agent to scrape data
scraped_data = agent.run("Scrape AI model and prompt data")
df_new = pd.DataFrame(scraped_data, columns=["Category", "Subcategory", "Title"])

# Save updated data
output_file = "updated_categories_subcategories.csv"
df_new.to_csv(output_file, index=False)

print(f"Scraped data saved to {output_file}")



"""
README

AI Web Scraper

Overview
This project is an AI-powered web scraper that extracts AI model and prompt data from various sources, including Hugging Face, PromptBase, and FlowGPT. It utilizes the **LangChain** framework to automate data retrieval and stores the extracted data in a structured CSV format.

Libraries to Install
Before running the script, ensure you have the following dependencies installed:
'''bash
pip install langchain openai requests beautifulsoup4 pandas python-dotenv
'''

Features
- Aim at scraping AI model and prompt data from multiple sources.
- Uses LangChain's agent framework to automate data collection.
- Appends new data to an existing CSV file to maintain historical records.

Functions

Scrape_huggingface()
Scrapes AI model data from Hugging Face.
- Extracts model names from the website.
- Returns a list of models categorized under "Hugging Face".

Scrape_promptbase()
Scrapes AI prompts from PromptBase.
- Extracts prompt titles.
- Returns a list of prompts categorized under "PromptBase".

scrape_flowgpt()
Target scraping AI prompts from FlowGPT.
- Extracts available prompt titles.
- Returns a list of prompts categorized under "FlowGPT".

Scrape_all()
This function aggregates scraped data from all sources.
- Calls individual scraper functions.
- Returns a combined list of AI models and prompts.

initialize_agent()
- This function uses LangChain framework to create an AI agent for automated data retrieval.
- Integrates a scraping tool for execution.

`save_to_csv()`
- Loads existing data from a CSV file.
- Merges new scraped data with existing records.
- Saves the updated dataset as a CSV file.

---
Instructions for Use

1. Set Up API Key
   - Create a `.env` file in the root directory and add your OpenAI API key:
   ```ini
   api_key=your_openai_api_key
   '''

2. Run the script
   Execute the Python script to scrape data and save it:
   on your terinal on vs code run:
   ''''
   python Agent_scraper.py
   '''

3. Verify the output
   - The scraped data will be saved in `updated_categories_subcategories.csv`.
   - If an existing CSV file is found, new data will be appended.

Notes
- Ensure the target websites are accessible to avoid request errors.
- Update the `file_path` variable if your CSV file is located elsewhere.
- Modify the scraping functions as needed to adapt to website changes.



"""

