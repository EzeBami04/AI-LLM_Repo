import pandas as pd
from openai import OpenAI
#pip install openai langchain-core
import csv
import os
from dotenv import load_dotenv
import logging
import ast   
import json
from langchain_core.prompts import PromptTemplate

# ============================== Config =============================
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

groq_key = os.getenv("groq_api")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_key, 
)

keywords = ["artist", "actor", "influencer", "creator", "comedian", "blogger", "fitness", "coach", "model", "fashion", "fitness"]

platforms = [
    "https://www.instagram.com",
    "https://www.youtube.com",
    "https://www.twitter.com",
    "https://www.tiktok.com"
    ]

prompt_template = PromptTemplate(
    input_variables=["keywords", "platforms"],
    template="""
    You are a websearch assistant. 
    Take the following keywords: {keywords} 
    and generate a uniform search for users on the following platforms: {platforms}.

    Only return a **valid JSON array** of usernames in Nigeria whose usernames or bios 
    contain any of the keywords. 

    Important: Do not include explanations or extra text.
    Just return a JSON list like: ["user1", "user2", "user3", ...].

    Ensure at least 800 results per platform.
    """
    )

# ====================================================================== #
def engine(keywords, platforms):
    rendered_prompt = prompt_template.format(
        keywords=", ".join(keywords),
        platforms=", ".join(platforms)
        )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": rendered_prompt}],
        temperature=0
        )
    raw_output = response.choices[0].message.content.strip()

    try:
        # passing to Json
        usernames = json.loads(raw_output)
    except:
        # fallback
        usernames = ast.literal_eval(raw_output)

    return usernames

if __name__ == "__main__":
    usernames = engine(keywords, platforms)

    logging.info(f"Total usernames extracted: {len(usernames)}")

    # Save to CSV
    with open("usernames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for u in usernames:
            writer.writerow([u])

    # check
    print(usernames[:20])
