#!/usr/bin/env python3
"""
Unified scraper:
- Uses Google dorking to find instagram profiles (no IG login).
- Visits IG profile pages to estimate follower counts (meta description fallback).
- Seeds YouTube/X/TikTok lookups from found IG usernames.
- Returns dict {platform: [usernames]} and saves usernames.json.
"""

from playwright.sync_api import sync_playwright
import re
import json
import time
import random
import urllib.request
import ssl
import logging
import os
import time, random
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import  PromptTemplate
from langchain_groq import ChatGroq

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor

# =================================== Config =======================================
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
KEYWORDS = ["comedian", "artist", "actor", "influencer", "content", "creator", "blogger"]
MIN_IG_FOLLOWERS = 50_000
MAX_GOOGLE_PAGES = 15    
SLEEP_BETWEEN_REQUESTS = (1.5, 4.5)  # random delay between requests (seconds)

# Proxy config (keeps your existing approach)
class Proxy:
    def __init__(self, server, username, password):
        self.server = server
        self.username = username
        self.password = password

    def get_proxy_info(self):
        url = "https://geo.brdtest.com/mygeo.json"
        proxy_str = f"http://{self.username}:{self.password}@{self.server}"
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'https': proxy_str, 'http': proxy_str}),
            urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
        )
        try:
            with opener.open(url, timeout=15) as resp:
                return {"server": self.server, "username": self.username, "password": self.password, "geo": json.load(resp)}
        except Exception as e:
            logging.warning(f"Could not fetch proxy info: {e}")
            return None

native_proxy = Proxy(
    server= os.getenv("serer"),
    username= os.getenv("username"),
    password= os.getenv("password")
)
PROXY_INFO = native_proxy.get_proxy_info()

# ---------------- Helpers ----------------
def parse_count(text: str) -> int:
    """Parse readable follower strings like '12.3K', '1,234', '1.2M' into integer."""
    if not text:
        return 0
    s = text.strip().upper().replace(",", "")
    # remove any words around numbers: e.g. "12.5K followers" -> "12.5K"
    m = re.search(r"([0-9,.]+(?:\.[0-9]+)?\s*[KMB]?)", s)
    if not m:
        # try to extract digits
        m = re.search(r"(\d+)", s)
        if not m:
            return 0
    token = m.group(1).strip()
    multiplier = 1
    if token.endswith("K"):
        multiplier = 1_000
        token = token[:-1]
    elif token.endswith("M"):
        multiplier = 1_000_000
        token = token[:-1]
    elif token.endswith("B"):
        multiplier = 1_000_000_000
        token = token[:-1]
    try:
        val = float(token)
        return int(val * multiplier)
    except Exception:
        try:
            return int(token)
        except Exception:
            return 0

def rand_sleep():
    time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))

# ---------------- Playwright page helper ----------------
def open_page_with_context(p, url):
    """
    Launch a persistent context and return (page, context).
    Caller MUST close context when done: context.close()
    """
    args = ["--disable-blink-features=AutomationControlled",
            "--ignore-certificate-errors",
            "--no-sandbox",
            "--disable-setuid-sandbox"
            ]
    launch_kwargs = dict(headless=True, args=args)
    if PROXY_INFO:
        launch_kwargs["proxy"] = {"server": PROXY_INFO["server"],
                                  "username": PROXY_INFO["username"],
                                  "password": PROXY_INFO["password"]}
    # Use launch_persistent_context to keep cookies if desired
    # Set a local user_data_dir to reuse if you want; keep ephemeral to avoid disk clutter here.
    context = p.chromium.launch_persistent_context(user_data_dir=None, **launch_kwargs)
    page = context.new_page()
    page.set_default_timeout(45000)
    page.goto(url, timeout=60000)
    # let the page settle
    page.wait_for_timeout(2000)
    return page, context

# ---------------- Scrapers ----------------
def google_instagram_search(page, keyword, page_index=0):
    """
    Return a list of hrefs found in Google search results for site:instagram.com {keyword}.
    page_index: zero-based (0 => start=0)
    """
    q = f"site:instagram.com {keyword} Nigeria"
    start = page_index * 10
    url = f"https://www.google.com/search?q={urllib_request_quote(q)}&start={start}"
    logging.info(f"Google search URL: {url}")
    page.goto(url, timeout=60000)
    page.wait_for_timeout(1500)
    # Select anchors under #rso that link to instagram.com
    try:
        anchors = page.locator('#rso a[href*="instagram.com"]').evaluate_all("els => els.map(e => e.href)")
        return [a for a in anchors if a and "instagram.com" in a]
    except Exception as e:
        logging.debug(f"No anchors or evaluate_all failed: {e}")
        return []

def urllib_request_quote(s: str) -> str:
    # safe URL encode s
    from urllib.parse import quote_plus
    return quote_plus(s)

def extract_ig_followers_from_meta(page) -> int:
    """Try to extract follower count from meta[name='description'] or meta[property='og:description']"""
    try:
        # Try meta[name="description"]
        meta = page.locator('meta[name="description"]').get_attribute("content")
        if meta:
            f = parse_count(meta)
            if f:
                return f
        # Try og:description
        meta = page.locator('meta[property="og:description"]').get_attribute("content")
        if meta:
            f = parse_count(meta)
            if f:
                return f
    except Exception:
        pass
    return 0

def extract_ig_followers_from_dom(page) -> int:
    """Fallback: try to read the followers count via the typical DOM pattern (li:nth-child(2))."""
    try:
        locator = page.locator("section header section ul li:nth-child(2) a")
        if locator.count() > 0:
            txt = locator.first.inner_text(timeout=3000)
            return parse_count(txt)
    except Exception:
        pass
    
    try:
        alt = page.locator("header ul li").nth(1).inner_text(timeout=3000)
        return parse_count(alt)
    except Exception:
        return 0

def scrape_instagram_usernames(playwright, keywords):
    """Main IG discovery: Google search -> visit each profile -> parse followers -> collect usernames >= threshold"""
    found_usernames = []
    p = playwright
    # fall back to reuse page and  browser context for Google searches and profile visits to reduce cost
    page, context = open_page_with_context(p, "https://www.google.com")
    try:
        for kw in keywords:
            for page_idx in range(MAX_GOOGLE_PAGES):
                try:
                    rand_sleep()
                    hrefs = google_instagram_search(page, kw, page_index=page_idx)
                    logging.info(f"Found {len(hrefs)} instagram links for '{kw}' page {page_idx+1}")
                    for href in hrefs:
                        # skip post URLs
                        if "/p/" in href or "/reel/" in href or "/tv/" in href:
                            continue
                        # normalize profile url
                        # google often wraps redirect URLs — attempt to extract direct IG url
                        m = re.search(r"(https?://(www\.)?instagram\.com/[^&?/]+)", href)
                        if m:
                            profile_url = m.group(1)
                        else:
                            profile_url = href
                        try:
                            rand_sleep()
                            page.goto(profile_url, timeout=60000)
                            page.wait_for_timeout(1800)
                            # first try meta description to avoid brittle DOM
                            followers = extract_ig_followers_from_meta(page)
                            if not followers:
                                followers = extract_ig_followers_from_dom(page)
                            if followers >= MIN_IG_FOLLOWERS:
                                username = profile_url.rstrip("/").split("/")[-1]
                                if username and username not in found_usernames:
                                    logging.info(f"IG: {username} ({followers} followers)")
                                    found_usernames.append(username)
                        except Exception as e:
                            logging.debug(f"Failed to visit profile {profile_url}: {e}")
                except Exception as e:
                    logging.warning(f"Google search failed for keyword {kw} page {page_idx+1}: {e}")
        return list(found_usernames)
    finally:
        try:
            context.close()
        except Exception:
            pass

def scrape_youtube_from_seeds(playwright, seeds):
    """Search YouTube for each seed username and try to extract channel handles/ids from results."""
    found = []
    p = playwright
    for seed in seeds:
        try:
            rand_sleep()
            page, context = open_page_with_context(p, f"https://www.youtube.com/results?search_query={urllib_request_quote(seed)}")
            try:
                page.wait_for_timeout(2000)
                # Collect links that likely point to channels
                anchors = page.locator('a[href*="/channel/"], a[href*="/c/"], a[href*="/user/"]').evaluate_all("els => els.map(e => e.href)")
                for a in anchors:
                    
                    if a and "youtube.com" in a:
                        identifier = a.rstrip("/").split("/")[-1]
                        if identifier and identifier not in found:
                            found.append(identifier)
            except Exception as e:
                logging.debug(f"YouTube error for {seed}: {e}")
            finally:
                context.close()
        except Exception as e:
            logging.debug(f"YouTube launch error for {seed}: {e}")
    return list(found)

def scrape_x_from_seeds(playwright, seeds):
    """Search X (twitter) for each seed; collect handles found on result pages (best-effort)."""
    found = []
    p = playwright
    for seed in seeds:
        try:
            rand_sleep()
            search_url = f"https://x.com/search?q={urllib_request_quote(seed)}&src=typed_query&f=user"
            page, context = open_page_with_context(p, search_url)
            try:
                page.wait_for_timeout(2500)
                # Grab @handle-like strings
                spans = page.locator("span.css-175oi2r.r-1wbh5a2.r-dnmrzs.r-1ny4l3l span")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future = executor.submit(spans.all_inner_texts)
                    handles = future.result()
                found.extend(handles)
            except Exception as e:
                logging.debug(f"X scrape error for {seed}: {e}")
            finally:
                context.close()
        except Exception as e:
            logging.debug(f"X launch error for {seed}: {e}")
    return list(found)

def scrape_tiktok_from_seeds(playwright, seeds):
    """Search TikTok for each seed (best-effort) and collect profile handles."""
    found = []
    p = playwright
    start_url = "https://www.tiktok.com"
    page, context = open_page_with_context(p, f"{start_url}")

    time.sleep(random.randint(5,10))
    logging.info(f"succesful in loading {len(page)}")
    context.close()

    for seed in seeds:
        try:
            rand_sleep()
            page, context = open_page_with_context(p, f"https://www.tiktok.com/search?q={urllib_request_quote(seed)}")
            try:
                
                page.wait_for_timeout(3000)
                anchors = page.locator("div[data-e2e='search-user-info-container'] a[href*='@']")
                with ThreadPoolExecutor(max_workers=2) as ex:
                    future = ex.submit(anchors.all_inner_texts)
                handles = future.result()
                for h in handles:
                    if h and h not in found:
                        found.append(h)
            except Exception as e:
                logging.debug(f"TikTok scrape error for {seed}: {e}")
            finally:
                context.close()
        except Exception as e:
            logging.debug(f"TikTok launch error for {seed}: {e}")
    return list(found)

# ---------------- Orchestration ----------------



def extract_usernames_all(**args):
    """Top-level: runs IG discovery (via Google) and then looks up other platforms from IG seeds."""
    with sync_playwright() as p:
        # ======================Instagram discovery ============================
        ig_users = scrape_instagram_usernames(p, KEYWORDS)
        logging.info(f"Discovered {len(ig_users)} Instagram usernames (>= {MIN_IG_FOLLOWERS})")

        seed = ig_users[:200] 
        seeds = [i.replace("_", "").lower() for i in seed]
        
        # ---------------- Internal helper for parallel scraping ----------------
        def scrape_platform_in_parallel(scrape_func, seeds, max_workers=2):
            found = []

            def fetch(seed):
                try:
                    return scrape_func(p, [seed])  
                except Exception as e:
                    logging.debug(f"{scrape_func.__name__} error for {seed}: {e}")
                    return []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch, seed): seed for seed in seeds}
                for future in as_completed(futures):
                    found.extend(future.result())
            return list(dict.fromkeys(found))  # dedupe

        # ---------------- Parallel scraping ----------------
        yt_users = scrape_platform_in_parallel(scrape_youtube_from_seeds, seeds)
        logging.info(f"Discovered {len(yt_users)} YouTube identifiers from seeds")

        x_users = scrape_platform_in_parallel(scrape_x_from_seeds, seeds)
        logging.info(f"Discovered {len(x_users)} X handles from seeds")

        tt_users = scrape_platform_in_parallel(scrape_tiktok_from_seeds, seeds)
        logging.info(f"Discovered {len(tt_users)} TikTok handles from seeds")

    results = {
        "instagram": list(dict.fromkeys(ig_users)), 
        "youtube": list(dict.fromkeys(yt_users)),
        "x": list(dict.fromkeys(x_users)),
        "tiktok": list(dict.fromkeys(tt_users)),
    }
    return results

def format_for_api(results: dict):
    formatted = []
    for platform, usernames in results.items():
        for u in usernames:
            formatted.append({"username": u, "platform": platform})
    return formatted



def engine():
    logging.info("---Starting Engine---")
    
   

    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

    # Define the tool
    tools = [
        Tool(
            name="web_social_media_crawler",
            func=lambda _: extract_usernames_all(), 
            description="Extracts social media usernames from various social media platforms using provided CSS selectors to get usernames and follower counts."
        )
    ]

    # ========= Define the customized prompt ================
    custom_prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template="""
        You are a helpful web research assistant that helps people find social media usernames.

        You can use the following tools:
        {tools}

        Available tool names: {tool_names}
        Instructions:
        The tool provide you with the specific Css selectors for each platform use those selectors to search for only users in Nigeria from instagram, Youtube, X and tiktok

        When responding:
        1. Use the web_social_media_crawler tool to extract usernames from Instagram, YouTube, TikTok, and X in N.
        2. Don't give explanations, just provide usernames.
        3. If no usernames are found, output: "No usernames found".
        4. If usernames are found, output them as a JSON array of objects: 
        [{{"user1": platform": "username"}}, {{"user2", "platform": "username"}}].

        Question: {input}

        {agent_scratchpad}
        """
        )


    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=custom_prompt
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, 
                                                        handle_parsing_errors=True,
                                                        max_iterations=50,
                                                        max_execution_time=None,
                                                        early_stopping_method="generate")

    return agent_executor

def run_engine_and_save():
    executor = engine()
    result = executor.invoke({"input": "Find social medial influencers  in Nigeria and West Africa"})
    return result

import csv
# ---------------- Main ----------------
# ---------------- Main ----------------
if __name__ == "__main__":
    raw_dict = extract_usernames_all()

    with open("usernames.json", "w", encoding="utf-8") as f:
        json.dump(raw_dict, f, indent=2)

    platforms = list(raw_dict.keys())
    max_len = max(len(usernames) for usernames in raw_dict.values())

    rows = []
    for i in range(max_len):
        row = {}
        for platform in platforms:
            row[platform] = raw_dict[platform][i] if i < len(raw_dict[platform]) else ""
        rows.append(row)

    with open("usernames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=platforms)
        writer.writeheader()
        writer.writerows(rows)
