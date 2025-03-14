import numpy as np
import pandas as pd
import re
import nltk
import datefinder
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from number_parser import parse
from collections import Counter
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================

# PART 1: Add 7 new features to the Haunted Places dataset

# ==========================================

# --------------------------------------------------------------

# 1. Import and clean Haunted Places dataset

# --------------------------------------------------------------

# Import Haunted Places dataset
haunted = pd.read_csv("../Data/haunted_places.csv")

# Drop duplicate rows
haunted = haunted.drop_duplicates()

# fill missing longitude with city_longitude, fill missing latitude with city_latitude
haunted["longitude"] = haunted["longitude"].fillna(haunted["city_longitude"])
haunted["latitude"] = haunted["latitude"].fillna(haunted["city_latitude"])

# Fill missing city_longitude with longitude, fill missing city_latitude with latitude
haunted["city_longitude"] = haunted["city_longitude"].fillna(haunted["longitude"])
haunted["city_latitude"] = haunted["city_latitude"].fillna(haunted["latitude"])

# Drop missing values
haunted = haunted.dropna()

# --------------------------------------------------------------

# 2. Define function used for adding features "Audio Evidence" and "Image/Video/Visual Evidence"

# --------------------------------------------------------------

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

def contains_keywords(description, keywords):
    """
    General keyword matching function to check if the description contains specified keywords 
    (e.g., audio_keywords).
    
    Parameters:
    - description (str): The text to be checked.
    - keywords (list): The list of keywords.

    Returns:
    - bool: Returns True if the text contains any keyword, otherwise returns False.
    """

    if isinstance(description, str): 
        # 1. Extract all words
        words = re.findall(r"\b\w+\b", description.lower())
        
        # 2. Generate adjacent word combinations (bigrams)
        bigrams = [" ".join(pair) for pair in zip(words, words[1:])]
        words += bigrams 

        # 3. Lemmatization (Nouns + Verbs)
        lemmatized_words = [
            lemmatizer.lemmatize(word, pos="v") for word in words  
        ]
        lemmatized_words = [
            lemmatizer.lemmatize(word, pos="n") for word in lemmatized_words  
        ]

        # 4. Check if keywords are present
        return any(keyword in lemmatized_words for keyword in keywords)

    return False

# Add feature "Audio Evidence"
audio_keywords = [
    "noise", "sound", "whisper", "scream", "cry", "chant", "voice", "footstep", "laugh",
    "growl", "moan", "knock", "howl", "wail", "whimper", "echo", "bang", "shout", "yell",
    "murmur", "mumble", "hiss", "roar", "groan", "bark", "squeak", "thump", "hum", "rattle",
    "rustle", "buzz", "clap", "clatter", "click", "crackle", "creak", "gasp", "grunt", "gurgle",
    "huff", "pant", "shriek", "snarl", "snicker", "sniff", "snore", "snort", "sob", "squeal",
    "stomp", "thud", "wheeze", "whoop", "hear", "horn", "ring", "tick", "holler", "whistle",
    "laughter", "gibberish"
]

haunted["Audio Evidence"] = haunted["description"].apply(lambda x: contains_keywords(x, audio_keywords))

# Add feature "Image/Video/Visual Evidence"
visual_keywords = [
    # Camera, Photo, Video
    "image", "photo", "picture", "snapshot", "photograph", "frame", "shot", "footage", "recording",
    "record", "video", "clip", "film", "movie", "broadcast", "screen", "tape", "projector", "cctv", 
    "picture", "surveillance", "channel",
    
    # Writing and marking
    "writing", "write", "marking", "mark", "graffiti", "inscription", "etching", "etch", "message", 
    "scribble", "paint"
]

haunted["Image/Video/Visual Evidence"] = haunted["description"].apply(lambda x: contains_keywords(x, visual_keywords))

# --------------------------------------------------------------

# 3. Add feature "Haunted Places Date"

# --------------------------------------------------------------

# Defining global date thresholds
MAX_DATE = datetime(2025, 1, 1)

def is_valid_date(date_str):
    """Verify whether the date is within a valid range"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj <= MAX_DATE
    except:
        return False

def extract_date(description):
    text = str(description).lower()
    
    # Priority 1: Handle full date format
    iso_date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso_date_match:
        date_str = iso_date_match.group()
        if is_valid_date(date_str):
            return date_str

    # Priority 2: Dealing with the ten-year statement
    decade_match = re.search(
        r"\b(?:the\s+)?(19\d{2}|20\d{2})['’]s?\b", 
        text, 
        re.IGNORECASE
    )
    if decade_match:
        year = int(decade_match.group(1))
        if year <= MAX_DATE.year:
            return f"{year}-01-01"

    # Priority 3: Processing month + year combinations
    month_year_match = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)[\.,]?\s+(\d{4})\b",
        text,
        re.IGNORECASE
    )
    if month_year_match:
        year = int(month_year_match.group(2))
        if year <= MAX_DATE.year:
            month_str = month_year_match.group(1)[:3].title()
            try:
                month_num = datetime.strptime(month_str, "%b").month
                return f"{year}-{month_num:02d}-01"
            except:
                pass

    # Priority 4: Handle the boot year
    year_match = re.search(
        r"\b(?:in|during|around|year\s+of)\s+(\d{4})\b", 
        text, 
        re.IGNORECASE
    )
    if year_match:
        year = int(year_match.group(1))
        if year <= MAX_DATE.year:
            return f"{year}-01-01"

    # Extracting dates using lenient mode
    try:
        matches = list(datefinder.find_dates(
            text, 
            strict=False,
            base_date=datetime(1800, 1, 1)
        ))
        
        # Double filtering: year range and maximum date
        valid_matches = [
            dt for dt in matches 
            if 1700 <= dt.year <= MAX_DATE.year and dt <= MAX_DATE
        ]
        
        if valid_matches:
            return valid_matches[0].strftime("%Y-%m-%d")
    except:
        pass

    # Final fallback: pure four-digit year
    pure_year = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if pure_year:
        year = int(pure_year.group(1))
        if year <= MAX_DATE.year:
            return f"{year}-01-01"

    return "2025-01-01"

haunted["Haunted Places Date"] = haunted["description"].apply(extract_date)

# --------------------------------------------------------------

# 4. Add feature "Haunted Places Witness Count"

# --------------------------------------------------------------

# Keywords that must have a number in front to be considered valid
numeric_witness_keywords = ["witness"]

# Keywords that increase the witness count each time they appear
count_witness_keywords = ["witness", "see"]

lemmatizer = WordNetLemmatizer()

def parse_witness_count(description):
    if isinstance(description, str):
        # Parse text and convert numbers in the text into Arabic numerals
        parsed_text = parse(description)  
        
        # Extract all words and numbers
        words = re.findall(r"\b\w+\b", parsed_text.lower())  

        # Lemmatization
        lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]  
        lemmatized_words = [lemmatizer.lemmatize(word, pos="n") for word in lemmatized_words]  

        # 1. Check for "number + numeric_witness_keywords" pattern
        for i in range(len(lemmatized_words) - 1):
            if lemmatized_words[i].isdigit() and lemmatized_words[i+1] in numeric_witness_keywords:
                num = int(lemmatized_words[i])
                if num < 30:  # Filter out large numbers like years
                    return num  # If a valid "number + keyword" pattern is found, return the number

        # 2. Count occurrences of count_witness_keywords
        witness_count = sum(1 for word in lemmatized_words if word in count_witness_keywords)

        # If the text contains count_witness_keywords but no number, return the count of keyword occurrences
        if witness_count > 0:
            return witness_count  # Increase witness count by +1 for each keyword occurrence
    
    return 0  # If neither pattern is found, return 0

haunted["Haunted Places Witness Count"] = haunted["description"].apply(parse_witness_count)

# --------------------------------------------------------------

# 5. Add feature "Time of Day"

# --------------------------------------------------------------

def detect_time_of_day(text):
    text = str(text).lower()
    time_keywords = {
        "Evening": ["evening", "night", "midnight", "dark", 'nighttime', 'nocturnal', 'nightfall', 'sundown', 'darkness', 'late hours', 'overnight', 'witching hour', 'vesper', 'candlelight hours', 'shadowy', ],
        "Morning": ["morning", "dawn", "sunrise", 'daybreak', 'sunup', 'early hours', 'first light', 'crack of dawn', 'pre-dawn', 'break of day', 'aurora', 'cockcrow', 'predawn', 'dewy'],
        "Dusk": ["dusk", "sunset", "twilight", 'gloaming', 'evening twilight', 'dim light', 'shadows lengthen', 'crepuscular', 'fading light']
    }
    
    for time_period, keywords in time_keywords.items():
        if any(keyword in text for keyword in keywords):
            return time_period
    return "Unknown"

haunted["Time of Day"] = haunted["description"].apply(detect_time_of_day)

# --------------------------------------------------------------

# 6. Add feature "Apparition Type"

# --------------------------------------------------------------

category_mapping = {
    "ghosts": "Several Ghosts",
    "spirits": "Several Ghosts",
    "figures": "Several Ghosts",
    "hauntings": "Several Ghosts",
    "shadows of teenager": "Several Ghosts",
    
    "ghost": "Ghost",
    "spirit": "Ghost",
    "phantom": "Ghost",
    "specter": "Ghost",
    "apparition": "Ghost",
    "poltergeist": "Ghost",
    "ghostly": "Ghost",
    
    "orb": "Orb",
    "white ball": "Orb",
    "fireball": "Orb",

    "ufo": "UFO",
    "Unidentified Flying Object": "UFO",
    
    "uap": "UAP",
    "Unidentified Aerial Phenomena": "UAP",

    "male": "Male",
    "man": "Male",
    "he": "Male",
    
    "female": "Female",
    "woman": "Female",
    "lady": "Female",
    "she": "Female",

    "child": "Child",
    "kid": "Child",
    "boy": "Child",
    "girl": "Child",

    "teen": "Teenager",
    "teenager": "Teenager",
    "adolescent": "Teenager",

    "human": "Human"
}

lemmatizer = WordNetLemmatizer()

def detect_apparition_type(description):
    if isinstance(description, str):  
        words = re.findall(r"\b\w+\b", description.lower()) 
        bigrams = [" ".join(pair) for pair in zip(words, words[1:])] 
        words += bigrams  

        # 1. Directly match original words (without lemmatization)
        for keyword, category in category_mapping.items():
            if keyword in words:
                return category 
        
        # 2. If no match is found, perform lemmatization
        lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]
        lemmatized_words = [lemmatizer.lemmatize(word, pos="n") for word in lemmatized_words]

        for keyword, category in category_mapping.items():
            if keyword in lemmatized_words:
                return category 

    # 3.If still no match, return "Unknown"
    return "Unknown"

haunted["Apparition Type"] = haunted["description"].apply(detect_apparition_type)

# --------------------------------------------------------------

# 7. Add feature "Event type"

# --------------------------------------------------------------

def classify_event(text):
    text = str(text).lower()
    event_categories = {
        "Murder": ["murder", "killed", "stabbed", "homicide", 'slay', 'assassination', 'ritual killing'],
        "Death": ["died", "death", "suicide", "passed away", 'passed on', 'deceased', 'rest in peace', 'met their end'],
        "Supernatural": ["ghost", "spirit", "haunted", "apparition", "paranormal", 'phantom', 'poltergeist' ]
    }
    
    for event_type, keywords in event_categories.items():
        if any(keyword in text for keyword in keywords):
            return event_type
    return "Unknown"

haunted["Event type"] = haunted["description"].apply(classify_event)

# ==========================================

# PART 2: Add 3 features from the HTTP dataset to the Haunted Places dataset

# ==========================================

# --------------------------------------------------------------

# 1. Dataset 1: Happiness Score by State 2024

# --------------------------------------------------------------

response = requests.get("https://worldpopulationreview.com/state-rankings/happiest-states")
soup = BeautifulSoup(response.text, "html.parser")
soup.title

# find the table
table = soup.find("table")

# Fetch table head
headers = [th.text.strip() for th in table.find("thead").find_all("th") if th.text.strip()]

# Extract table data
data = []
rows = table.find("tbody").find_all("tr")
for row in rows:
    cols = row.find_all("td")
    if cols:
        values = [col.text.strip() for col in cols[1:]]
        data.append(values)

# Create DataFrame
happiness = pd.DataFrame(data, columns=headers)

# Rename columns
happiness.columns = ["state", "total happiness score", "emotional & physical well-being rank",
                     "community & environment rank", "work environment rank"]

# --------------------------------------------------------------

# 2. Dataset 2: Mental Health Statistics by State 2024

# --------------------------------------------------------------

response2 = requests.get("https://worldpopulationreview.com/state-rankings/mental-health-statistics-by-state")
soup2 = BeautifulSoup(response2.text, "html.parser")
soup2.title

# find the table
table2 = soup2.find("table")

# Fetch table head
headers2 = [th.text.strip() for th in table2.find("thead").find_all("th") if th.text.strip()]

# Extract table data
data2 = []
rows2 = table2.find("tbody").find_all("tr")
for row in rows2:
    cols = row.find_all("td")
    if cols:
        values = [col.text.strip() for col in cols[1:]]
        data2.append(values)

# Create DataFrame
mental_illness = pd.DataFrame(data2, columns=headers2)

# Rename columns
mental_illness.columns = ["state", "rates of mental illness", "adults with anxiety or depression",
                          "adults with severe mental illness", "overll mental health standing (youth & adults)"]

# --------------------------------------------------------------

# 3. Dataset 3: Suicide Rates by State 2024

# --------------------------------------------------------------

response3 = requests.get("https://worldpopulationreview.com/state-rankings/suicide-rates-by-state")
soup3 = BeautifulSoup(response3.text, "html.parser")
soup3.title

# find the table
table3 = soup3.find("table")

# Fetch table head
headers3 = [th.text.strip() for th in table3.find("thead").find_all("th") if th.text.strip()]

# Extract table data
data3 = []
rows3 = table3.find("tbody").find_all("tr")
for row in rows3:
    cols = row.find_all("td")
    if cols:
        values = [col.text.strip() for col in cols[1:]]
        data3.append(values)

# Create DataFrame
suicide = pd.DataFrame(data3, columns=headers3)

# Rename columns
suicide.columns = ["state", "suicide rate (per 100k)", "sucides"]

# Standardize suicide rate
suicide["suicide rate (per 100k)"] = suicide["suicide rate (per 100k)"].astype(float)
suicide["suicide rate (%)"] = (suicide["suicide rate (per 100k)"] / 1000).round(4)

# --------------------------------------------------------------

# 4. Merge datasets and extract 3 key features

# --------------------------------------------------------------

# Merge 3 datasets
happiness_mental = happiness.merge(mental_illness, on="state")
happiness_mental_suicide = happiness_mental.merge(suicide, on="state")

psychological_condition = happiness_mental_suicide[["state", "total happiness score","rates of mental illness",
                              "suicide rate (%)"]]

# Extract 3 key features
psychological_condition = psychological_condition.rename(columns={
    "total happiness score":"happiness score",
    "rates of mental illness":"mental illness rate (%)"})

psychological_condition["happiness score"] = psychological_condition["happiness score"].astype(float)
psychological_condition["mental illness rate (%)"] = psychological_condition["mental illness rate (%)"].str.rstrip("%").astype(float)

# --------------------------------------------------------------

# 5. Merge psychological_condition dataset with Haunted Places dataset

# --------------------------------------------------------------

# Add state "Washington DC"
washington_row = psychological_condition[psychological_condition["state"] == "Washington"].copy()

washington_row["state"] = "Washington DC"

psychological_condition = pd.concat([psychological_condition, washington_row], ignore_index=True)

# Merge dataframes
haunted_joined = haunted.merge(psychological_condition, on="state")

# ==========================================

# PART 3: Add 3 features from the Excel dataset to the Haunted Places dataset.

# ==========================================

# --------------------------------------------------------------

# 1. Import and clean the Excel dataset

# --------------------------------------------------------------
df = pd.read_excel("../Data/frpp_df.xlsx")

# 1. Filter all rows where the Legal Interest Code is 'G'
df_filtered = df[df['Legal Interest Code'] == 'G']
df_filtered = df_filtered[df_filtered['State Name'].notna()]

# 2. Filter rows where the Utilization column is not empty and convert the values
df_filtered = df_filtered[df_filtered['Utilization'].isin(['Utilized', 'Unutilized'])]
df_filtered['Utilization'] = df_filtered['Utilization'].map({'Utilized': 1, 'Unutilized': 0})

# 3. Filter rows where the Replacement Value column is not empty and clean the data
df_filtered = df_filtered[df_filtered['Replacement Value'].notna()]  # Filter rows where Replacement Value is not empty

# 4. Remove the first character and comma, convert to float
df_filtered['Replacement Value'] = df_filtered['Replacement Value'].apply(lambda x: float(str(x)[1:].replace(',', '')))

# 5. Eliminate rows where the value of Age of Property is "Year of Construction Cannot Be Determined"
df_filtered = df_filtered[df_filtered['Age of Property'] != "Year of Construction Cannot Be Determined"]
df_filtered = df_filtered[['State Name', 'Age of Property', 'Utilization', 'Replacement Value']]
df_filtered['Age of Property'] = pd.to_numeric(df_filtered['Age of Property'], errors='coerce')

# --------------------------------------------------------------

# 2. Process 3 features（Age of Property，Utilization，Replacement Value） and output

# --------------------------------------------------------------

# 1. Group by State Name and calculate the average
grouped_df = df_filtered.groupby('State Name').agg({
    'Age of Property': 'mean',
    'Utilization': 'mean',
    'Replacement Value': 'mean'
}).reset_index()

# 2. Rename the column names to add the "Ave_" prefix
grouped_df = grouped_df.rename(columns={
    'Age of Property': 'Ave_Age of Property',
    'Utilization': 'Ave_Utilization',
    'Replacement Value': 'Ave_Replacement Value'
})

# --------------------------------------------------------------

# 3. Merge dataframe

# --------------------------------------------------------------

# read Excel and CSV

# Remove leading and trailing spaces & unify capitalization
grouped_df["State Name"] = grouped_df["State Name"].str.strip().str.lower()
haunted_joined["state"] = haunted_joined["state"].str.strip().str.lower()

# Merge data
haunted_joined = haunted_joined.merge(grouped_df, left_on="state", right_on="State Name", how="left")

# Remove duplicate State Name columns
haunted_joined.drop(columns=["State Name"], inplace=True)

# Save the merged CSV
haunted_joined.to_csv("../Data/haunted_joined.csv", index=False, encoding="utf-8-sig")

# --------------------------------------------------------------

# 4. Visualization

# --------------------------------------------------------------

# Visualization 1: Complete state abbreviation mapping
state_abbrev = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT',
    'DELAWARE': 'DE', 'DISTRICT OF COLUMBIA': 'DC', 'FLORIDA': 'FL',
    'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL',
    'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS', 'KENTUCKY': 'KY',
    'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA',
    'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
    'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH',
    'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR',
    'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY'
}

# Add a state abbreviation column
grouped_df['State_Code'] = grouped_df['State Name'].str.upper().map(state_abbrev)

# Create a geographic heat map
fig = px.choropleth(
    grouped_df,
    locations='State_Code',
    locationmode="USA-states",
    color='Ave_Replacement Value',
    scope="usa",
    color_continuous_scale='Blues',
    title='Heat map of real estate replacement value distribution in each state of the United States',
    height=600  # Adjust the height
)

# Optimize display format
fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    coloraxis_colorbar={
        'title': 'Replacement value',
        'tickprefix': '$',
        'tickformat': ',.0f'
    }
)

fig.show()

# Visualization 2

# Set the graphic size
plt.figure(figsize=(12, 6))

# Draw a bar chart, colored by state
sns.barplot(x="State Name", y="Ave_Age of Property", hue="State Name", data=grouped_df, palette="viridis", legend=False)

# Rotate the X-axis label
plt.xticks(rotation=90)

# Add a title and tags
plt.title("Average age of buildings by state")
plt.xlabel("State Name")
plt.ylabel("Average Age")

# show
plt.show()

# Visualization 3

# Draw a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Ave_Utilization", y="Ave_Replacement Value", hue="State Name", palette="coolwarm", data=grouped_df)

# Add a title and tags
plt.title("Asset Replacement Value vs. Utilization")
plt.xlabel("Utilization")
plt.ylabel("Asset Replacement Value")

plt.show()

# Visualization 4
plt.figure(figsize=(8, 5))

# Plotting a Histogram
sns.histplot(grouped_df["Ave_Age of Property"], bins=20, kde=True, color="blue")

# Add a title
plt.title("Average age of buildings")
plt.xlabel("age")
plt.ylabel("quantity")

plt.show()

# ==========================================

# PART 4: Drop NaN values from Date column (run last)

# ==========================================
df_cleaned = haunted_joined.dropna(subset=['Haunted Places Date'])
df_cleaned = df_cleaned.rename(columns={'Haunted Places Date': 'Haunted Places D'})
df_cleaned.to_csv("cleaned_file.csv", index=False)
print("ok")