"""
proxies.py — Fast algorithmic proxies for PCA/LDA linguistic features.
"""

import re
import string

COLUMNS = [
    "Proxy_Open_Closed_Ratio",
    "Proxy_Density",
    "Proxy_Adverb_Count",
    "Proxy_Task_Nouns_Count",
    "Proxy_OOV_Error_Count"
]

# A fast-lookup set of common English closed-class words 
# (pronouns, prepositions, conjunctions, determiners, auxiliary verbs)
CLOSED_CLASS_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "few", "all", "any", "some", "every", "no", "not",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "shall", 
    "should", "will", "would", "may", "might", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "in", "on", "at", "to", "from", "by", "with", "about", "against", "between", 
    "into", "through", "during", "before", "after", "above", "below", "up", "down",
    "of", "for", "without", "under", "over", "around", "here", "there", "where"
}

# Target nouns specific to the AphasiaBank elicitation tasks
TASK_NOUNS = {
    # Cinderella narrative
    "cinderella", "stepmother", "sister", "sisters", "stepsister", "stepsisters", 
    "prince", "king", "queen", "fairy", "godmother", "mouse", "mice", 
    "ball", "dance", "shoe", "slipper", "glass", "dress", "gown", "pumpkin", "carriage", "midnight",
    # PB&J Sandwich narrative
    "peanut", "butter", "jelly", "jam", "bread", "knife", "plate", "jar", "slice", "crust"
}

def compute(text):
    text_lower = text.lower()
    
    # 1. Adverb Proxy: Count words ending in "-ly"
    adverb_count = len(re.findall(r'\b\w+ly\b', text_lower))
    
    # 2. Error/OOV Proxy: Count stutters (c-c-cinderella) and trailing fragments (the boy was run-)
    stutter_count = len(re.findall(r'\b(\w)-\1', text_lower))
    fragment_count = len(re.findall(r'\b\w+- ', text_lower))
    oov_error_count = stutter_count + fragment_count
    
    # Clean the text of punctuation to accurately isolate whole words for the sets
    clean_text = text_lower.translate(str.maketrans('', '', string.punctuation))
    words = clean_text.split()
    total_words = len(words)
    
    # 3. Closed/Open class & Density
    closed_count = sum(1 for w in words if w in CLOSED_CLASS_WORDS)
    open_count = total_words - closed_count
    
    # Prevent ZeroDivisionError on completely empty or blank transcripts
    safe_closed = max(1, closed_count)
    safe_total = max(1, total_words)
    
    open_closed_ratio = open_count / safe_closed
    density = open_count / safe_total
    
    # 4. Task Noun Count
    task_noun_count = sum(1 for w in words if w in TASK_NOUNS)
    
    return {
        "Proxy_Open_Closed_Ratio": float(open_closed_ratio),
        "Proxy_Density": float(density),
        "Proxy_Adverb_Count": float(adverb_count),
        "Proxy_Task_Nouns_Count": float(task_noun_count),
        "Proxy_OOV_Error_Count": float(oov_error_count)
    }
