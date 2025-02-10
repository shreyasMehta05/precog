# generate_similarity_dataset.py
# This script creates a comprehensive Hindi-English similarity dataset
# with words from various categories

word_pairs = [
    # Basic objects and nature
    ("कार", "car", 8.94),
    ("कुत्ता", "dog", 9.12),
    ("पानी", "water", 8.73),
    ("गरम", "hot", 7.56),
    ("ठंडा", "cold", 7.89),
    ("सूर्य", "sun", 9.20),
    ("चाँद", "moon", 8.65),
    ("पेड़", "tree", 8.80),
    ("फूल", "flower", 8.50),
    ("हवा", "wind", 8.40),
    
    # Animals
    ("बिल्ली", "cat", 9.15),
    ("घोड़ा", "horse", 8.95),
    ("चिड़िया", "bird", 8.82),
    ("मछली", "fish", 9.05),
    ("हाथी", "elephant", 9.30),
    
    # Food and drinks
    ("चाय", "tea", 9.40),
    ("दूध", "milk", 9.25),
    ("चावल", "rice", 9.18),
    ("रोटी", "bread", 8.75),
    ("मिठाई", "sweet", 8.45),
    
    # Colors
    ("लाल", "red", 9.10),
    ("नीला", "blue", 9.08),
    ("हरा", "green", 9.12),
    ("काला", "black", 9.15),
    ("सफेद", "white", 9.20),
    
    # Family relations
    ("माता", "mother", 9.45),
    ("पिता", "father", 9.42),
    ("बहन", "sister", 9.38),
    ("भाई", "brother", 9.40),
    ("दादी", "grandmother", 9.15),
    
    # Common verbs
    ("खाना", "eat", 8.85),
    ("पीना", "drink", 8.88),
    ("सोना", "sleep", 8.92),
    ("देखना", "see", 8.75),
    ("चलना", "walk", 8.70),
    
    # Time and numbers
    ("दिन", "day", 9.25),
    ("रात", "night", 9.22),
    ("एक", "one", 9.50),
    ("दो", "two", 9.48),
    ("तीन", "three", 9.45),
    
    # Body parts
    ("आँख", "eye", 9.30),
    ("नाक", "nose", 9.28),
    ("हाथ", "hand", 9.25),
    ("पैर", "foot", 9.20),
    ("सिर", "head", 9.15),
    
    # Emotions and feelings
    ("खुश", "happy", 8.65),
    ("दुखी", "sad", 8.62),
    ("गुस्सा", "angry", 8.58),
    ("प्यार", "love", 8.75),
    ("डर", "fear", 8.70),
    
    # Weather and seasons
    ("बारिश", "rain", 9.15),
    ("धूप", "sunshine", 8.85),
    ("बर्फ", "snow", 9.20),
    ("गर्मी", "summer", 8.95),
    ("सर्दी", "winter", 8.92),
    
    # Professions
    ("डॉक्टर", "doctor", 9.35),
    ("शिक्षक", "teacher", 9.30),
    ("किसान", "farmer", 9.25),
    ("वकील", "lawyer", 9.28),
    ("इंजीनियर", "engineer", 9.32),
    
    # Places
    ("घर", "house", 9.15),
    ("स्कूल", "school", 9.25),
    ("बाज़ार", "market", 8.85),
    ("अस्पताल", "hospital", 9.30),
    ("मंदिर", "temple", 8.75),
    
    # Technology
    ("कंप्यूटर", "computer", 9.45),
    ("फ़ोन", "phone", 9.42),
    ("इंटरनेट", "internet", 9.40),
    ("वेबसाइट", "website", 9.38),
    ("लैपटॉप", "laptop", 9.44),
    
    # Direction and position
    ("ऊपर", "up", 8.85),
    ("नीचे", "down", 8.82),
    ("अंदर", "inside", 8.78),
    ("बाहर", "outside", 8.75),
    ("पास", "near", 8.70),
    
    # Common adjectives
    ("बड़ा", "big", 8.95),
    ("छोटा", "small", 8.92),
    ("अच्छा", "good", 8.65),
    ("बुरा", "bad", 8.62),
    ("नया", "new", 8.88)
]
# Additional categories and word pairs
new_word_pairs = [
    # Transportation
    ("बस", "bus", 9.4),
    ("ट्रेन", "train", 9.35),
    ("साइकिल", "bicycle", 9.3),
    ("हवाई जहाज़", "airplane", 9.25),
    ("नाव", "boat", 9.1),

    # Education
    ("किताब", "book", 9.4),
    ("कलम", "pen", 9.2),
    ("पेंसिल", "pencil", 9.1),
    ("कक्षा", "classroom", 8.9),
    ("विद्यार्थी", "student", 9.0),

    # Clothing
    ("शर्ट", "shirt", 9.3),
    ("पैंट", "pants", 9.25),
    ("जूता", "shoe", 9.2),
    ("टोपी", "hat", 9.1),
    ("साड़ी", "sari", 8.8),

    # Household Items
    ("मेज़", "table", 9.3),
    ("कुर्सी", "chair", 9.25),
    ("बिस्तर", "bed", 9.4),
    ("दर्पण", "mirror", 9.1),
    ("तकिया", "pillow", 9.0),

    # Sports
    ("फुटबॉल", "football", 9.3),
    ("क्रिकेट", "cricket", 9.4),
    ("बैडमिंटन", "badminton", 9.2),
    ("तैराकी", "swimming", 8.9),
    ("खेल", "game", 8.8),

    # Government/Politics
    ("राष्ट्रपति", "president", 9.4),
    ("प्रधानमंत्री", "prime minister", 9.35),
    ("चुनाव", "election", 9.3),
    ("कानून", "law", 9.2),
    ("सरकार", "government", 9.4),

    # Banking/Finance
    ("पैसा", "money", 9.5),
    ("बैंक", "bank", 9.4),
    ("ऋण", "loan", 9.2),
    ("बचत", "savings", 9.0),
    ("निवेश", "investment", 9.1),

    # Medical Terms
    ("दवा", "medicine", 9.3),
    ("नर्स", "nurse", 9.3),
    ("रोग", "disease", 9.1),
    ("इलाज", "treatment", 9.0),
    ("टीका", "vaccine", 8.8),

    # Nature
    ("पहाड़", "mountain", 9.3),
    ("नदी", "river", 9.4),
    ("झील", "lake", 9.2),
    ("जंगल", "forest", 9.1),
    ("रेत", "sand", 9.0),

    # Technology
    ("ऐप", "app", 9.4),
    ("सोशल मीडिया", "social media", 9.3),
    ("डेटा", "data", 9.5),
    ("सॉफ़्टवेयर", "software", 9.45),
    ("हार्डवेयर", "hardware", 9.4),

    # Kitchen Items
    ("चम्मच", "spoon", 9.3),
    ("कांटा", "fork", 9.2),
    ("चाकू", "knife", 9.25),
    ("प्लेट", "plate", 9.3),
    ("गिलास", "glass", 9.1),

    # Musical Instruments
    ("गिटार", "guitar", 9.3),
    ("तबला", "tabla", 8.9),
    ("पियानो", "piano", 9.2),
    ("बांसुरी", "flute", 9.1),
    ("ड्रम", "drum", 9.0),

    # Shopping
    ("दुकान", "shop", 9.2),
    ("कीमत", "price", 9.1),
    ("छूट", "discount", 8.9),
    ("खरीदना", "buy", 8.8),
    ("ग्राहक", "customer", 9.0),

    # Travel/Tourism
    ("होटल", "hotel", 9.4),
    ("यात्रा", "travel", 9.1),
    ("पासपोर्ट", "passport", 9.3),
    ("सूटकेस", "suitcase", 9.2),
    ("पर्यटक", "tourist", 9.0),

    # Legal Terms
    ("न्यायालय", "court", 9.3),
    ("न्यायाधीश", "judge", 9.35),
    ("गवाह", "witness", 9.0),
    ("अपराध", "crime", 9.1),
    ("क़ानूनी", "legal", 9.2)
]

# Combine with original list
word_pairs += new_word_pairs

# Name of the output file
output_filename = "./data/similarity_dataset.txt"

# Write the dataset to the file using UTF-8 encoding
with open(output_filename, "w", encoding="utf-8") as f:
    # Write header  
    f.write("Hindi\tEnglish\tSimilarity_Score\n")
    
    # Write data
    for hindi, english, score in word_pairs:
        f.write(f"{hindi}\t{english}\t{score}\n")

print(f"Extended dataset has been written to {output_filename}")
print(f"Total word pairs: {len(word_pairs)}")