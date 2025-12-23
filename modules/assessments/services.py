import librosa
import random

def calculate_phoneme_accuracy(audio_path):
    # phoneme_accuracy_model
    return round(random.uniform(70, 95), 2)

def calculate_word_level_accuracy(audio_path):
    # word_level_accuracy_model
    return round(random.uniform(60, 95), 2)

def calculate_fluency(audio_path):
    # calculate_fluency_model
    return round(random.uniform(65, 90), 2)

def weak_phonemes(audio_path):
    # weak_phonemes_model
    phonemes = [
    # --- CONSONANTS (24) ---
    "B", "D", "F", "G", "HH", "Y", "K", "L", "M", "N", "P", "R", 
    "S", "T", "V", "W", "Z", "SH", "ZH", "CH", "JH", "TH", "DH", "NG",

    # --- SHORT VOWELS (7) ---
    "AE",  # cat
    "EH",  # bed
    "IH",  # pig
    "AA",  # hot (US), father
    "AH",  # cup, love
    "UH",  # book
    "AX",  # about (The Schwa)

    # --- LONG VOWELS (5) ---
    "IY",  # sheep
    "AA",  # father (often same as short 'hot' in ARPABET)
    "AO",  # door, saw
    "UW",  # moon
    "ER",  # bird, nurse

    # --- DIPHTHONGS (8) ---
    "EY",    # day
    "AY",    # eye
    "OY",    # boy
    "AW",    # cow
    "OW",    # boat
    "IH R",  # ear (Combined ARPABET)
    "EH R",  # chair (Combined ARPABET)
    "UH R"   # tour (Combined ARPABET)
]
    weak = random.sample(phonemes, k=3)
    return weak

