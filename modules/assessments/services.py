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
    phonemes = ['p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 's', 'z', 'f', 'v']
    weak = random.sample(phonemes, k=3)
    return weak

