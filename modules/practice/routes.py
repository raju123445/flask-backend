from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
import requests
import json
import re

from extensions import db
from modules.sentences.models import Sentence
from modules.assessments.services import (
    calculate_phoneme_accuracy,
    calculate_word_level_accuracy,
    calculate_fluency
)

# Get API key from environment or config
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')


def convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    return obj


routes_bp = Blueprint('practice', __name__)
# practice sentence recommendation for the chosen weak phonemes


@routes_bp.route('/recommendation', methods=['GET'])
def recommend_practice_sentences():
    # Support GET with query param `phonemes=AE,IH` or POST JSON {"weak_phonemes": [...]}.
    # 1. Get weak_phonemes from query params (e.g., ?weak_phonemes=R,L,TH)
    weak_phonemes_input = request.args.get('weak_phonemes', '')
    weak_words = request.args.get('weak_words', '')
    focus_clause = ""
    if weak_phonemes_input and weak_words:
        # Split by comma and clean up any extra whitespace
        phoneme_list = [p.strip() for p in weak_phonemes_input.split(',') if p.strip()]
        weak_words_list = [p.strip() for p in weak_words.split(',') if p.strip()]   
        # Join them back into a clean, comma-spaced string for the AI
        clean_phonemes = ", ".join(phoneme_list)
        clean_words = ", ".join(weak_words_list)
        
        focus_clause = f"The sentence MUST heavily feature these specific ARPAbet phonemes and words: {clean_phonemes} and {clean_words}."

    prompt = f"""
    Generate ONE random English practice sentence (8–12 words) for speech practice.
    {focus_clause}
    Avoid repeating similar sentences.
    Return ONLY valid JSON in this exact format:
    {{"text":"","difficulty":""}}

    Use ARPAbet phonemes.
    Difficulty must be easy, medium, or hard.
    """.strip()

    # --- REST OF YOUR LLM CALL CODE ---
    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 200
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=15
        )
        print("OpenRouter response status:", response)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # print("Generated content:", content)

        # Attempt to extract a JSON object from the model output robustly
        def _extract_json_from_text(text):
            # Find candidate JSON object(s) using braces - try the largest first
            matches = re.findall(r'\{.*?\}', text, re.DOTALL)
            for m in sorted(matches, key=len, reverse=True):
                try:
                    return json.loads(m)
                except json.JSONDecodeError:
                    # try a relaxed fallback by replacing single quotes
                    try:
                        return json.loads(m.replace("'", '"'))
                    except Exception:
                        continue
            # Fallback: slice from first '{' to last '}' and try again
            if '{' in text and '}' in text:
                s = text[text.find('{'): text.rfind('}')+1]
                try:
                    return json.loads(s)
                except Exception:
                    import ast
                    try:
                        return ast.literal_eval(s)
                    except Exception:
                        return None
            return None

        parsed = _extract_json_from_text(content)
        if parsed is None:
            return jsonify({"error": "Failed to parse model response", "raw": content}), 502

        # Normalize keys and types
        if isinstance(parsed, dict):
            # Accept 'phon' as alias for 'phonemes'
            if 'phon' in parsed and 'phonemes' not in parsed:
                parsed['phonemes'] = parsed.pop('phon')

        # Ensure phonemes is a list of strings
        if 'phonemes' in parsed:
            if isinstance(parsed['phonemes'], str):
                parsed['phonemes'] = [p.strip(' "[],.') for p in re.split(
                    r'[,\s]+', parsed['phonemes']) if p.strip()]
            elif isinstance(parsed['phonemes'], (list, tuple)):
                parsed['phonemes'] = [str(p) for p in parsed['phonemes']]

        # Normalize difficulty
        diff = parsed.get('difficulty', '')
        if not isinstance(diff, str) or diff.lower() not in ('easy', 'medium', 'hard'):
            parsed['difficulty'] = 'medium'
        else:
            parsed['difficulty'] = diff.lower()

        # Validate text
        text_val = parsed.get('text', '')
        if not isinstance(text_val, str) or not text_val.strip():
            return jsonify({"error": "Model output missing or invalid 'text' field", "raw": content}), 502

        return jsonify({
            "recommendations": [{
                "sentence_id": None,
                "text": text_val.strip(),
                "difficulty": parsed['difficulty'],
                # "phonemes": parsed.get('phonemes', []),
                "matched": []  # For AI-generated sentences, no pre-matched phonemes
            }]
        }), 200

    # If model returned non-object JSON (e.g., an array), return an error
        return jsonify({"error": "Model returned non-object JSON", "raw": content}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Submit practice audio and return assessment using existing services (no DB persistence by default)
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@routes_bp.route('/submit', methods=['POST'])
def submit_practice_audio():
    # Expects multipart/form-data with 'audio' file and 'sentence_id' (and optional 'user_id')
    if 'audio' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['audio']
    sentence_id = request.form.get('sentence_id')
    # user_id = request.form.get('user_id')  # Not used for practice submissions

    # Handle AI-generated sentences (sentence_id is null or "null")
    # Practice submissions don't require a valid sentence in the database
    sentence = None
    if sentence_id and sentence_id.lower() not in ('null', 'none', ''):
        try:
            sentence = Sentence.query.get(int(sentence_id))
            if not sentence:
                # If sentence_id is provided but invalid, log warning but continue
                print(
                    f"Warning: sentence_id {sentence_id} not found in database, continuing with practice evaluation")
        except (ValueError, TypeError):
            # Invalid sentence_id format, but we can still evaluate the audio
            print(
                f"Warning: Invalid sentence_id format '{sentence_id}', continuing with practice evaluation")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        _, ext = os.path.splitext(file.filename)
        timestamp = int(time.time())
        filename = secure_filename(file.filename)
        filename = f"practice_audio_{timestamp}{ext}"
        upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Call assessment services
        phoneme_data = calculate_phoneme_accuracy(file_path)
        word_data = calculate_word_level_accuracy(file_path)
        fluency_score = calculate_fluency(file_path)

        # Convert all numpy types to native Python types
        phoneme_data = convert_to_native_types(phoneme_data)
        word_data = convert_to_native_types(word_data)
        fluency_score = convert_to_native_types(fluency_score)

        # Helper: derive numeric average from model outputs
        def _avg_score_from_list(data):
            try:
                if data is None:
                    return None
                if isinstance(data, list):
                    scores = []
                    for item in data:
                        if isinstance(item, dict) and 'score' in item:
                            try:
                                s = float(item.get('score', 0))
                                scores.append(s)
                            except Exception:
                                continue
                        elif isinstance(item, (int, float)):
                            scores.append(float(item))
                    if not scores:
                        return None
                    return round(sum(scores) / len(scores), 2)
                elif isinstance(data, (int, float)):
                    return float(data)
            except Exception:
                pass
            return None

        # Convert detailed outputs to DB-friendly numeric values
        phoneme_accuracy = _avg_score_from_list(phoneme_data)
        word_accuracy = _avg_score_from_list(word_data)

        # Get weak phonemes
        try:
            computed_weak = []
            from modules.assessments.services import weak_phonemes as get_weak
            computed_weak = get_weak(file_path)
            computed_weak = convert_to_native_types(computed_weak)
        except Exception:
            computed_weak = []

        # Only check for matched phonemes if we have a valid sentence from database
        matched_phonemes = []
        if sentence and sentence.phonemes:
            matched_phonemes = list(
                set(sentence.phonemes).intersection(set(computed_weak)))

        # Prepare detailed results similar to assessment endpoint
        details = {
            "phoneme_details": phoneme_data,
            "word_details": word_data
        }

        return jsonify({
            "message": "Practice submission evaluated",
            "data": {
                "phoneme_accuracy": phoneme_accuracy,
                "word_accuracy": word_accuracy,
                "fluency_score": fluency_score,
                "computed_weak_phonemes": computed_weak,
                "details": details,
                "sentence_id": sentence.id if sentence else None,
            }
        }), 201

    return jsonify({"error": "Invalid file type"}), 400
