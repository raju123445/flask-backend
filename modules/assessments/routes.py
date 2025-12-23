from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
from sqlalchemy import not_

from .models import AudioRecord
# importing sentence model fomr sentence directory
# from ..sentences.models import Sentence
from modules.sentences.models import Sentence
from extensions import db
from .services import (
    calculate_phoneme_accuracy,
    calculate_word_level_accuracy,
    calculate_fluency,
    weak_phonemes
)

assess_bp = Blueprint("assessment", __name__)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@assess_bp.route("/add-sentence", methods=["POST"])
def add_sentence():
    sentence_text = request.json.get("text")
    if not sentence_text:
        return jsonify({"error": "Sentence text is required"}), 400

    new_sentence = Sentence(
        text=sentence_text,
        source_type='custom',
        difficulty='medium',
        )
    db.session.add(new_sentence)
    db.session.commit()

    return jsonify({"message": "Sentence added successfully", "sentence_id": new_sentence.id}), 201

@assess_bp.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["audio"]
    user_id = request.form.get("user_id")
    sentence_id = request.form.get("sentence_id")
    # print(file, "  ", user_id, "  ", sentence_id)

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not sentence_id:
        return jsonify({"error": "Sentence ID missing"}), 400

    sentence = Sentence.query.get(sentence_id)
    
    if not sentence:
        return jsonify({"error": "Invalid sentence_id"}), 404
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    if file and allowed_file(file.filename):
        _, ext = os.path.splitext(file.filename)
        
        timestamp = int(time.time()) 
        filename = secure_filename(file.filename)
        filename = f"assessment_audio_{timestamp}{ext}"
        # filename = secure_filename(file.filename)
        upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        phoneme_accuracy = calculate_phoneme_accuracy(file_path)
        word_accuracy = calculate_word_level_accuracy(file_path)
        fluency_score = calculate_fluency(file_path)
        weak_phonemes_list = weak_phonemes(file_path)

        audio_record = AudioRecord(
            user_id=user_id,
            file_path=file_path,
            sentence_id=sentence_id,
            phoneme_accuracy=phoneme_accuracy,
            word_accuracy=word_accuracy,
            fluency_score=fluency_score,
            weak_phonemes=weak_phonemes_list
        )
        db.session.add(audio_record)
        db.session.commit()

        return jsonify({
            "message": "File uploaded and assessed successfully",
            "data": {
                "phoneme_accuracy": phoneme_accuracy,
                "word_accuracy": word_accuracy,
                "fluency_score": fluency_score,
                "weak_phonemes": weak_phonemes_list
            }
        }), 201
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    
# Sentence recomondation algorithm route
@assess_bp.route("/recommend/<int:user_id>", methods=["GET"])
def recommend_sentence(user_id):
    try:
        # 1. Get user assessment history
        assessments = AudioRecord.query.filter_by(user_id=user_id).all()

        # 2. Cold start: First assessment
        if not assessments:
            sentence = Sentence.query.first()
            if not sentence:
                return jsonify({"message": "No sentences available"}), 404

            return jsonify({
                "sentence_id": sentence.id,
                "text": sentence.text,
                "difficulty": sentence.difficulty,
                "phonemes": sentence.phonemes,
                "reason": "First assessment"
            }), 200

        # 3. Sentences already completed
        used_sentence_ids = {a.sentence_id for a in assessments}

        # 4. Collect all weak phonemes
        weak_phonemes = set()
        for a in assessments:
            if a.weak_phonemes:
                weak_phonemes.update(a.weak_phonemes)

        # 5. Candidate sentences (not repeated)
        candidates = Sentence.query.filter(
            not_(Sentence.id.in_(used_sentence_ids))
        ).all()

        if not candidates:
            return jsonify({
                "message": "All sentences completed",
                "reason": "All sentence are done with this user(exhausted)"
            }), 200

        # 6. Score sentences by phoneme overlap
        best_sentence = None
        best_score = 0

        for sentence in candidates:
            if not sentence.phonemes:
                continue

            overlap = set(sentence.phonemes).intersection(weak_phonemes)
            score = len(overlap)

            if score > best_score:
                best_score = score
                best_sentence = sentence

        # 7. If phoneme match found
        if best_sentence:
            return jsonify({
                "sentence_id": best_sentence.id,
                "text": best_sentence.text,
                "difficulty": best_sentence.difficulty,
                "matched_phonemes": list(
                    set(best_sentence.phonemes).intersection(weak_phonemes)
                ),
                "phonmes": best_sentence.phonemes,
                "reason": "weak_phoneme_match"
            }), 200

        # 8. Fallback: no phoneme match
        fallback = candidates[0]
        return jsonify({
            "sentence_id": fallback.id,
            "text": fallback.text,
            "difficulty": fallback.difficulty,
            "phonemes": fallback.phonemes,
            "reason": "no_phoneme_match"
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Recommendation failed",
            "details": str(e)
        }), 500