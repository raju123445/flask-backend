from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os

from .models import AudioRecord, Sentence
from config import db
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

    # if not sentence_id:
    #     return jsonify({"error": "Sentence ID missing"}), 400

    # sentence = Sentence.query.get(sentence_id)
    
    # if not sentence:
    #     return jsonify({"error": "Invalid sentence_id"}), 404
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
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