"""
Seed script to add initial assessment sentences to the database.
Run this script to populate the sentences table with sample data.
"""

import requests
import json

# Backend API base URL
BASE_URL = "http://localhost:5000/api"

# Sample sentences with phonemes for speech assessment
SAMPLE_SENTENCES = [
    {
        "text": "The quick brown fox jumps over the lazy dog.",
        "source_type": "assessment",
        "difficulty": "easy",
        "phonemes": ["TH", "K", "W", "B", "R", "F", "JH", "Z", "L", "D", "G"]
    },
    {
        "text": "She sells seashells by the seashore.",
        "source_type": "assessment",
        "difficulty": "medium",
        "phonemes": ["SH", "S", "Z", "L", "B", "TH"]
    },
    {
        "text": "Peter Piper picked a peck of pickled peppers.",
        "source_type": "assessment",
        "difficulty": "hard",
        "phonemes": ["P", "T", "R", "K", "D", "L"]
    },
    {
        "text": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "source_type": "assessment",
        "difficulty": "hard",
        "phonemes": ["W", "CH", "K", "D", "M"]
    },
    {
        "text": "The rain in Spain stays mainly in the plain.",
        "source_type": "assessment",
        "difficulty": "medium",
        "phonemes": ["R", "N", "S", "P", "M", "L"]
    },
    {
        "text": "I scream, you scream, we all scream for ice cream.",
        "source_type": "assessment",
        "difficulty": "easy",
        "phonemes": ["S", "K", "R", "M", "W", "L"]
    },
    {
        "text": "Red lorry, yellow lorry.",
        "source_type": "assessment",
        "difficulty": "medium",
        "phonemes": ["R", "L", "Y"]
    },
    {
        "text": "A proper copper coffee pot.",
        "source_type": "assessment",
        "difficulty": "medium",
        "phonemes": ["P", "R", "K", "F", "T"]
    },
    {
        "text": "The sixth sick sheikh's sixth sheep's sick.",
        "source_type": "assessment",
        "difficulty": "hard",
        "phonemes": ["S", "K", "TH", "SH"]
    },
    {
        "text": "Can you can a can as a canner can can a can?",
        "source_type": "assessment",
        "difficulty": "medium",
        "phonemes": ["K", "N", "Y"]
    }
]


def check_existing_sentences():
    """Check if there are already sentences in the database"""
    try:
        response = requests.get(f"{BASE_URL}/sentences/list")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data['count']} existing sentences in database")
            return data['count']
        else:
            print(
                f"⚠ Could not check existing sentences: {response.status_code}")
            return 0
    except Exception as e:
        print(f"⚠ Error checking sentences: {e}")
        return 0


def add_sentences():
    """Add sample sentences to the database"""
    print("\n🌱 Seeding database with sample sentences...\n")

    existing_count = check_existing_sentences()

    if existing_count > 0:
        response = input(
            f"\nDatabase already has {existing_count} sentences. Add more? (y/n): ")
        if response.lower() != 'y':
            print("Skipping seed operation.")
            return

    print(f"\nAdding {len(SAMPLE_SENTENCES)} sentences...\n")

    success_count = 0
    for i, sentence_data in enumerate(SAMPLE_SENTENCES, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/sentences/add-sentence",
                json={
                    "sentence": sentence_data["text"],
                    "source_type": sentence_data["source_type"],
                    "difficulty": sentence_data["difficulty"],
                    "phonemes": sentence_data["phonemes"]
                },
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 201:
                data = response.json()
                print(
                    f"✓ [{i}/{len(SAMPLE_SENTENCES)}] Added: {sentence_data['text'][:50]}... (ID: {data['id']})")
                success_count += 1
            else:
                print(
                    f"✗ [{i}/{len(SAMPLE_SENTENCES)}] Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"✗ [{i}/{len(SAMPLE_SENTENCES)}] Error: {e}")

    print(
        f"\n✅ Successfully added {success_count}/{len(SAMPLE_SENTENCES)} sentences")

    # Show final count
    final_count = check_existing_sentences()
    print(f"\n📊 Total sentences in database: {final_count}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Speech Therapy Platform - Database Seed Script")
    print("=" * 60)

    try:
        add_sentences()
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
