import json
import os
import time
from typing import List
import google.generativeai as genai
from pydantic import BaseModel, ValidationError, field_validator

# Configuration
INPUT_FILE = "descriptions_dataset.json"
OUTPUT_FILE = "enhanced_dataset_with_mcqs.json"
MODEL_NAME = "gemini-2.5-pro-preview-05-06"
RATE_LIMIT_DELAY = 1  # Seconds between API calls

class QAPair(BaseModel):
    question: str
    choices: List[str]
    answer: str

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v):
        if len(v) < 2 or len(v) > 4:
            raise ValueError('Must have 2-3 choices')
        if len(set(v)) != len(v):
            raise ValueError('Choices must be unique')
        return v

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v, values):
        if 'choices' in values.data and v not in values.data['choices']:
            raise ValueError('Answer must be in choices')
        return v

class UpdatedRecord(BaseModel):
    description: str
    category: str
    qa_pairs: List[QAPair]

def configure_api():
    """Set up Gemini API credentials"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)

def generate_questions(description: str, model) -> List[QAPair]:
    """Generate MCQs for a single description"""
    prompt = f"""
    Generate 2-3 multiple choice questions based EXCLUSIVELY on this description:
    "{description}"

    Requirements:
    1. Questions must test explicit information from the description
    2. Each question must have 2-4 unique choices
    3. Answers must be directly from the description content
    4. Format response as JSON:
    {{
        "qa_pairs": [
            {{
                "question": "...",
                "choices": ["...", "..."],
                "answer": "..."
            }}
        ]
    }}

    Example 1:
    Description: "a purple forest at dusk"
    {{
        "qa_pairs": [
            {{
                "question": "What is the main setting of the image?",
                "choices": ["beach", "desert", "forest", "mountain"],
                "answer": "forest"
            }},
            {{
                "question": "What time of day is shown?",
                "choices": ["dawn", "dusk", "midday", "midnight"],
                "answer": "dusk"
            }}
        ]
    }}

    Example 2:
    Description: "orange corduroy overalls"
    {{
        "qa_pairs": [
            {{
                "question": "What type of clothing is shown?",
                "choices": ["dress", "overalls", "shirt", "skirt"],
                "answer": "overalls"
            }},
            {{
                "question": "What material is used?",
                "choices": ["corduroy", "denim", "leather", "silk"],
                "answer": "corduroy"
            }}
        ]
    }}

    Example 3:
    Description: "purple pyramids spiraling around a bronze cone"
    {{
        "qa_pairs": [
            {{
                "question": "What shape is mentioned?",
                "choices": ["cone", "cube", "pyramid", "sphere"],
                "answer": "pyramid"
            }},
            {{
                "question": "What color is the central structure?",
                "choices": ["black", "bronze", "silver", "white"],
                "answer": "bronze"
            }}
        ]
    }}

    Example 4:
    Description: "a starlit night over snow-covered peaks"
    {{
        "qa_pairs": [
            {{
                "question": "What time is depicted?",
                "choices": ["dawn", "midday", "night", "sunset"],
                "answer": "night"
            }},
            {{
                "question": "What covers the peaks?",
                "choices": ["clouds", "forest", "snow", "trees"],
                "answer": "snow"
            }}
        ]
    }}

    Now generate questions for this description:
    """
  

    try:
        response = model.generate_content(prompt)
        raw_json = response.text[response.text.find('{'):response.text.rfind('}')+1]
        data = json.loads(raw_json)
        return [QAPair(**qa) for qa in data.get("qa_pairs", [])]
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return []

def process_dataset():
    """Main processing function"""
    configure_api()
    model = genai.GenerativeModel(MODEL_NAME)

    # Load existing data
    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)
    
    enhanced_records = []
    total = len(dataset['records'])
    success_count = 0
    
    for i, record in enumerate(dataset['records']):
        try:
            print(f"Processing {i+1}/{total}: {record['description'][:50]}...")
            qa_pairs = generate_questions(record['description'], model)
            
            if 2 <= len(qa_pairs) <= 3:
                enhanced = UpdatedRecord(
                    description=record['description'],
                    category=record['category'],
                    qa_pairs=qa_pairs
                )
                enhanced_records.append(enhanced.model_dump())
                success_count += 1
            else:
                print(f"Skipped record {i+1} - invalid QA count: {len(qa_pairs)}")
            
            time.sleep(RATE_LIMIT_DELAY)
            
        except ValidationError as e:
            print(f"Validation failed for record {i+1}: {str(e)}")
        except Exception as e:
            print(f"Error processing record {i+1}: {str(e)}")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"records": enhanced_records}, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{total}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()