import os
import json
import math
import time
import uuid
from typing import Literal
from collections import defaultdict
import google.generativeai as genai
from pydantic import BaseModel, ValidationError, field_validator

# Configuration
MODEL_NAME = "gemini-2.5-pro-preview-05-06"
TARGET_COUNT = 10000
PRIORITY_CATEGORIES = ["landscape", "abstract", "fashion"]
OTHER_CATEGORIES = [
    "architecture", "interior", "technology", "food",
    "flora", "fauna", "sports equipment", "office supplies", "toys"
]
OUTPUT_ROOT = os.path.expanduser("/Users/pavankumartaddi/Desktop/llm-Drawing/results")  # Customize this path
RATE_LIMIT_DELAY = 1.2

# Create output directory structure
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Pydantic Models
class QAPair(BaseModel):
    question: str
    choices: list[str]
    answer: str

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v):
        if len(v) < 2 or len(v) > 4:
            raise ValueError('Must have 2-4 choices')
        if len(set(v)) != len(v):
            raise ValueError('Choices must be unique')
        return v

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v, values):
        if 'choices' in values.data and v not in values.data['choices']:
            raise ValueError('Answer must be in choices')
        return v

class Record(BaseModel):
    description: str
    category: Literal[
        "landscape", "abstract", "fashion", "architecture", 
        "interior", "technology", "food", "flora", "fauna", 
        "sports equipment", "office supplies", "toys"
    ]
    qa_pairs: list[QAPair]

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        v = v.strip()
        if len(v) < 15 or len(v) > 200:
            raise ValueError('Description must be 15-200 characters')
        prohibited = ['brand', 'logo', 'person', 'man', 'woman']
        if any(term in v.lower() for term in prohibited):
            raise ValueError('Description contains prohibited terms')
        return v

    @field_validator('qa_pairs')
    @classmethod
    def validate_qa_pairs(cls, v):
        if len(v) < 2 or len(v) > 4:
            raise ValueError('Must have 2-4 QA pairs')
        return v

class RecordBatch(BaseModel):
    records: list[Record]

# API-Compatible Function Schema
FUNCTION_SCHEMA = {
    "name": "generate_record_batch",
    "description": "Generate validated MCQA records",
    "parameters": {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": PRIORITY_CATEGORIES + OTHER_CATEGORIES
                        },
                        "qa_pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "choices": {"type": "array", "items": {"type": "string"}},
                                    "answer": {"type": "string"}
                                },
                                "required": ["question", "choices", "answer"]
                            }
                        }
                    },
                    "required": ["description", "category", "qa_pairs"]
                }
            }
        },
        "required": ["records"]
    }
}

def configure_api():
    """Set up Gemini API credentials"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)

def build_prompt() -> str:
    """Construct system prompt with examples"""
    return f"""
    Generate synthetic MCQA data with these requirements:

    1. Priority Categories (50% of total):
    - {", ".join(PRIORITY_CATEGORIES)}
    
    2. Other Categories (50% of total):
    - {", ".join(OTHER_CATEGORIES)}

    2. Record Structure:
    - description: Short, concise scene/object description (max 200 chars)
      - Focus on visual elements of COMMON, GENERIC SUBJECTS
      - MUST NOT contain:
        1. Brand names or trademarks
        2. Personal names
        3. People (even in generic form like "a person" or "a man")
      - MUST NOT be empty
      - Average length ~50 characters
    - category: ONE category from the provided list
    - qa_pairs: 2-4 multiple-choice QA pairs

    3. QA Pair Requirements:
    - Questions MUST probe EXPLICIT elements from the description
    - Choices must be 2-4 unique strings
    - Answer MUST be one of the provided choices
    - Base answers ONLY on the description content

    4. Generation Rules:
    - Vary categories within each batch
    - Maintain strict factual consistency
    - Respond ONLY using generate_record_batch function
    - ANY reference to brands, people, or names makes the record INVALID

    Example 1:
    Description: a purple forest at dusk
    Category: landscape
    QA Pairs:
    - Question: What is the main setting of the image?
      Choices: [beach, desert, forest, mountain]
      Answer: forest
    - Question: Is there anything purple in the image?
      Choices: [no, yes]
      Answer: yes
    - Question: What time of day is suggested in the image?
      Choices: [dawn, dusk, midday, midnight]
      Answer: dusk
    - Question: What color is prominently featured in the image?
      Choices: [green, orange, purple, white]
      Answer: purple

    Example 2:
    Description: orange corduroy overalls
    Category: fashion
    QA Pairs:
    - Question: What color is the coat?
      Choices: [blue, brown, gray, red]
      Answer: gray
    - Question: What part of the coat has faux fur?
      Choices: [collar, hem, pockets, sleeves]
      Answer: collar
    - Question: Is the coat purple?
      Choices: [no, yes]
      Answer: no
    - Question: What material is the coat made of?
      Choices: [cotton, leather, silk, wool]
      Answer: wool

    Example 3:
    Description: purple pyramids spiraling around a bronze cone
    Category: abstract
    QA Pairs:
    - Question: Is there an ocean visible in the image?
      Choices: [no, yes]
      Answer: yes
    - Question: What is the spatial relationship between the lighthouse and the ocean?
      Choices: [inside, next to, overlooking, under]
      Answer: overlooking
    - Question: Is there a desert in the image?
      Choices: [no, yes]
      Answer: no
    - Question: Is the lighthouse located under the ocean?
      Choices: [no, yes]
      Answer: no

    Example 4:
    Description: A sleek black gaming laptop glowing blue on a desk
    Category: technology
    QA Pairs:
    - Question: What item is on the desk?
      Choices: [monitor, keyboard, laptop, mouse]
      Answer: laptop
    - Question: What color is the laptop primarily?
      Choices: [white, silver, black, grey]
      Answer: black
    - Question: What color is the laptop glowing?
      Choices: [red, green, blue, yellow]
      Answer: blue
    - Question: Is the laptop described as bulky?
      Choices: [yes, no]
      Answer: no

    Generate 3-5 new records now. Ensure:
    - No brands/people/names in descriptions
    - Category variety within batch
    - Strict adherence to example format
    """

def process_response(response) -> RecordBatch:
    """Extract and validate response data"""
    try:
        func_call = response.candidates[0].content.parts[0].function_call
        raw_data = func_call.args.get("records", [])
        return RecordBatch.model_validate({"records": raw_data})
    except (KeyError, AttributeError, ValidationError) as e:
        raise ValueError(f"Invalid response: {str(e)}")

def save_record(record: Record):
    """Save record to category folder with progress tracking"""
    # Create category folder
    folder_name = record.category.replace(" ", "_").lower()
    category_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(category_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{uuid.uuid4().hex}.json"
    filepath = os.path.join(category_dir, filename)
    
    # Save record
    with open(filepath, "w") as f:
        json.dump(record.model_dump(), f, indent=2)

def print_progress(counts: dict, targets: dict, total: int):
    """Display real-time generation progress"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
    
    print("=== Dataset Generation Progress ===")
    print(f"Total Generated: {total}/{TARGET_COUNT}\n")
    
    print("Priority Categories (50% target):")
    for cat in PRIORITY_CATEGORIES:
        count = counts.get(cat, 0)
        target = targets[cat]
        print(f"- {cat:<16}: {count:>4}/{target} ({count/target:.1%})")
    
    print("\nOther Categories (50% target):")
    for cat in OTHER_CATEGORIES:
        count = counts.get(cat, 0)
        target = targets[cat]
        print(f"- {cat:<16}: {count:>4}/{target} ({count/target:.1%})")

def main():
    configure_api()
    
    # Calculate targets
    total_priority = TARGET_COUNT // 2
    priority_target = math.ceil(total_priority / len(PRIORITY_CATEGORIES))
    priority_targets = {cat: priority_target for cat in PRIORITY_CATEGORIES}
    
    total_other = TARGET_COUNT // 2
    other_target = math.ceil(total_other / len(OTHER_CATEGORIES))
    other_targets = {cat: other_target for cat in OTHER_CATEGORIES}
    
    category_targets = {**priority_targets, **other_targets}

    model = genai.GenerativeModel(
        MODEL_NAME,
        tools=[{"function_declarations": [FUNCTION_SCHEMA]}]
    )

    counts = defaultdict(int)
    total = 0
    
    try:
        while total < TARGET_COUNT:
            try:
                response = model.generate_content(
                    build_prompt(),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=2048
                    )
                )
                
                batch = process_response(response)
                new_records = 0
                
                for record in batch.records:
                    cat = record.category
                    if counts[cat] < category_targets[cat]:
                        save_record(record)
                        counts[cat] += 1
                        total += 1
                        new_records += 1
                        
                        if total >= TARGET_COUNT:
                            break
                
                print_progress(counts, category_targets, total)
                time.sleep(RATE_LIMIT_DELAY)
                
            except ValueError as e:
                print(f"Skipping invalid batch: {str(e)}")
            except Exception as e:
                print(f"API error: {str(e)}")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nGeneration stopped by user")
    
    finally:
        print("\n=== Final Dataset Summary ===")
        print_progress(counts, category_targets, total)
        print(f"\nAll records saved in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()