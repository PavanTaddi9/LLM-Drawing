import os
import json
import math
import time
from typing import Literal
from collections import defaultdict
import google.generativeai as genai
from pydantic import BaseModel, ValidationError, field_validator

# Configuration
MODEL_NAME = "gemini-2.5-pro-preview-05-06"
TARGET_COUNT = 5000
CATEGORIES = [
    "landscape", "abstract", "fashion"
]
OUTPUT_PATH = "descriptions_dataset.json"
RATE_LIMIT_DELAY = 1.2  # Seconds between API calls

# Track existing descriptions
existing_descriptions = set()

class Record(BaseModel):
    description: str
    category: Literal[
        "landscape", "abstract", "fashion", "architecture", 
        "interior", "technology", "food", "flora", "fauna", 
        "sports equipment", "office supplies", "toys"
    ]

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        v = v.strip()
        if len(v) < 15 or len(v) > 200:
            raise ValueError('Description must be 15-200 characters')
        if v.lower() in existing_descriptions:
            raise ValueError('Description already exists')
        return v

class RecordBatch(BaseModel):
    records: list[Record]

def configure_api():
    """Set up Gemini API credentials"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)

def build_prompt() -> str:
    """Construct generation prompt"""
    return f"""
    Generate unique visual descriptions with these requirements:

    1. Dataset Targets:
    - Total needed: {TARGET_COUNT}
    - Categories: {", ".join(CATEGORIES)}
    - {math.ceil(TARGET_COUNT/len(CATEGORIES))} per category
    - ABSOLUTELY NO REPETITIONS

    2. Description Rules:
    - Length: 15-200 characters
    - Focus on visual elements only
    - No brand names, people, or trademarks
    - Create DISTINCT, non-generic descriptions
    - Vary sentence structures and patterns
    - Include specific, uncommon details

    3. Example Format:
    Description: [unique description]
    Category: [category]

    4. Few-Shot Examples:
    Description: a purple forest at dusk
    Category: landscape
    Description: gray wool coat with a faux fur collar
    Category: fashion
    Description: crimson rectangles forming a chaotic grid
    Category: abstract
    Description: a green lagoon under a cloudy sky
    Category: landscape
    Description: burgundy corduroy pants with patch pockets
    Category: fashion
    Description: purple pyramids spiraling around a bronze cone
    Category: abstract
    Description: orange corduroy overalls
    Category: fashion
    Description: a starlit night over snow-covered peaks
    Category: landscape
    Description: magenta trapezoids layered on a transluscent surface
    Category: abstract
    Description: black and white checkered pants
    Category: fashion

    5. Anti-Repetition Measures:
    - No similar phrases or reworded versions
    - Each must be completely unique
    - If unsure, generate a different description

    Generate 5-10 new descriptions now with their categories.
    Follow these guidelines STRICTLY:
    - Use the exact example format
    - Maintain category balance
    - Prioritize uniqueness over quantity
    - No markdown formatting
    """

def generate_records(model) -> RecordBatch:
    """Generate and validate descriptions"""
    try:
        response = model.generate_content(
            build_prompt(),
            generation_config=genai.types.GenerationConfig(
                temperature=0.85,
                top_p=0.95,
                max_output_tokens=2048
            )
        )

        # Parse response text directly
        raw_text = response.text.strip()
        records = []
        
        current_desc = None
        for line in raw_text.split('\n'):
            if line.startswith("Description:"):
                current_desc = line.split(":", 1)[1].strip()
            elif line.startswith("Category:") and current_desc:
                category = line.split(":", 1)[1].strip().lower()
                if category in CATEGORIES:
                    records.append({"description": current_desc, "category": category})
                    current_desc = None

        return RecordBatch.model_validate({"records": records})
    except (KeyError, AttributeError, ValidationError) as e:
        raise ValueError(f"Invalid response format: {str(e)}")
    except Exception as e:
        raise ValueError(f"API error: {str(e)}")

def main():
    configure_api()
    model = genai.GenerativeModel(MODEL_NAME)

    counts = defaultdict(int)
    category_target = math.ceil(TARGET_COUNT / len(CATEGORIES))
    
    try:
        with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
            f.write('{\n  "records": [\n')
            first_entry = True
            total_written = 0
            
            while total_written < TARGET_COUNT:
                try:
                    batch = generate_records(model)
                    valid_records = []
                    
                    for record in batch.records:
                        desc = record.description.strip()
                        cat = record.category
                        
                        if (desc.lower() not in existing_descriptions and 
                            counts[cat] < category_target):
                            
                            existing_descriptions.add(desc.lower())
                            valid_records.append(record)
                            counts[cat] += 1
                            total_written += 1
                            
                            if total_written >= TARGET_COUNT:
                                break

                    # Write valid records
                    for record in valid_records:
                        json_str = json.dumps({
                            "description": record.description,
                            "category": record.category
                        }, ensure_ascii=False)
                        if not first_entry:
                            f.write(",\n")
                        f.write(f"    {json_str}")
                        first_entry = False
                    
                    print(f"Added {len(valid_records)} records (Total: {total_written})")
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except ValueError as e:
                    print(f"Skipping invalid batch: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    time.sleep(5)
                    
    except KeyboardInterrupt:
        print("\nEarly termination requested")
    
    finally:
        # Finalize JSON file
        with open(OUTPUT_PATH, "a", encoding='utf-8') as f:
            f.write('\n  ]\n}')
        
        # Print summary
        print("\nGeneration complete!")
        print(f"Total descriptions generated: {total_written}")
        print("Category distribution:")
        for cat in CATEGORIES:
            count = counts.get(cat, 0)
            target = math.ceil(TARGET_COUNT / len(CATEGORIES))
            print(f"- {cat}: {count}/{target} ({count/target:.1%})")

if __name__ == "__main__":
    main()