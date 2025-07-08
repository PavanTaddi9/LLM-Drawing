import os
import json
import math
import time
from typing import Literal
from collections import defaultdict
import google.generativeai as genai
from pydantic import BaseModel, ValidationError, field_validator

# Configurationcle
MODEL_NAME = "models/gemini-2.0-flash"
TARGET_COUNT = 10000
CATEGORIES = [
    "landscape", "abstract", "fashion", "architecture", "interior",
    "technology", "food", "flora", "fauna", "sports equipment",
    "office supplies", "toys"
]
OUTPUT_PATH = "synthetic_dataset.json"
RATE_LIMIT_DELAY = 1.2  # Seconds between API calls

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
        return v

    @field_validator('qa_pairs')
    @classmethod
    def validate_qa_pairs(cls, v):
        if len(v) < 2 or len(v) > 4:
            raise ValueError('Must have 2-4 QA pairs')
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
    """Construct system prompt with few-shot examples and critical instructions"""
    return f"""
    Generate synthetic MCQA data with these strict requirements:

     You are an expert data synthesizer. Generate records with these strict requirements:

    1. Dataset Targets:
    - Total records needed: {TARGET_COUNT}
    - Evenly distributed across {len(CATEGORIES)} categories: {", ".join(CATEGORIES)}
    - {math.ceil(TARGET_COUNT/len(CATEGORIES))} records per category

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

def generate_records(model) -> RecordBatch:
    """Generate and validate a batch of records"""
    try:
        response = model.generate_content(
            build_prompt(),
            generation_config=genai.types.GenerationConfig(
                temperature=0.75,
                top_p=0.95,
                max_output_tokens=2048
            )
        )

        func_call = response.candidates[0].content.parts[0].function_call
        raw_records = func_call.args.get("records", [])
        return RecordBatch.model_validate({"records": raw_records})
    except (KeyError, AttributeError, ValidationError) as e:
        raise ValueError(f"Invalid response format: {str(e)}")
    except Exception as e:
        raise ValueError(f"API error: {str(e)}")

def main():
    configure_api()
    model = genai.GenerativeModel(
        MODEL_NAME,
        tools=[{
            "function_declarations": [{
                "name": "generate_record_batch",
                "description": "Generate validated MCQA records",
                "parameters": RecordBatch.model_json_schema()
            }]
        }]
    )

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
                        cat = record.category
                        if counts[cat] < category_target:
                            valid_records.append(record)
                            counts[cat] += 1
                            total_written += 1
                            
                            if total_written >= TARGET_COUNT:
                                break

                    # Write valid records
                    for record in valid_records:
                        json_str = record.model_dump_json(indent=4)
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
        print(f"Total records generated: {total_written}")
        print("Category distribution:")
        for cat in CATEGORIES:
            count = counts.get(cat, 0)
            target = math.ceil(TARGET_COUNT / len(CATEGORIES))
            print(f"- {cat}: {count}/{target} ({count/target:.1%})")

if __name__ == "__main__":
    main()