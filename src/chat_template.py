import json
from transformers import AutoTokenizer

# Load Qwen3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
# Define system prompt with constraints
SYSTEM_PROMPT = """You are an Expert AI assistant that generates SVG code from natural language descriptions. Generate SVG code while respecting the given constraints.
<constraints>
* Allowed Elements: svg, path, circle, rect, ellipse, line, polyline, polygon, g, linearGradient, radialGradient, stop, defs
* Allowed Attributes: viewBox, width, height, fill, stroke, stroke-width, d, cx, cy, r, x, y, rx, ry, x1, y1, x2, y2, points, transform, opacity
</constraints>"""

# Few-shot examples (embedded in user message)
FEW_SHOT_EXAMPLES = """
<example>
<description>"A rainbow arc with clouds"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
<defs>
<linearGradient id="rainbow" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="red"/>
<stop offset="16%" stop-color="orange"/>
<stop offset="33%" stop-color="yellow"/>
<stop offset="50%" stop-color="green"/>
<stop offset="66%" stop-color="blue"/>
<stop offset="83%" stop-color="indigo"/>
<stop offset="100%" stop-color="violet"/>
</linearGradient>
</defs>
<path d="M40,180 A100,100 0 0,1 216,180" stroke="url(#rainbow)" stroke-width="15" fill="none"/>
<g fill="white" stroke="lightgray" stroke-width="1">
<circle cx="60" cy="190" r="20"/>
<circle cx="80" cy="180" r="18"/>
<circle cx="95" cy="190" r="22"/>
<circle cx="75" cy="200" r="15"/>

<circle cx="180" cy="190" r="20"/>
<circle cx="200" cy="180" r="18"/>
<circle cx="215" cy="190" r="22"/>
<circle cx="195" cy="200" r="15"/> </g>
</svg>
```
</example>

<example>
<description>"An abstract face with geometric shapes"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
<circle cx="128" cy="128" r="100" fill="lightyellow" stroke="orange" stroke-width="2"/>
<circle cx="90" cy="100" r="15" fill="black"/>
<circle cx="166" cy="100" r="15" fill="black"/>
<ellipse cx="128" cy="170" rx="30" ry="10" fill="red"/>
<line x1="88" y1="60" x2="112" y2="70" stroke="black" stroke-width="3"/>
<line x1="168" y1="60" x2="144" y2="70" stroke="black" stroke-width="3"/>
</svg>
```
</example>

<example>
<description>"A simple house with a chimney"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
<rect x="60" y="120" width="140" height="100" fill="brown"/>
<polygon points="60,120 130,60 200,120" fill="red"/>
<rect x="160" y="70" width="20" height="50" fill="gray"/>
<rect x="100" y="150" width="40" height="70" fill="blue"/>
<circle cx="130" cy="185" r="5" fill="yellow"/>
</svg>
```
</example>

<example>
<description>"A flower with six petals"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
<circle cx="128" cy="128" r="25" fill="yellow" stroke="orange" stroke-width="2"/>
<ellipse cx="128" cy="78" rx="20" ry="30" fill="pink" stroke="hotpink" stroke-width="1"/>
<ellipse cx="128" cy="178" rx="20" ry="30" fill="pink" stroke="hotpink" stroke-width="1"/>
<ellipse cx="78" cy="128" rx="30" ry="20" fill="pink" stroke="hotpink" stroke-width="1"/>
<ellipse cx="178" cy="128" rx="30" ry="20" fill="pink" stroke="hotpink" stroke-width="1"/>
<ellipse cx="98" cy="98" rx="25" ry="25" transform="rotate(-45 98 98)" fill="pink" stroke="hotpink" stroke-width="1"/>
<ellipse cx="158" cy="158" rx="25" ry="25" transform="rotate(-45 158 158)" fill="pink" stroke="hotpink" stroke-width="1"/>
</svg>
```
</example>

<example>
<description>"A simple clock face showing 10:10"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
<circle cx="128" cy="128" r="100" fill="white" stroke="black" stroke-width="3"/>
<circle cx="128" cy="128" r="5" fill="black"/>
<line x1="128" y1="128" x2="90" y2="90" stroke="black" stroke-width="4"/>
<line x1="128" y1="128" x2="170" y2="90" stroke="black" stroke-width="2"/>
<circle cx="128" cy="50" r="5" fill="black"/>
<circle cx="128" cy="206" r="5" fill="black"/>
<circle cx="50" cy="128" r="5" fill="black"/>
<circle cx="206" cy="128" r="5" fill="black"/>
<circle cx="67" cy="67" r="5" fill="black"/>
<circle cx="189" cy="67" r="5" fill="black"/>
<circle cx="67" cy="189" r="5" fill="black"/>
<circle cx="189" cy="189" r="5" fill="black"/>
</svg>
```
</example>
"""
INPUT_JSON_FILE = "output_with_svgs.json"
OUTPUT_JSONL_FILE = "qwen3_finetune_dataset.jsonl"

def main():
# Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    qwen3_chat_template = """
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' }}
        <think>

        </think>

        {{- message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}

"""

    tokenizer.chat_template = qwen3_chat_template

    # Load dataset
    with open(INPUT_JSON_FILE, "r") as f:
        data = json.load(f)

    records = data["records"]

    # Open output file
    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as out_f:
        for item in records:
            description = item["description"]
            svg_code = item["svg"].strip()

            # Build conversation in dictionary format
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{FEW_SHOT_EXAMPLES}\n\nPlease ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints.\nFocus on a clear and concise representation of the input description within the given limitations.\nAlways give the complete SVG code with nothing omitted. Never use an ellipsis.\n\nHere is the description:\n<description>\"{description}\"</description>\n```svg\n<svg viewBox=\"0 0 256 256\" width=\"256\" height=\"256\">\n```"
                },
                {
                    "role": "assistant",
                    "content": f"""Here is the SVG code:```svg\n{svg_code}\n```"""
                }
            ]

            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

            # Write to .jsonl
            out_f.write(json.dumps({"text": formatted_text}, ensure_ascii=False) + "\n")

    print(f"✅ Dataset saved to {OUTPUT_JSONL_FILE}")

if __name__ == "__main__":
    main()