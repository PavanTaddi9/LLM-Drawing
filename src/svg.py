import os
import json
import re
import logging
import time
from typing import Optional, List
from lxml import etree
import google.generativeai as genai
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVGConstraints:
    """Contains allowed elements and attributes"""
    def __init__(self):
        self.allowed_elements = {
            'common': ['viewBox', 'width', 'height', 'fill', 'stroke', 
                      'stroke-width', 'transform', 'opacity'],
            'svg': [],
            'path': ['d'],
            'circle': ['cx', 'cy', 'r'],
            'rect': ['x', 'y', 'width', 'height', 'rx', 'ry'],
            'ellipse': ['cx', 'cy', 'rx', 'ry'],
            'line': ['x1', 'y1', 'x2', 'y2'],
            'polyline': ['points'],
            'polygon': ['points'],
            'g': [],
            'linearGradient': ['id', 'x1', 'y1', 'x2', 'y2'],
            'radialGradient': ['id', 'cx', 'cy', 'r'],
            'stop': ['offset', 'stop-color'],
            'defs': []
        }

class SVGGenerator:
    def __init__(self):
        self.prompt_template = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>
<example>
<description>"A green triangle with a yellow star inside"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <polygon points="128,40 216,200 40,200" fill="green"/>
  <polygon points="128,80 143,120 188,120 152,146 166,190 128,164 90,190 104,146 68,120 113,120" fill="yellow"/>
</svg>
```
</example>
<example>
<description>"Concentric circles with gradient colors"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <defs>
    <radialGradient id="grad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="purple" />
      <stop offset="100%" stop-color="pink" />
    </radialGradient>
  </defs>
  <circle cx="128" cy="128" r="100" fill="url(#grad)" opacity="0.8"/>
  <circle cx="128" cy="128" r="80" fill="lightblue" opacity="0.6"/>
  <circle cx="128" cy="128" r="60" fill="teal" opacity="0.5"/>
  <circle cx="128" cy="128" r="40" fill="navy" opacity="0.4"/>
  <circle cx="128" cy="128" r="20" fill="white"/>
</svg>
```
</example>
<example>
<description>"A cloud and sun scene"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="180" cy="80" r="40" fill="yellow" stroke="orange" stroke-width="2"/>
  <g fill="white" stroke="lightgray" stroke-width="1">
    <circle cx="80" cy="120" r="30"/>
    <circle cx="110" cy="110" r="25"/>
    <circle cx="130" cy="120" r="28"/>
    <circle cx="100" cy="140" r="25"/>
  </g>
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
<description>"A simple flower with six petals"</description>
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
    <circle cx="195" cy="200" r="15"/>
  </g>
</svg>
```
</example>
<example>
<description>"A simple cartoon robot face"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <rect x="58" y="60" width="140" height="150" rx="10" ry="10" fill="silver" stroke="gray" stroke-width="2"/>
  <rect x="88" y="210" width="20" height="20" fill="gray"/>
  <rect x="148" y="210" width="20" height="20" fill="gray"/>
  <rect x="78" y="90" width="40" height="30" fill="lightblue" stroke="blue" stroke-width="1"/>
  <rect x="138" y="90" width="40" height="30" fill="lightblue" stroke="blue" stroke-width="1"/>
  <rect x="88" y="150" width="80" height="20" fill="black"/>
  <rect x="88" y="150" width="10" height="20" fill="red"/>
  <rect x="108" y="150" width="10" height="20" fill="red"/>
  <rect x="128" y="150" width="10" height="20" fill="red"/>
  <rect x="148" y="150" width="10" height="20" fill="red"/>
  <circle cx="58" cy="60" r="10" fill="red"/>
  <circle cx="198" cy="60" r="10" fill="red"/>
</svg>
```
</example>
<example>
<description>"A simple mountain landscape with sun"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <rect x="0" y="0" width="256" height="256" fill="skyblue"/>
  <circle cx="200" cy="50" r="30" fill="yellow" stroke="orange" stroke-width="2"/>
  <polygon points="0,256 100,120 150,180 256,256" fill="green"/>
  <polygon points="120,256 180,100 256,256" fill="darkgreen"/>
  <polygon points="0,256 60,150 150,256" fill="forestgreen"/>
  <polygon points="0,190 80,120 180,190 200,170 256,190 256,256 0,256" fill="saddlebrown"/>
  <rect x="0" y="190" width="256" height="66" fill="saddlebrown"/>
</svg>
```
</example>

Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""
        self.constraints = SVGConstraints()
        self.default_svg = '''<svg viewBox="0 0 256 256" width="256" height="256">
    <rect width="256" height="256" fill="white"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="red">
        Invalid SVG
    </text>
</svg>'''
        self.max_retries = 3
        self.retry_delay = 2
        self.api_delay = 1.5

    def configure_api(self):
        """Set up Gemini API credentials"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)

    def enforce_constraints(self, svg_string: str) -> str:
        """Enforces constraints on SVG string"""
        try:
            # Remove XML declarations and sanitize
            svg_string = re.sub(r'<\?xml.*?\?>', '', svg_string, flags=re.DOTALL)
            
            parser = etree.XMLParser(
                remove_blank_text=True,
                remove_comments=True,
                resolve_entities=False
            )
            root = etree.fromstring(svg_string, parser=parser)
        except etree.ParseError as e:
            logger.error('SVG Parse Error: %s', e)
            return self.default_svg

        elements_to_remove = []
        for element in root.iter():
            tag_name = etree.QName(element.tag).localname

            # Remove disallowed elements
            if tag_name not in self.constraints.allowed_elements:
                elements_to_remove.append(element)
                continue

            # Remove disallowed attributes
            allowed_attrs = (
                self.constraints.allowed_elements.get(tag_name, []) +
                self.constraints.allowed_elements['common']
            )
            
            for attr in list(element.attrib.keys()):
                attr_name = etree.QName(attr).localname
                if attr_name not in allowed_attrs:
                    del element.attrib[attr]
                    logger.debug('Removed attribute: %s from %s', attr_name, tag_name)

                # Validate href attributes
                if attr_name == 'href' and (not element.attrib[attr].startswith('#') or len(element.attrib[attr]) > 100):
                    del element.attrib[attr]
                    logger.debug('Removed invalid href in %s', tag_name)

            # Validate path elements
            if tag_name == 'path':
                d_attribute = element.get('d', '')
                if not self.validate_path(d_attribute):
                    elements_to_remove.append(element)
                    logger.warning('Removed invalid path element')

        # Remove disallowed elements
        for element in elements_to_remove:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

        try:
            cleaned_svg = etree.tostring(root, encoding='unicode')
            return self._sanitize_svg(cleaned_svg)
        except Exception as e:
            logger.error('SVG serialization failed: %s', e)
            return self.default_svg

    def validate_path(self, d_attribute: str) -> bool:
        """Validate path 'd' attribute using regex with fallback"""
        try:
            # Strict validation
            path_regex = re.compile(
                r'^'
                r'(?:'
                    r'[MmZzLlHhVvCcSsQqTtAa]'  # Command
                    r'\s*'  # Optional whitespace
                    r'(?:'  # Parameters group
                        r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'  # Number with optional exponent
                        r'(?:[\s,]+-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)*'  # Subsequent numbers
                    r')?'  # Parameters are optional
                    r'\s*'  # Optional trailing whitespace
                r')+'  # One or more command sequences
                r'$',
                re.ASCII
            )
            return bool(path_regex.match(d_attribute))
        except re.error:
            # Fallback lenient check
            return bool(re.match(r'^[MmZzLlHhVvCcSsQqTtAa].*', d_attribute))

    def _sanitize_svg(self, svg_string: str) -> str:
        """Additional security sanitization"""
        # Remove risky elements/attributes
        svg_string = re.sub(r'on\w+=".*?"', '', svg_string)
        svg_string = re.sub(r'<script.*?</script>', '', svg_string, flags=re.DOTALL)
        svg_string = re.sub(r'<!DOCTYPE.*?>', '', svg_string, flags=re.DOTALL)
        return svg_string

    def generate_valid_svg(self, description: str, model, attempt=1) -> str:
        """Generate SVG with validation and retries"""
        if attempt > self.max_retries:
            logger.warning('Max retries reached for: %s', description)
            return self.default_svg

        try:
            response = model.generate_content(
                self.prompt_template.format(description),
                safety_settings={
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                }
            )
            
            # Extract SVG code safely
            raw_svg = response.text
            svg_match = re.search(r'(<svg.*?</svg>)', raw_svg, re.DOTALL)
            if not svg_match:
                raise ValueError("No valid SVG code found in response")
                
            initial_svg = svg_match.group(1).strip()
            sanitized_svg = self.enforce_constraints(initial_svg)
            
            # Final validation
            if sanitized_svg == self.default_svg or not self.validate_compliance(sanitized_svg):
                raise ValueError("SVG failed final compliance check")
                
            return sanitized_svg
            
        except Exception as e:
            logger.warning('Attempt %d failed: %s', attempt, str(e))
            time.sleep(self.retry_delay * attempt)
            return self.generate_valid_svg(description, model, attempt+1)

    def validate_compliance(self, svg: str) -> bool:
        """Final XML validation"""
        try:
            etree.fromstring(svg)
            return True
        except Exception as e:
            logger.error('Final validation failed: %s', e)
            return False

    def process_dataset(self, input_file: str, output_file: str):
        """Process dataset with dynamic JSON writing"""
        self.configure_api()
        model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")

        with open(input_file, 'r') as f:
            dataset = json.load(f)

        total = len(dataset['records'])
        for idx, record in enumerate(dataset['records']):
          record['id'] = idx+1
        processed_count = 0

        try:
            with open(output_file, 'w') as f:
                f.write('{\n  "records": [\n')
                first_entry = True
                
                for i, record in enumerate(dataset['records']):
                    try:
                        logger.info('Processing %d/%d: %s', i+1, total, record['id'])
                        sanitized_svg = self.generate_valid_svg(record['description'], model)
                        
                        # Create output record
                        output_record = {
                            **record,
                            'svg': sanitized_svg
                        }
                        
                        # Write to file immediately
                        if not first_entry:
                            f.write(',\n')
                        else:
                            first_entry = False
                            
                        json_str = json.dumps(
                            output_record, 
                            indent=4, 
                            ensure_ascii=False
                        )
                        f.write(f'    {json_str}')
                        f.flush()  # Force write to disk
                        
                        processed_count += 1
                        time.sleep(self.api_delay)
                        
                    except Exception as e:
                        logger.error('Failed to process record %s: %s', record['id'], str(e))
                        continue

        finally:
            # Properly close the JSON array
            with open(output_file, 'a') as f:
                f.write('\n  ]\n}')

        logger.info(f"Successfully processed {processed_count}/{total} records")

if __name__ == "__main__":
    generator = SVGGenerator()
    generator.process_dataset("descriptions_dataset.json", "output_with_svgs.json")