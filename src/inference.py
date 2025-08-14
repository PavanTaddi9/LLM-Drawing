#| export

num_attempt = 6

class Model:
    def __init__(self):
        self.model_path = kagglehub.model_download('..finetune/qwen-lm/qwen-3/transformers/8b')
        self.llm = vllm.LLM(
            self.model_path,
            quantization=None,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=5120,
            disable_log_stats=True
        )
        self.sampling_params = vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_k=20,  # Float that controls the cumulative probability of the top tokens to consider.
            top_p=0.8,
            temperature=0.7,  # randomness of the sampling
            repetition_penalty=1.05,
            seed=777, # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=1024,  # Maximum number of tokens to generate per output sequence.
        )
        self.tokenizer = self.llm.get_tokenizer()
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
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90

    # You could try increasing `max_new_tokens`
    def predict(self, description: str, max_new_tokens=1024) -> str:
        def apply_template(prompt, tokenizer):
            messages = [
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking = False)
            return text
        
        def parse_svg_from_response(response):
            matchs = re.findall(r'<svg.*?</svg>', response, re.S)
            if matchs:
                return matchs[-1].strip()
            else:
                return ''
        
        def check_svg_valid(svg):
            try:
                cairosvg.svg2png(bytestring=svg.encode('utf-8'))
                return True
            except:
                return False
        
        def generate_svg():
            try:
                prompt = self.prompt_template.format(description)
                inputs = [apply_template(prompt, self.tokenizer)] * num_attempt
                responses = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
                responses = [x.outputs[0].text for x in responses]
                svgs = [parse_svg_from_response(x) for x in responses]
                # use the first valid svg
                choosen_svg = None
                for svg in svgs:
                    if check_svg_valid(svg):
                        svg = self.enforce_constraints(svg)
                        if check_svg_valid(svg):
                            choosen_svg = svg
                            break
                
                assert choosen_svg is not None
                return svg

            except Exception as e:
                logging.error('Exception during SVG generation: %s', e)
                return self.default_svg
        
        return generate_svg()

        # # Execute SVG generation in a new thread to enforce time constraints
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #     future = executor.submit(generate_svg)
        #     try:
        #         return future.result(timeout=self.timeout_seconds)
        #     except concurrent.futures.TimeoutError:
        #         logging.warning("Prediction timed out after %s seconds.", self.timeout_seconds)
        #         return self.default_svg
        #     except Exception as e:
        #         logging.error(f"An unexpected error occurred: {e}")
        #         return self.default_svg

    def enforce_constraints(self, svg_string: str) -> str:
        """Enforces constraints on an SVG string, removing disallowed elements
        and attributes.

        Parameters
        ----------
        svg_string : str
            The SVG string to process.

        Returns
        -------
        str
            The processed SVG string, or the default SVG if constraints
            cannot be satisfied.
        """
        logging.info('Sanitizing SVG...')

        try:
            parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
            root = etree.fromstring(svg_string, parser=parser)
        except etree.ParseError as e:
            logging.error('SVG Parse Error: %s. Returning default SVG.', e)
            return self.default_svg
    
        elements_to_remove = []
        for element in root.iter():
            tag_name = etree.QName(element.tag).localname
    
            # Remove disallowed elements
            if tag_name not in self.constraints.allowed_elements:
                elements_to_remove.append(element)
                continue  # Skip attribute checks for removed elements
    
            # Remove disallowed attributes
            attrs_to_remove = []
            for attr in element.attrib:
                attr_name = etree.QName(attr).localname
                if (
                    attr_name
                    not in self.constraints.allowed_elements[tag_name]
                    and attr_name
                    not in self.constraints.allowed_elements['common']
                ):
                    attrs_to_remove.append(attr)
    
            for attr in attrs_to_remove:
                logging.debug(
                    'Attribute "%s" for element "%s" not allowed. Removing.',
                    attr,
                    tag_name,
                )
                del element.attrib[attr]
    
            # Check and remove invalid href attributes
            for attr, value in element.attrib.items():
                 if etree.QName(attr).localname == 'href' and not value.startswith('#'):
                    logging.debug(
                        'Removing invalid href attribute in element "%s".', tag_name
                    )
                    del element.attrib[attr]

            # Validate path elements to help ensure SVG conversion
            if tag_name == 'path':
                d_attribute = element.get('d')
                if not d_attribute:
                    logging.warning('Path element is missing "d" attribute. Removing path.')
                    elements_to_remove.append(element)
                    continue # Skip further checks for this removed element
                # Use regex to validate 'd' attribute format
                path_regex = re2.compile(
                    r'^'  # Start of string
                    r'(?:'  # Non-capturing group for each command + numbers block
                    r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                    r'\s*'  # Optional whitespace after command
                    r'(?:'  # Non-capturing group for optional numbers
                    r'-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?'  # First number
                    r'(?:[\s,]+-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)*'  # Subsequent numbers with mandatory separator(s)
                    r')?'  # Numbers are optional (e.g. for Z command)
                    r'\s*'  # Optional whitespace after numbers/command block
                    r')+'  # One or more command blocks
                    r'\s*'  # Optional trailing whitespace
                    r'$'  # End of string
                )
                if not path_regex.match(d_attribute):
                    logging.warning(
                        'Path element has malformed "d" attribute format. Removing path.'
                    )
                    elements_to_remove.append(element)
                    continue
                logging.debug('Path element "d" attribute validated (regex check).')
        
        # Remove elements marked for removal
        for element in elements_to_remove:
            if element.getparent() is not None:
                element.getparent().remove(element)
                logging.debug('Removed element: %s', element.tag)

        try:
            cleaned_svg_string = etree.tostring(root, encoding='unicode')
            return cleaned_svg_string
        except ValueError as e:
            logging.error(
                'SVG could not be sanitized to meet constraints: %s', e
            )
            return self.default_svg

import kaggle_evaluation
kaggle_evaluation.test(Model)
