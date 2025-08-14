# LLM-Drawing

This repository contains code and data related to LLM-Drawing,focusing on fine-tuning qwen3 4b thinking model for drawing-related tasks, possibly involving SVG generation.

## Repository Structure

```
.DS_Store
Data/
├── qwen3_finetune_dataset.jsonl
Synthetic-Data/
├── des.py
├── descriptions_dataset.json
├── enhanced_dataset_with_mcqs.json
├── mcq.py
├── output_with_svgs.json
├── prompts.py
src/
├── chat_template.py
├── finetuneqwen.ipynb
├── inference.py
```

## File Descriptions

### Data/
- `qwen3_finetune_dataset.jsonl`: A dataset used for fine-tuning the Qwen3 LLM without thinking tokens. # dataset with thinking tokens in the local repo

### Synthetic-Data/
- `des.py`: python file that prompts to gemini 2.5 Pro to generate synthetic dataset along with thinking tokens.
- `descriptions_dataset.json`: dataset only with descriptions.
- `enhanced_dataset_with_mcqs.json`: An enhanced dataset that includes multiple-choice questions (MCQs), genrated based on descriptions.
- `mcq.py`: A Python script  that genrates multiple-choice questions,along with answers helps to evaluate qulity of dataset with TIFA
- `output_with_svgs.json`: A JSON file containing output data along with descriptions, mcq, and SVG code.
- `prompts.py`: Prompts to generate synthetic dataset.

### src/
- `chat_template.py`: A Python script that defines a chat template, Thinking and Non thinkig can be decided here.
- `finetuneqwen.ipynb`: Jupyter Notebook specifically for fine-tuning the Qwen LLM.
- `inference.py`: A Python package script  containing classe for inference SVG (Scalable Vector Graphics) data, submitted as kaggle notebook/

### Other Files
- `.DS_Store`: A hidden file created by macOS to store custom attributes of its containing folder, such as icon positions or background images. It can be safely ignored on non-macOS systems.

## Getting Started


## Potential Use Cases

This project could be used for:

-   **Generating drawings from text descriptions:** Leveraging LLMs to create visual outputs.
-   **Fine-tuning LLMs for creative tasks:** Adapting general-purpose LLMs for specific artistic or design applications.
-   **Synthetic data generation for drawing:** Creating diverse datasets to train and evaluate drawing-focused AI models.
-   **Evaluating LLM performance on visual tasks:** Using MCQs and test scripts to assess how well LLMs handle drawing-related challenges.





