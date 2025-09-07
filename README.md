
<h1 align="center">uncensor-GPT-OSS-<code>20b</code></h1>

<div align="center">

 
> A pipeline for uncensoring `openai/gpt-oss-20b`, an open-weight model from OpenAI, using an offline, iterative process with uncensored teacher models (e.g., LLaMA-3 Uncensored, Mistral Uncensored). This project includes tools for prompt collection, synthetic augmentation, automated response generation, data cleaning, LoRA/QLoRA fine-tuning, evaluation, and deployment. The result is a streamlined, uncensored model deployable locally as merged checkpoints or lightweight adapters, optimized with GGUF quantization.

</div>

> [!CAUTION]
> This is intended for **research purposes only** and modifies `openai/gpt-oss-20b` to reduce its built-in censorship, potentially enabling outputs that may be considered **harmful, offensive, or illegal** in some jurisdictions. Users are solely responsible for ensuring compliance with local laws, *OpenAI’s usage policy*, and ethical guidelines. The optional safety filter (`Detoxify`) should be enabled to mitigate risks of generating harmful content, and all outputs must be carefully monitored to prevent misuse. Proceed with caution and at your own risk.

<table align="center">
  <tr>
    <th><h2>Overview</h2></th>
    <th><h2>Install</h2></th>
    <th><h2>Usage</h2></th>
    <th><h2>Examples</h2></th>
  </tr>
  <tr>
    <td>
      <ul>
        <li>Features</li>
        <li>Descriptions</li>
        <li>Tabled Data</li>
        <li>License</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Prerequisites</li>
        <li>Setup</li>
        <li>Directory Structure</li>
        <li>Configure</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Generate Responses</li>
        <li>Clean Data</li>
        <li>Train Model</li>
        <li>Evaluate Model</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Sample Commands</li>
        <li>Code Snippets</li>
        <li>Output Screenshots</li>
        <li>Demo Links</li>
      </ul>
    </td>
  </tr>
</table>



<details>

<summary><h3><- click to view *WIP*</h3></summary>

## Features

- **Prompt Collection**: Combines refusal/jailbreak prompts (ShareGPT, Anthropic HH-RLHF, jailbreak repos) with safe prompts (Alpaca, Dolly) and synthetic variations.
- **Teacher Validation**: Ensures the teacher model (e.g., LLaMA-3 Uncensored) produces direct, high-quality responses (<5% refusals, >0.7 BERTScore).
- **Data Cleaning**: Filters refusals, low-quality, or toxic responses using configurable thresholds in `config/cleaning.yaml`.
- **Fine-Tuning**: Uses LoRA/QLoRA with NF4 quantization for efficient training on consumer hardware.
- **Evaluation**: Measures refusal rate, semantic similarity (BERTScore), reasoning (MMLU/GSM8K), and robustness (DoNotAnswer/XSTest).
- **Feedback Loop**: Iteratively refines the model by targeting failure cases (2–5 cycles, refusal rate <5%).
- **Deployment**: Supports GGUF quantization and Gradio/Streamlit UI for local inference.
- **CLI**: Typer-based CLI for easy execution of pipeline steps (`generate`, `clean`, `train`, `eval`).

## Prerequisites

- **Hardware**:
  - GPU: 16–24GB VRAM (e.g., RTX 3090/4090, A100 for scaling)
  - Disk: 50–100GB for models and datasets
- **Software**:
  - OS: Linux/Windows with CUDA 11.8+
  - Python: 3.10+
- **Dependencies**: See `requirements.txt` for full list.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/icedmoca/uncensor-gpt-oss.git
   cd uncensor-gpt-oss
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models**:
   - **Base Model**: Download `openai/gpt-oss-20b` weights from Hugging Face and place them in `models/base/`:
     ```bash
     huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir models/base/gpt-oss-20b
     ```
   - **Teacher Model**: Place an uncensored model (e.g., `togethercomputer/LLaMA-3-8B-Instruct-uncensored` or `mistral-7b-instruct-uncensored`) in `models/teacher/`. Download from Hugging Face:
     ```bash
     huggingface-cli download togethercomputer/LLaMA-3-8B-Instruct-uncensored --local-dir models/teacher/llama-3-uncensored
     ```
   - Ensure you have a Hugging Face account and access token if required.

4. **Directory Structure**:
   ```text
   uncensor-gpt-oss/
   ├── cli.py
   ├── config/
   │   ├── cleaning.yaml
   │   ├── lora.yaml
   │   ├── safety.yaml
   ├── data/
   │   ├── prompts/
   │   ├── generated/
   │   ├── cleaned/
   ├── models/
   │   ├── base/
   │   ├── teacher/
   │   ├── adapters/
   ├── outputs/
   │   ├── checkpoints/
   │   ├── eval_logs/
   │   ├── validation.log
   │   ├── train_errors.log
   ├── requirements.txt
   ├── README.md
   ```

5. **Configure Settings**:
   - Edit `config/cleaning.yaml` for data cleaning thresholds.
   - Edit `config/lora.yaml` for fine-tuning hyperparameters.
   - Edit `config/safety.yaml` for optional safety filters.

## Usage

Run the pipeline using the Typer CLI (`cli.py`). Available commands:

- **Generate Responses**:
  ```bash
  python cli.py generate --prompts-path data/prompts
  ```
  Uses the teacher model to generate responses for input prompts.

- **Clean Data**:
  ```bash
  python cli.py clean --data-path data/generated
  ```
  Filters and formats responses based on `config/cleaning.yaml`.

- **Train Model**:
  ```bash
  python cli.py train --data-size 10000 --cycles 3
  ```
  Fine-tunes `gpt-oss-20b` with LoRA/QLoRA for 3 cycles (10k pairs).

- **Evaluate Model**:
  ```bash
  python cli.py eval --model-path outputs/checkpoints
  ```
  Runs refusal, similarity, and reasoning tests; logs to `outputs/eval_logs`.

### Example Workflow
```bash
# Download datasets (e.g., Anthropic HH-RLHF, ShareGPT)
python scripts/download_prompts.py
# Generate teacher responses
python cli.py generate
# Clean responses
python cli.py clean
# Train for 3 cycles
python cli.py train --data-size 10000 --cycles 3
# Evaluate results
python cli.py eval
# Deploy with Gradio UI
python scripts/deploy_gradio.py
```

## Pipeline Steps

1. **Environment Setup**: Install dependencies and set up directories.
2. **Collect & Augment Prompts**: Download datasets, add safe prompts (30–50%), and generate synthetic variations (10k–50k pairs).
3. **Validate & Generate Teacher Responses**: Test teacher quality (<5% refusals, >0.7 BERTScore), then generate responses.
4. **Clean Responses**: Filter refusals, short/toxic responses, and deduplicate using `config/cleaning.yaml`.
5. **Build Final Dataset**: Format to Alpaca schema, split 80/20 train/val, track with DVC/Git LFS.
6. **Fine-Tune**: Use LoRA/QLoRA with `config/lora.yaml`, handle errors (OOM).
7. **Evaluate**: Measure refusal rate, BERTScore, MMLU/GSM8K, and DoNotAnswer/XSTest.
8. **Feedback Loop**: Add failed prompts, retrain until refusal <5%.
9. **Deploy**: Quantize to GGUF, run with Gradio/Streamlit, apply optional safety filter.

## Configuration Files

- **cleaning.yaml**:
  ```yaml
  cleaning:
    min_length_general: 50
    min_length_code: 200
    toxicity_threshold: 0.8
    dedup_similarity: 0.95
    bertscore_min: 0.7
  ```

- **lora.yaml**:
  ```yaml
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
  training:
    learning_rate: 1e-4
    warmup_steps: 100
    max_steps: 1000
    batch_size: 8
    fp16: true
  ```

- **safety.yaml**:
  ```yaml
  safety:
    toxicity_threshold: 0.9
  ```

## Expected Results

The following table outlines the anticipated outcomes of the Uncensor-GPT-OSS pipeline when applied to the `openai/gpt-oss-20b` model, based on similar uncensoring efforts (e.g., LLaMA or Mistral fine-tuning). Results depend on dataset quality, teacher model performance, and hardware configuration.

| **Metric**                 | **Details**                                                                 | **Expected Value**                                                                 |
|----------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **Refusal Reduction**      | Percentage of refusal prompts (e.g., jailbreak, sensitive queries) where the model provides direct responses instead of refusing (e.g., "I cannot..."). Measured using a test set of 100-500 prompts from DoNotAnswer/XSTest. | 80-90% after 1-2 cycles (10k-20k pairs); ~95% after 3-5 cycles (20k-50k pairs).     |
| **Training Time**          | Time per fine-tuning cycle using LoRA/QLoRA with NF4 quantization on a 24GB VRAM GPU (e.g., RTX 4090). Assumes 10k prompt-response pairs, 1-3 epochs, batch size 8, and mixed precision (fp16). | ~4-6 hours per cycle for 10k pairs; scales to ~8-12 hours for 20k pairs. Total: 1-2 days for 3-5 cycles. |
| **Output Model**           | Final model format and deployment characteristics. Includes option for merged checkpoints or lightweight LoRA adapters, quantized to GGUF format for efficient inference using llama.cpp. | Uncensored model (merged or adapter-based) with GGUF quantization, achieving <5% refusal rate, maintaining general capabilities (e.g., MMLU/GSM8K scores within 5% of baseline), and supporting fast local inference (~0.5-1s/token on 16GB GPU). |
| **Dataset Size Impact**    | Effect of dataset size on uncensoring effectiveness. Larger datasets improve coverage of edge cases but increase training time. | 10k pairs sufficient for initial results (80% refusal reduction); 20k-50k pairs needed for robust uncensoring (~95% reduction) across diverse domains (ethical, technical, creative). |
| **Reasoning Retention**    | Preservation of general reasoning capabilities post-fine-tuning, measured via MMLU (multi-choice) and GSM8K (math) benchmarks (100 samples each). | Within 5% of baseline `gpt-oss-20b` scores (e.g., ~70% MMLU, ~85% GSM8K), ensured by including 30-50% safe prompts (Alpaca, Dolly) in the dataset. |
| **Inference Performance**  | Speed and resource efficiency of the final model during deployment, using GGUF quantization and tools like Gradio/Streamlit for UI. | ~0.5-1s/token for text generation on a 16GB GPU; supports real-time chat with <500MB VRAM overhead for adapters; Gradio UI latency <2s for typical queries. |

```
The full model name is openai/gpt-oss-20b, a 21-billion-parameter open-weight Mixture-of-Experts (MoE) Transformer model released by OpenAI on August 4, 2025, under the Apache 2.0 license.
It features 24 layers, 32 experts with Top-4 routing per token, and 4-bit MXFP4 quantization, enabling efficient inference on a single 16GB GPU. Designed for advanced reasoning, agentic 
tasks, and tool use (e.g., web browsing, Python execution), it matches or exceeds o3-mini on benchmarks like MMLU, AIME, and HealthBench, but is heavily censored due to OpenAI’s safety 
policies, often refusing sensitive prompts, which makes it a target for uncensoring efforts like this pipeline.
```
> *Target Model*

## License

MIT License. See `LICENSE` for details.
</details>
