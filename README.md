# Fine-tuning Llama-3.2-11B-Vision-Instruct for Bank Cheque Information Extraction

## 📌 Overview

This project focuses on fine-tuning the **`unsloth/Llama-3.2-11B-Vision-Instruct`** model to perform **information extraction from bank cheque images**.
The fine-tuned model is trained to extract structured fields such as payer name, account number, amount, bank name, and other relevant cheque details in **JSON format**.

The project uses **Unsloth** to enable efficient fine-tuning with **4-bit quantization (QLoRA)**, significantly reducing GPU memory requirements while maintaining high accuracy.

---

## 🧠 Model

* **Base model**: `unsloth/Llama-3.2-11B-Vision-Instruct`
* **Architecture**: Multimodal (Vision + Language)
* **Fine-tuning method**: QLoRA (4-bit)
* **Task**: Visual Information Extraction (VIE)
* **Output format**: Structured JSON

Example output:

```json
{
  "gt_parse": {
    "cheque_details": [
      {
        "payer_name": "Joseph Cooper",
        "address": "3714 Darlene Ports, Port Davidton, CT 31205",
        "cheque_date": "2024-06-26",
        "payee_name": "Cortez Inc",
        "memo": "Front-line 5th generation hierarchy",
        "amt_in_figures": "5992.9",
        "amt_in_words": "Five Thousand, Nine Hundred And Ninety-Two Dollars and 90/100",
        "routing_number": 98335601,
        "account_number": 7741488526,
        "cheque_number": 568562,
        "bank_name": "JP Morgan Chase & Co."
      }
    ]
  }
}
```

---

## 📂 Dataset

* **Dataset**: `07kamal03/cheque_dataset_bank`
* **Split used**: `train`, `test`
* **Format**:

  * `image`: Bank cheque image (PIL Image)
  * `ground_truth`: JSON string containing structured cheque information

---

## ⚙️ Environment Setup

### 1️⃣ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries:

* `torch`
* `transformers`
* `datasets`
* `unsloth`
* `trl`
* `bitsandbytes`
* `peft`
* `pandas`

---

### 2️⃣ Hardware Requirements

| Configuration  | Recommended                           |
| -------------- | ------------------------------------- |
| GPU            | ≥ 24GB VRAM (RTX 3090 / 4090 / A5000) |
| VRAM (minimum) | 16GB (batch size = 1)                 |
| RAM            | ≥ 32GB                                |
| Precision      | bfloat16                              |
| Quantization   | 4-bit (QLoRA)                         |

---

## 🏋️ Fine-tuning

The model is fine-tuned using **Unsloth's FastVisionModel** with supervised fine-tuning (SFT).

Key components:

* `FastVisionModel`
* `UnslothVisionDataCollator`
* `SFTTrainer` from TRL
* Gradient checkpointing
* 4-bit quantization

The training objective is to teach the model to generate **structured JSON outputs** from cheque images.

---

## 🔍 Inference

Inference is performed using HuggingFace Transformers:

```python
from transformers import MllamaForConditionalGeneration, AutoProcessor

model = MllamaForConditionalGeneration.from_pretrained(
    "Llama-3.2-11B-Vision-Cheque",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained("Llama-3.2-11B-Vision-Cheque")
```

Each cheque image is passed together with the instruction:

```
"Extract information from bank cheque"
```

The model generates a JSON-formatted response.

---

## 📊 Evaluation

The model is evaluated using **Field-level Accuracy** on the test set.

* **Total test samples**: 600
* **Overall field-level accuracy**: ~95%

Evaluation metrics include:

* Exact field match
* Per-field accuracy (payer name, address, amount, etc.)

---

## 📈 Results Summary
| Field          | Correct | Total | Accuracy (%) |
| -------------- | ------- | ----- | ------------ |
| payer_name     | 573     | 600   | 95.50        |
| address        | 556     | 600   | 92.67        |
| cheque_date    | 589     | 600   | 98.17        |
| payee_name     | 571     | 600   | 95.17        |
| memo           | 548     | 600   | 91.33        |
| amt_in_figures | 568     | 600   | 94.67        |
| amt_in_words   | 554     | 600   | 92.33        |
| routing_number | 584     | 600   | 97.33        |
| account_number | 580     | 600   | 96.67        |
| cheque_number  | 586     | 600   | 97.67        |
| bank_name      | 592     | 600   | 98.67        |
---

## 📁 Project Structure

```text
.
├── finetune.py
├── test.py
├── requirements.txt
└── README.md
```

---

## 🚀 Future Work

* Improve robustness for handwritten cheques
* Apply fuzzy matching for free-text fields
* Extend to other financial documents (invoices, receipts)
* Deploy as a REST API using FastAPI

---

## 📜 License

This project is for **academic and research purposes** only.
Please follow the license of the original LLaMA and Unsloth models.

---

## ✉️ Contact

For questions or collaboration:

* **Author**: Tran Khanh
* **Email**: [khanh091103@gmail.com](mailto:khanh091103@gmail.com)
