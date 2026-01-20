import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import pandas as pd
import json

# Load dataset và model
dataset = load_dataset("07kamal03/cheque_dataset_bank", split = "test") 

model_id ="/root/finetune_model/Llama-3.2-11B-Vision-Cheque"

model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)


# Dự đoán
def extract_json_string(text: str):
    """
    Trích JSON string từ output model
    (giữ nguyên string, không json.loads)
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    return text[start:end+1]

predict_rows = []
for idx, sample in enumerate(dataset):
    image = sample["image"]

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract information from bank cheque"}
                ]
            }
        ]
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024
        )

    decoded = processor.decode(
        output[0],
        skip_special_tokens=False
    )

    json_str = extract_json_string(decoded)

    if json_str is None:
        print(f"[WARN] {idx}: cannot extract JSON")
        json_str = ""

    predict_rows.append({
        "predict": json_str
    })

# Đánh giá từng field

df_eval = pd.DataFrame({
    "ground_truth": dataset["ground_truth"],
    "predict": [row["predict"] for row in predict_rows]
})

def safe_json_load(s):
    try:
        return json.loads(s)
    except:
        return None

df_eval["gt_json"] = df_eval["ground_truth"].apply(safe_json_load)
df_eval["pred_json"] = df_eval["predict"].apply(safe_json_load)

def get_cheque_details(j):
    try:
        return j["gt_parse"]["cheque_details"][0]
    except:
        return {}

df_eval["gt_details"] = df_eval["gt_json"].apply(get_cheque_details)
df_eval["pred_details"] = df_eval["pred_json"].apply(get_cheque_details)

FIELDS = [
    "payer_name",
    "address",
    "cheque_date",
    "payee_name",
    "memo",
    "amt_in_figures",
    "amt_in_words",
    "routing_number",
    "account_number",
    "cheque_number",
    "bank_name",
]

def normalize_value(v):
    if v is None:
        return ""
    return str(v).strip().lower()

field_results = []

for field in FIELDS:
    gt_vals = df_eval["gt_details"].apply(
        lambda x: normalize_value(x.get(field))
    )
    pred_vals = df_eval["pred_details"].apply(
        lambda x: normalize_value(x.get(field))
    )

    match = (gt_vals == pred_vals)

    field_results.append({
        "field": field,
        "correct": match.sum(),
        "total": len(match),
        "accuracy": match.mean()
    })

field_accuracy_df = pd.DataFrame(field_results)
field_accuracy_df
