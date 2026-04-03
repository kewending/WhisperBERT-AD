from datasets import load_dataset
import time
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

train_path = r"data/ADReSS/Train"
test_path = r"data/ADReSS/Test"

AD_dataset_train = load_dataset("audiofolder", data_dir=train_path, split="all")
AD_dataset_test = load_dataset("audiofolder", data_dir=test_path, split="all")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def parse_prediction(response_text):
    """
    Extracts the classification label from Gemini's free-form response.
    Looks for the '- Classification: [...]' line defined in the prompt format.
    Falls back to keyword search if the structured line isn't found.
    Returns 'Dementia', 'Healthy Control', or 'Unknown'.
    """
    if not response_text:
        return "Unknown"

    # Primary: match the structured report line from the prompt
    match = re.search(
        r"-\s*Classification\s*:\s*(Dementia|Healthy Control)",
        response_text,
        re.IGNORECASE
    )
    if match:
        label = match.group(1).strip().title()
        # Normalise casing to exactly match your label strings
        return "Healthy Control" if "healthy" in label.lower() else "Dementia"

    # Fallback: plain keyword scan
    lower = response_text.lower()
    if "healthy control" in lower:
        return "Healthy Control"
    if "dementia" in lower:
        return "Dementia"

    return "Unknown"


def save_results(results, model_name, output_dir="results"):
    """
    Saves two files:
      1. A human-readable .txt report with per-sample details + metrics summary.
      2. A machine-readable .json file with every field for later analysis.
    """
    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "-").replace(":", "-")
    base_name = f"results_{safe_model}_{timestamp}"
    txt_path  = os.path.join(output_dir, base_name + ".txt")
    json_path = os.path.join(output_dir, base_name + ".json")

    # ── Separate known vs unknown predictions ──────────────────────────────
    known   = [r for r in results if r["parsed_prediction"] != "Unknown"]
    unknown = [r for r in results if r["parsed_prediction"] == "Unknown"]

    y_true, y_pred = [], []
    if known:
        y_true = [r["actual"]           for r in known]
        y_pred = [r["parsed_prediction"] for r in known]

    # ── Write .txt report ──────────────────────────────────────────────────
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("GEMINI DEMENTIA DETECTION — RESULTS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model     : {model_name}\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Total     : {len(results)}  |  Evaluated: {len(known)}  |  Unknown: {len(unknown)}\n")
        f.write("=" * 70 + "\n\n")

        # Per-sample block
        f.write("── PER-SAMPLE DETAILS ──\n\n")
        for r in results:
            correct = (
                "✓" if r["parsed_prediction"] == r["actual"]
                else "✗" if r["parsed_prediction"] != "Unknown"
                else "?"
            )
            f.write(f"ID            : {r['id']}\n")
            f.write(f"Ground Truth  : {r['actual']}\n")
            f.write(f"Parsed Label  : {r['parsed_prediction']}  [{correct}]\n")
            f.write(f"Raw Response  :\n{r['prediction']}\n")
            f.write("-" * 70 + "\n\n")

        # Metrics summary
        f.write("── METRICS SUMMARY ──\n\n")
        if known:
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, pos_label="Dementia",
                           average="binary", zero_division=0)
            f.write(f"Accuracy  : {acc:.4f}  ({acc*100:.1f}%)\n")
            f.write(f"F1 Score  : {f1:.4f}  (binary, positive='Dementia')\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, zero_division=0))
            f.write("\nConfusion Matrix  (rows=Actual, cols=Predicted):\n")
            labels = ["Dementia", "Healthy Control"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            header = f"{'':20s}" + "  ".join(f"{l:>15s}" for l in labels)
            f.write(header + "\n")
            for label, row in zip(labels, cm):
                f.write(f"{label:20s}" + "  ".join(f"{v:>15d}" for v in row) + "\n")
        else:
            f.write("No parseable predictions — metrics unavailable.\n")

        if unknown:
            f.write(f"\nUnparseable samples ({len(unknown)}): "
                    + ", ".join(r["id"] for r in unknown) + "\n")

    # ── Write .json file ───────────────────────────────────────────────────
    json_payload = {
        "model": model_name,
        "timestamp": timestamp,
        "total_samples": len(results),
        "evaluated_samples": len(known),
        "unknown_samples": len(unknown),
        "metrics": {},
        "results": results,
    }
    if known:
        json_payload["metrics"] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_dementia": round(
                f1_score(y_true, y_pred, pos_label="Dementia",
                         average="binary", zero_division=0), 4
            ),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved → {txt_path}")
    print(f"✅ JSON saved    → {json_path}")
    return txt_path, json_path


def print_metrics(results):
    """Prints a concise metrics table to stdout."""
    known = [r for r in results if r["parsed_prediction"] != "Unknown"]
    if not known:
        print("No parseable predictions to evaluate.")
        return

    y_true = [r["actual"]            for r in known]
    y_pred = [r["parsed_prediction"] for r in known]

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"Samples evaluated : {len(known)} / {len(results)}")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 (Dementia)     : {f1_score(y_true, y_pred, pos_label='Dementia', average='binary', zero_division=0):.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def process_dataset_with_gemini(gemini_model, dataset_split, start_index, end_index):
    results = []
    dataset_sample = dataset_split.select(range(start_index, end_index))
    num_samples = len(dataset_sample)

    for i in range(num_samples):
        sample = dataset_sample[i]
        audio_data     = sample['audio']['path']
        transcript_text = sample['transcript']
        actual_label   = "Dementia" if sample['label'] == 1 else "Healthy Control"
        sample_id      = sample['id']

        print(f"\n[{i+1}/{num_samples}] Processing ID: {sample_id}...")

        try:
            audio_file = client.files.upload(file=audio_data)
            while audio_file.state == 'PROCESSING':
                time.sleep(2)
                audio_file = client.files.get(name=audio_file.name)

            prompt = f"""
            SYSTEM INSTRUCTION: You are a clinical diagnostic assistant specializing in neurodegenerative diseases.
            TASK: Analyze the provided audio recording and its corresponding transcript to determine the likelihood of dementia.
            
            TRANSCRIPT: "{transcript_text}"
            
            REPORT FORMAT:
            - Classification: [Dementia / Healthy Control]
            - Reasoning: [Briefly explain based on audio and transcript]
            """

            response = client.models.generate_content(
                model=gemini_model,
                contents=[prompt, audio_file]
            )

            raw_text        = response.text
            parsed_label    = parse_prediction(raw_text)

            results.append({
                "id": sample_id,
                "actual": actual_label,
                "prediction": raw_text,
                "parsed_prediction": parsed_label,
            })

            status = "✓" if parsed_label == actual_label else ("?" if parsed_label == "Unknown" else "✗")
            print(f"  Ground Truth : {actual_label}")
            print(f"  Prediction   : {parsed_label}  [{status}]")

        except Exception as e:
            print(f"  Error processing {sample_id}: {e}")
            results.append({
                "id": sample_id,
                "actual": actual_label,
                "prediction": f"ERROR: {e}",
                "parsed_prediction": "Unknown",
            })

        time.sleep(4)

    return results


# ── EXECUTION ──────────────────────────────────────────────────────────────
model = "gemini-3.1-flash-lite-preview" 

test_results = process_dataset_with_gemini(
    model, AD_dataset_test, start_index=0, end_index=48
)

print_metrics(test_results)
save_results(test_results, model_name=model, output_dir=r"results")