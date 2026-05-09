# ---------------------------------------------------------------------------
# batch_service.py
#
# This file takes a CSV of messages, sends each message to the LLM chain,
# collects the predictions, compares them with the actual labels from the
# dataset, and finally saves the results to a CSV file.
# ---------------------------------------------------------------------------

import io                                   # used to read CSV bytes (uploaded file) like a file
import pandas as pd                         # pandas helps us read/write CSV files easily
from pathlib import Path                    # Path makes file path handling cleaner than strings
from datetime import datetime               # used to add a timestamp to the output file name

# Project imports — these come from other folders inside our app
from app.chain.scam_chain import build_chain
from app.core.config import get_setings
from app.prompts.registry import prompt_registry
from app.schemas.response import BatchResult, BatchSummary, MessageResult


# Load app settings (like default prompt version) from config
settings = get_setings()


# ---------------------------------------------------------------------------
# File path setup
# ---------------------------------------------------------------------------
# DATASET_PATH points to our default CSV file (used when user does not upload one).
# OUTPUTS_DIR is the folder where we save result CSV files.
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "dataset.csv"
OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"

# Make sure the outputs folder exists. If it does not exist, create it.
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Function: load_dataframe
# Purpose : Load the CSV file into a pandas DataFrame.
#           - If the user uploaded a file (file_bytes), use that.
#           - Otherwise, use the default CSV file from DATASET_PATH.
# ---------------------------------------------------------------------------
def load_dataframe(file_bytes: bytes = None) -> pd.DataFrame:

    # Case 1: User uploaded a file (we got its raw bytes)
    if file_bytes:
        # io.BytesIO turns bytes into a file-like object pandas can read
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Case 2: No file uploaded — use the default dataset
    else:
        # First, check if the default file actually exists on disk
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Default dataset not found at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)

    # We need these two columns in the CSV. If any are missing, raise an error.
    required_columns = {"message_text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")

    return df


# ---------------------------------------------------------------------------
# Function: save_results_to_csv
# Purpose : Take the list of MessageResult objects and save them as a CSV file
#           inside the outputs folder. The file name includes the prompt
#           version and a timestamp, so each run creates a new file.
# ---------------------------------------------------------------------------
def save_results_to_csv(results: list[MessageResult], version: str) -> str:

    # Build a timestamp like "20260506_153045" to keep file names unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build the full output file path
    filepath = OUTPUTS_DIR / f"results_{version}_{timestamp}.csv"

    # Convert each MessageResult object into a normal dictionary,
    # because pandas works nicely with a list of dictionaries.
    rows = []
    for r in results:
        row = {
            "message": r.message,
            "actual_label": r.actual_label,
            "predicted_label": r.predicted_label,
            "is_correct": r.is_correct,
            "intent_type": r.intent_type,           # already a string
            "confidence_score": r.confidence_score,
            "reasoning": r.reasoning,
            "prompt_version_used": r.prompt_version_used,
        }
        rows.append(row)

    # Convert the list of dicts into a DataFrame and write it to CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(filepath, index=False, encoding="utf-8")

    # Return the file path as a string so the caller knows where it was saved
    return str(filepath)


# ---------------------------------------------------------------------------
# Function: run_batch
# Purpose : The main function. It:
#           1) Picks a prompt version
#           2) Loads the dataset
#           3) Runs every message through the LLM chain
#           4) Compares LLM predictions with actual labels
#           5) Saves results to CSV
#           6) Returns a summary + per-message results
# ---------------------------------------------------------------------------
async def run_batch(
    file_bytes: bytes = None,
    prompt_version: str = None,
    sample_size: int = None,
) -> BatchResult:

    # -----------------------------------------------------------------------
    # Step 1: Decide which prompt version to use.
    # If the caller did not pass one, fall back to the default from settings.
    # -----------------------------------------------------------------------
    version = prompt_version or settings.default_prompt_version

    # Make sure the prompt version actually exists in our registry
    available_versions = prompt_registry.list_versions()
    if version not in available_versions:
        raise ValueError(
            f"Invalid prompt version: '{version}'. Available: {available_versions}"
        )

    # -----------------------------------------------------------------------
    # Step 2: Load the dataset (either uploaded file or default file)
    # -----------------------------------------------------------------------
    df = load_dataframe(file_bytes)

    # If the caller wants to test on only a few rows, randomly sample them.
    # random_state=42 makes the random pick repeatable (same rows every time).
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Step 3: Build the LLM chain for this prompt version, and prepare input.
    # The chain expects a list of dicts like [{"message": "..."}].
    # -----------------------------------------------------------------------
    chain = build_chain(version)

    inputs = []
    for _, row in df.iterrows():
        inputs.append({"message": row["message_text"]})

    # -----------------------------------------------------------------------
    # Step 4: Run all messages through the chain.
    # max_concurrency=5 means up to 5 messages can be processed at the same time.
    # return_exceptions=True means: if a message fails, return the error
    # instead of crashing the whole batch.
    # -----------------------------------------------------------------------
    scam_results = chain.batch(
        inputs,
        config={"max_concurrency": 5},
        return_exceptions=True,
    )

    # -----------------------------------------------------------------------
    # Step 5: Loop through results and build our final list of MessageResult.
    # Also keep counters so we can build a summary at the end.
    # -----------------------------------------------------------------------
    results = []
    correct = 0
    scam_count = 0
    not_scam_count = 0
    uncertain_count = 0

    # zip pairs each LLM result with the matching row from the DataFrame
    for scam_result, (_, row) in zip(scam_results, df.iterrows()):

        message = row["message_text"]
        actual_label = row["label"]

        # ---- Case A: The chain raised an error for this message ----
        if isinstance(scam_result, Exception):
            error_result = MessageResult(
                message=message,
                actual_label=actual_label,
                predicted_label="Error",
                intent_type="Unknown",
                confidence_score=0.0,
                reasoning=str(scam_result),     # store the error message
                is_correct=False,
                prompt_version_used=version,
            )
            results.append(error_result)
            # Skip the rest of this loop iteration — go to next message
            continue

        # ---- Case B: The chain returned a normal ScamResult ----

        # The label inside ScamResult is an Enum; .value gives us the string
        predicted_label = scam_result.label.value

        # Compare predicted vs actual (lowercase to ignore casing differences)
        is_correct = predicted_label.lower() == actual_label.lower()

        # Update the counters
        if is_correct:
            correct += 1

        if predicted_label == "Scam":
            scam_count += 1
        elif predicted_label == "Not Scam":
            not_scam_count += 1
        else:
            uncertain_count += 1

        # Build the MessageResult object for this row
        single_result = MessageResult(
            message=message,
            actual_label=actual_label,
            predicted_label=predicted_label,
            intent_type=scam_result.intent_type.value,   # Enum → string
            confidence_score=scam_result.confidence_score,
            reasoning=scam_result.reasoning,
            is_correct=is_correct,
            prompt_version_used=version,
        )
        results.append(single_result)

    # -----------------------------------------------------------------------
    # Step 6: Save the results to a CSV file in the outputs folder
    # -----------------------------------------------------------------------
    saved_path = save_results_to_csv(results, version)

    # -----------------------------------------------------------------------
    # Step 7: Build a summary (aggregate numbers) for the whole batch
    # -----------------------------------------------------------------------
    total = len(results)

    # Avoid divide-by-zero: only compute accuracy if total > 0
    if total > 0:
        accuracy = round(correct / total, 4)
    else:
        accuracy = 0.0

    summary = BatchSummary(
        total_processed=total,
        correct_predictions=correct,
        accuracy=accuracy,
        scam_detected=scam_count,
        not_scam_detected=not_scam_count,
        uncertain_detected=uncertain_count,
        prompt_version_used=version,
    )

    # Return both the summary and the detailed per-message results
    return BatchResult(summary=summary, results=results)