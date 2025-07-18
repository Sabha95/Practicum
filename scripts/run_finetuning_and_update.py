#!/usr/bin/env python3
import openai
import os
import time
from pathlib import Path

# --- CONFIGURATION ---

# The very first fine-tuned model you created.
# This is used ONLY if the 'latest_gpt_model.txt' file doesn't exist yet.
# It's the "seed" for your entire iterative process.
INITIAL_FINETUNED_MODEL = "ft:gpt-3.5-turbo-0125:personal::Bspa6enW"  # <-- IMPORTANT: Set this correctly

# The path to your high-reward training data.
TRAINING_FILE_PATH = "rl_data/reward.jsonl"

# The file that tracks the lineage of your models.
MODEL_TRACKING_FILE = "models/latest_gpt_model.txt"
os.environ["OPEN_API_KEY"] = ""


def get_model_to_improve() -> str:
    """Gets the latest model ID from the tracking file to use as the base for the next job."""
    project_root = Path(__file__).parent.parent
    model_file = project_root / MODEL_TRACKING_FILE

    if not model_file.exists():
        print(f"Tracking file not found. Using initial seed model: {INITIAL_FINETUNED_MODEL}")
        return INITIAL_FINETUNED_MODEL

    with open(model_file, 'r') as f:
        lines = [line for line in f.read().splitlines() if line]
        if not lines:
            print(f"Tracking file is empty. Using initial seed model: {INITIAL_FINETUNED_MODEL}")
            return INITIAL_FINETUNED_MODEL

    latest_model = lines[-1]
    print(f"Found latest model to improve: {latest_model}")
    return latest_model


def run_automated_iterative_finetuning():
    """
    Identifies the latest model, fine-tunes it with new data, and records the new version.
    """
    project_root = Path(__file__).parent.parent
    training_file_path = project_root / TRAINING_FILE_PATH
    model_file = project_root / MODEL_TRACKING_FILE

    # --- FIX: Added comprehensive error handling and checks ---

    # 1. Securely get API key
    openai.api_key = os.environ.get("OPEN_API_KEY")
    if not openai.api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        return

    # 2. Check if the training data file exists BEFORE trying to upload
    if not training_file_path.exists():
        print(f"❌ Error: Training data file not found at '{training_file_path}'")
        print("   -> Please run 'prepare_finetuning_data.py' first.")
        return

    try:
        # 3. Determine which model to improve
        model_to_fine_tune = get_model_to_improve()

        # 4. Upload the new training data
        print(f"Uploading training data from {training_file_path}...")
        with open(training_file_path, "rb") as f:
            training_file_object = openai.files.create(file=f, purpose="fine-tune")

        # 5. Start the new fine-tuning job based on the previous model
        print(f"Starting fine-tuning job based on model '{model_to_fine_tune}'...")
        job = openai.fine_tuning.jobs.create(
            training_file=training_file_object.id,
            model=model_to_fine_tune
        )
        job_id = job.id
        print(f"   -> Fine-tuning job started with ID: {job_id}")

        # 6. Wait for the job to complete (using poll for simplicity and efficiency)
        print("Waiting for job to complete...")
        new_model_id = ""
        # The poll method waits for the job to complete
        while True:
            job_status = openai.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status
            print(f"   -> Current status: {status} (checking again in 60 seconds)")

            if status == 'succeeded':
                new_model_id = job_status.fine_tuned_model
                print(f"\n✅ Fine-tuning job succeeded!")
                print(f"   -> New improved model ID: {new_model_id}")

                # Record the new model ID
                model_file = project_root / MODEL_TRACKING_FILE
                with open(model_file, 'a', encoding='utf-8') as f:
                    if model_file.stat().st_size > 0:
                        f.write("\n")
                    f.write(new_model_id)

                print(f"📝 New model ID successfully appended to '{MODEL_TRACKING_FILE}'.")
                break  # Exit the while loop

            elif status in ['failed', 'cancelled']:
                print(f"\n❌ Job finished with status: {status}. Check the OpenAI dashboard for details.")
                break  # Exit the while loop

            # Wait for 60 seconds before checking the status again
            time.sleep(60)

            # 7. Record the new model ID
            with open(model_file, 'a') as f:
                # Add a newline before the ID if the file is not empty
                if model_file.stat().st_size > 0:
                    f.write("\n")
                f.write(new_model_id)

            print(f"📝 New model ID successfully appended to '{MODEL_TRACKING_FILE}'.")
            print("\nIMPORTANT: Restart the RASA action server to load the improved model.")

    except openai.APIError as e:
        print(f"❌ An OpenAI API error occurred: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_automated_iterative_finetuning()