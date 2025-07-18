#!/usr/bin/env python3
import json
from pathlib import Path

# --- CONFIGURATION ---

# The raw feedback data collected by your cron job.
# This is the INPUT file.
RAW_FEEDBACK_FILE = "data/user_logs.json"

# The clean, formatted file that will be used for fine-tuning.
# This is the OUTPUT file.
FINETUNING_DATA_FILE = "rl_data/high_reward_finetuning_data.jsonl"


def create_finetuning_dataset_from_rewards():
    """
    Reads collected feedback data, filters for high-reward interactions,
    and formats them into a new .jsonl file ready for OpenAI fine-tuning.
    """
    project_root = Path(__file__).parent.parent
    input_file_path = project_root / RAW_FEEDBACK_FILE
    output_file_path = project_root / FINETUNING_DATA_FILE

    # 1. Check if the input file exists
    if not input_file_path.exists():
        print(f"❌ Error: Raw feedback file not found at '{input_file_path}'")
        print("   -> Is your cron job running and creating the feedback file?")
        return

    high_reward_pairs = []
    print(f"Reading raw feedback from '{input_file_path}'...")

    # 2. Read the raw data and filter for good examples
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)

                # The core filtering logic: only keep examples with a positive reward.
                # Using .get() is safer in case a line is missing a "reward" key.
                if data.get("reward", 0) > 0:
                    # 3. Format the data into the required OpenAI chat format
                    formatted_example = {
                        "messages": [
                            {"role": "user", "content": data.get("state")},
                            {"role": "assistant", "content": data.get("action")}
                        ]
                    }
                    high_reward_pairs.append(formatted_example)

            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not decode line {i + 1}. Skipping.")
            except AttributeError:
                print(f"⚠️ Warning: Line {i + 1} might be missing 'state' or 'action' keys. Skipping.")

    # 4. Save the clean, formatted data to the new file
    if high_reward_pairs:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in high_reward_pairs:
                f.write(json.dumps(item) + "\n")

        print(f"\n✅ Success! Created a new fine-tuning file with {len(high_reward_pairs)} high-reward examples.")
        print(f"   -> File saved to: '{output_file_path}'")
    else:
        print("\nℹ️ No new high-reward examples were found in the raw feedback file.")


if __name__ == "__main__":
    create_finetuning_dataset_from_rewards()