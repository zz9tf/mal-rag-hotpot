import sys
sys.path.append("./utils")
import json
from utils.configs import load_configs
import argparse
from utils.hotpot_database import HotpotDatabaseHandler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="insert", dest="")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.action == "insert":
        config = load_configs()

        # Load HotpotQA data
        print(f"Loading data from {config['HotpotDatabase']['data_path']}...")
        with open(config['HotpotDatabase']['data_path'], 'r', encoding='utf-8') as f:
            hotpot_data = json.load(f)

        # Initialize database handler
        db_hotpot = HotpotDatabaseHandler(config['HotpotDatabase']['db_path'])

        # Insert data with optimized bulk insertion
        stats = db_hotpot.bulk_insert_hotpot_qa_data(
            hotpot_data, 
            question_type='train',
            batch_size=2000  # Adjust based on your system's memory
        )

        print(f"Inserted {stats['content_items_inserted']} content items")
        print(f"Inserted {stats['questions_inserted']} questions")
        print(f"Skipped {stats['skipped_questions']} existing questions")
        print(f"Encountered {stats['errors']} errors")
    
    elif args.action == "embedding":
        # embedding the content data and question text in the database and insert the embeddings into the embedding database
if __name__ == "__main__":
    main()