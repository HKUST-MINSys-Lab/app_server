import argparse
from pymongo import MongoClient
from datetime import datetime, timedelta
import csv
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Export a MongoDB collection to CSV.")
parser.add_argument("collection", help="The name of the collection to export")
parser.add_argument("date", nargs="?", default=None, help="Optional: date filter in format %Y-%m-%d")
args = parser.parse_args()

# Directory to save the CSV file
root = "/root/ops"

# Connect to MongoDB (adjust the URI if needed)
client = MongoClient('mongodb://localhost:27017/')

# Select the 'app' database and the collection provided by the argument
db = client['app']
col = db[args.collection]

# Build a query based on the presence of a date argument
query = {}
if args.date:
    try:
        # Parse the date string into a datetime object
        start_date = datetime.strptime(args.date, "%Y-%m-%d")
        # Calculate the end of the day (exclusive)
        end_date = start_date + timedelta(days=1)
        query = {"timestamp": {"$gte": start_date, "$lt": end_date}}
    except ValueError as e:
        print(f"Error parsing date: {e}")
        exit(1)

# Retrieve documents from the collection using the query filter (if any)
documents = list(col.find(query))

if not documents:
    print("No documents found in the collection.")
else:
    # Determine all keys present in the documents
    header_keys = set()
    for doc in documents:
        header_keys.update(doc.keys())
    header_keys = list(header_keys)
    header_keys.sort()  # Optional: sort header keys alphabetically

    # Create a CSV filename with the collection name, current timestamp, and date filter (if provided)
    timestamp = datetime.now().strftime('%m%dT%H%M')
    csv_filename = os.path.join(root, "outputs", f"{args.collection}_{timestamp}{'_' + args.date if args.date else ''}.csv")

    # Write data to CSV using DictWriter
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_keys)
        writer.writeheader()
        for doc in documents:
            # Convert _id to string for proper CSV output
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            # Replace newline characters with the escaped sequence \n in all string fields
            for key, value in doc.items():
                if isinstance(value, str):
                    doc[key] = value.replace("\n", "\\n")
            writer.writerow(doc)

    print(f"Data saved to {csv_filename}")
