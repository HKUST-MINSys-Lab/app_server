import io
import os
import sys
import csv
from pymongo import MongoClient

# 将项目的根目录加入到 sys.path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from src.routes.upload_routes import upload_bp
from src.routes.gpt_routes import gpt_bp
from src.services.generate_summary import generate_summary
from src.services.query_gpt import query
from dotenv import load_dotenv
import pandas as pd
import datetime
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

def get_db():
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client.get_database(os.getenv("MONGODB_DB"))

def create_app():
    # Initialize MongoDB connection
    app.mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
    app.db = app.mongodb_client.get_database(os.getenv("MONGODB_DB"))
    
    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(gpt_bp)
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)

    return app

@app.route('/get_intervention', methods=['POST'])
def get_intervention():
    try:
        # CSV header for reference (not used directly in the logic)
        csv_header = (
            "timestamp,volume,screen_on_ratio,wifi_connected,wifi_ssid,"
            "network_traffic,Rx_traffic,Tx_traffic,stepcount_sensor,"
            "gpsLat,gpsLon,battery,current app,bluetooth devices\n"
        )  # timestamp is index

        user_id = request.json.get("user_id")
        if not user_id:
            return jsonify({"status": "error", "message": "Missing user_id"}), 400

        db = get_db()
        user_collection = db[f"uploads_{user_id}"]

        # Query last 1 hour data
        last_hour = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        # Recommended: use count_documents instead of count() if using a recent PyMongo version.
        if user_collection.count_documents({"timestamp": {"$gt": last_hour}}) == 0:
            return jsonify({"status": "success", "intervention": None}), 200

        last_hour_data = user_collection.find({"timestamp": {"$gt": last_hour}})
        # 统计总量屏时间占比，所有app名称中前三的三个app名称，
        # 出现最多的wifi名称，
        # 总网络数据量，
        # 总步数，
        # Initialize aggregation variables.
        total_screen_ratio = 0.0
        count_ratio = 0
        app_counts = {}
        wifi_counts = {}
        total_network_data = 0.0
        total_steps = 0

        # Iterate over the documents returned by the query.
        for row in last_hour_data:
            # Screen on ratio
            try:
                ratio = float(row.get("screen_on_ratio", 0))
            except Exception:
                ratio = 0.0
            total_screen_ratio += ratio
            count_ratio += 1

            # Count current app occurrences.
            app = row.get("current app", "").strip()
            if app and app != None and app != "" and app != "null":
                app_counts[app] = app_counts.get(app, 0) + 1

            # Count wifi_ssid occurrences.
            wifi = row.get("wifi_ssid", "").strip()
            if wifi:
                wifi_counts[wifi] = wifi_counts.get(wifi, 0) + 1

            # Sum network traffic.
            try:
                network_val = float(row.get("network_traffic", 0))
            except Exception:
                network_val = 0.0
            total_network_data += network_val

            # Sum step count.
            try:
                steps = int(row.get("stepcount_sensor", 0))
            except Exception:
                steps = 0
            total_steps += steps

        # Calculate average screen on ratio.
        avg_screen_on_ratio = total_screen_ratio / count_ratio if count_ratio > 0 else 0.0

        # Determine the top three most frequent apps.
        # Option 1: Only names (current approach)
        # top_apps = [app for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Option 2: Names with usage percentages (if desired)
        total_app_counts = sum(app_counts.values())
        top_apps = [
            {"app": app, "percentage": (count / total_app_counts) * 100 if total_app_counts > 0 else 0}
            for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        # Determine the most frequent wifi_ssid.
        most_frequent_wifi = max(wifi_counts.items(), key=lambda x: x[1])[0] if wifi_counts else None

        # Construct the prompt.
        # NOTE: Verify that each metric is in the intended place.
        prompt = """Assuming you are a health assistant, used to remind and intervene with various smartphone users. I will provide statistical data for the past hour: the percentage of sensor data to total screen activation time per hour is {}, the top three application names and their corresponding usage rates are {} (including system applications, please identify and exclude), the wifi name is {}, the volume percentage is mainly {}, the total network data volume (MB) is {}, and the total number of steps is {}. Please comprehensively analyze the possible behavioral states of users, select the most meaningful one or two aspects of health reminders based on data, and provide friendly and humanized intervention reminders. Must be concise, reasonable. Please return the generated content directly."""
        # You might want to adjust the fourth parameter if total_screen_ratio is not meant to be the sound volume percentage.
        intervention = query(prompt.format(
            avg_screen_on_ratio,
            top_apps,
            most_frequent_wifi,
            total_screen_ratio,  # Consider revising if this is not the intended metric
            total_network_data,
            total_steps
        ), 'gpt-4o-mini')

        return jsonify({"status": "success", "intervention": intervention}), 200

    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/send_intervention_feedback', methods=['POST'])
def send_intervention_feedback():
    try:
        user_id = request.json.get("user_id")
        intervention = request.json.get("intervention")
        interventionRating = request.json.get("interventionRating")
        feedback = request.json.get("feedback")

        db = get_db()
        db.intervention_feedbacks.insert_one({
            "user_id": user_id,
            "intervention": intervention,
            "rating": interventionRating,
            "feedback": feedback,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "message": "successful"}), 200
    except:
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

@app.route('/get_summary', methods=['POST'])
def get_summary():
    user_id = request.json.get("user_id")
    if user_id:
        db = get_db()
        summaryA, summaryB = generate_summary(user_id, db)
        return jsonify({"status": "success", 
                        "summaryA": summaryA,
                        "summaryB": summaryB,
                        }), 200
    else: 
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

@app.route('/send_summary_feedback', methods= ["POST"])
def send_summary_feedback():
    try:
        user_id = request.json.get("user_id")
        summaryA = request.json.get("summaryA")
        summaryB = request.json.get("summaryB")
        chosen = request.json.get("chosen")
        feedbackRating = request.json.get("feedbackRating")
        userSummary = request.json.get("userSummary")
        timeStamp = request.json.get("timestamp")

        feedback_data = {
            "user_id": user_id,
            "summaryA": summaryA,
            "summaryB": summaryB,
            "chosen": chosen,
            "feedbackRating": feedbackRating,
            "userSummary": userSummary,
            "timestamp": timeStamp  # Ensure timeStamp is in an appropriate format (e.g., ISO 8601)
        }

        db = get_db()
        db.summary_feedbacks.insert_one(feedback_data)

        return jsonify({"status": "success", "message": "successful"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/send_weekly_survey', methods= ["POST"])
def send_weekly_survey():
    try: 
        user_id =  request.json.get("user_id")
        phq4 = request.json.get("phq4")
        pss4 = request.json.get("pss4")
        gad7 = request.json.get("gad7")

        db = get_db()
        db.weekly_surveys.insert_one({
            "user_id": user_id,
            "phq4": phq4,
            "pss4": pss4,
            "gad7": gad7,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "message": "successful!"}), 200
    except:
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

@app.route("/upload", methods=['POST'])
def upload():
    user_id = request.json.get("user_id")
    csv_content = request.json.get("csv_content")
    if user_id and csv_content:
        # Correct the CSV header
        csv_header = (
            "timestamp,volume,screen_on_ratio,wifi_connected,wifi_ssid,"
            "network_traffic,Rx_traffic,Tx_traffic,stepcount_sensor,"
            "gpsLat,gpsLon,battery,current app,bluetooth devices,UNK\n"
        )
        csv_content = csv_header + csv_content
        
        csv_io = io.StringIO(csv_content)
        csv_reader = csv.DictReader(csv_io)
        csv_data = []
        for row in csv_reader:
            # Convert from string to integer
            timestamp_ms = int(row["timestamp"])
            # Convert milliseconds to a Python datetime in UTC
            row["timestamp"] = datetime.datetime.utcfromtimestamp(timestamp_ms / 1000.0)
            csv_data.append(row)
        
        # Basic check if CSV data is not empty
        if not csv_data:
            return jsonify({"status": "error", "message": "No valid CSV data provided"}), 400

        db = get_db()
        user_collection = db[f"uploads_{user_id}"]
        user_collection.create_index("timestamp", unique=True)
        
        try:
            result = user_collection.insert_many(csv_data, ordered=False)
        except Exception as e:
            # Handle insertion errors, for example duplicate key errors
            return jsonify({"status": "error", "message": str(e)}), 500
        
        # Return a success message with inserted IDs
        return jsonify({"status": "success", "user_id": user_id}), 200

    else: 
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

@app.route("/upload_ios", methods=['POST'])
def upload_ios():
    # please imatate the upload function above
    user_id = request.json.get("user_id")
    csv_content = request.json.get("csv_content")
    if user_id and csv_content: 
        file_path = "uploads/uploads_ios.csv"
        # Open the file in append mode and write the csv_content
        with open(file_path, 'a') as file:
            file.write(csv_content + '\n')  # Adding a newline for better formatting
        return jsonify({"status": "success", "user_id": user_id, "csv_content": csv_content}), 200
    else: 
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0")
