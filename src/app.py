import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import io
import os
import sys
import csv
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

import threading
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ios_push import send_silent_push


from flask import Flask, request, jsonify
from src.routes.upload_routes import upload_bp
from src.routes.gpt_routes import gpt_bp
from src.services.generate_summary import generate_summary
from src.services.query_gpt import query
from dotenv import load_dotenv
import pandas as pd
import datetime
import logging
from src.routes.ios_push_routes import ios_push_bp

from services.mq import send_to_queue

log_file = '/data/log/app_output.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

# Console handler: prints INFO and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler: saves WARNING and above
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.propagate = False

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
    app.register_blueprint(ios_push_bp)
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)

    @app.before_request
    def log_request_info():
        try:
            if request.path == "/upload_imu":
                # 针对 /upload_imu 接口，只记录 csv_content 的长度信息，但不修改实际数据
                json_data = request.get_json(silent=True)
                if json_data and "csv_content" in json_data:
                    log_json = dict(json_data)  # 做一份拷贝，避免修改原始数据
                    csv_length = len(log_json["csv_content"])
                    log_json["csv_content"] = f"[{csv_length} characters]"
                    req_body = str(log_json)
                else:
                    req_body = request.get_data(as_text=True)
            else:
                req_body = request.get_data(as_text=True)
        except Exception as e:
            req_body = "Error reading request body"
            logger.exception("Error when reading request body")
        logger.info(f"Received request: method={request.method}, url={request.url}, body={req_body}")

    @app.after_request
    def log_response_info(response):
        logger.info(f"Response: status={response.status}")
        return response

    @app.errorhandler(Exception)
    def handle_exceptions(e):
        logger.exception("Unhandled Exception occurred")
        return jsonify({"status": "error", "message": str(e)}), 500

    logger.info("应用create_app()执行完毕")
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

@app.route("/upload_imu", methods=['POST'])
def upload_imu():
    user_id = request.json.get("user_id")
    csv_content = request.json.get("csv_content")
    
    if not user_id:
        return jsonify({"status": "error", "message": "no user_id"}), 400
    if not csv_content:
        return jsonify({"status": "error", "message": "no csv_content"}), 400

    # 定义预期的表头，不要重复添加
    expected_header = "timestamp,acc_X,acc_Y,acc_Z,gyro_X,gyro_Y,gyro_Z,mag_X,mag_Y,mag_Z"
    if not csv_content.startswith(expected_header):
        csv_header = expected_header + "\n"
        csv_content = csv_header + csv_content
    
    csv_io = io.StringIO(csv_content)
    csv_reader = csv.DictReader(csv_io)
    csv_data = []
    for row in csv_reader:
        # 如果遇到表头或其他非数字数据则跳过
        timestamp_str = row["timestamp"]
        # Convert milliseconds to a Python datetime in UTC
        if timestamp_str.isdigit():
            if len(timestamp_str) == 13:
                ts = int(timestamp_str) / 1000.0  # 毫秒
            elif len(timestamp_str) == 16:
                ts = int(timestamp_str) / 1000000.0  # 微秒
            else:
                ts = int(timestamp_str) / 1000.0
            row["timestamp"] = datetime.datetime.utcfromtimestamp(ts)
        else:
            row["timestamp"] = datetime.datetime.fromisoformat(timestamp_str)
        csv_data.append(row)
    
    # Basic check if CSV data is not empty
    if not csv_data:
        return jsonify({"status": "error", "message": "No valid CSV data provided"}), 400

    db = get_db()
    imu_user_collection = db[f"imu_{user_id}"]
    
    # 优化1：在应用启动时或第一次使用时创建索引，避免每次请求都调用
    if "timestamp_1" not in imu_user_collection.index_information():
        imu_user_collection.create_index("timestamp", unique=True)

    # 获取所有待插入数据的 timestamp 列表
    incoming_timestamps = [row["timestamp"] for row in csv_data]

    # 优化2：使用 find 查询，返回 projection 仅包含 timestamp 字段，转换为 set 后过滤
    cursor = imu_user_collection.find(
        {"timestamp": {"$in": incoming_timestamps}},
        {"timestamp": 1, "_id": 0}
    )
    existing_timestamps = {doc["timestamp"] for doc in cursor}
    filtered_data = [row for row in csv_data if row["timestamp"] not in existing_timestamps]

    try:
        if filtered_data:
            result = imu_user_collection.insert_many(filtered_data, ordered=False)
        else:
            logger.info("No new data to insert after filtering duplicates.")
    except BulkWriteError as bwe:
        logger.warning(f"Error inserting data for user {user_id}: {bwe.details}")
        return jsonify({"status": "error", "message": "Insertion failed due to duplicate keys."}), 500
    except Exception as e:
        logger.warning(f"Error inserting data for user {user_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    # 将数据发送到 RabbitMQ 队列进行异步处理
    try:
        send_to_queue({"user_id": user_id, "data": filtered_data})
    except Exception as e:
        logging.error("Send message failed: %s", e)
        # 可根据需求返回错误或继续返回 success 状态
        return jsonify({"status": "error", "message": "Failed to send to queue"}), 500

    # Return a success message with inserted IDs
    return jsonify({"status": "success", "user_id": user_id}), 200

@app.route("/upload", methods=['POST'])
def upload():
    user_id = request.json.get("user_id")
    csv_content = request.json.get("csv_content")
    if user_id and csv_content:
        # Correct the CSV header
        csv_header = (
            "timestamp,volume,screen_on_ratio,wifi_connected,wifi_ssid,"
            "network_traffic,Rx_traffic,Tx_traffic,stepcount_sensor,"
            "gpsLat,gpsLon,battery,current app,bluetooth devices\n"
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
        existing_timestamps = user_collection.distinct("timestamp", {
            "timestamp": {"$in": [row["timestamp"] for row in csv_data]}
        })
        csv_data = [row for row in csv_data if row["timestamp"] not in existing_timestamps]
        
        try:
            if csv_data:  
                result = user_collection.insert_many(csv_data, ordered=False)
            else:
                logger.info("No new data to insert after filtering duplicates.")
        except BulkWriteError as bwe:
            logger.warning(f"Error inserting data for user {user_id}: {bwe.details}")
            return jsonify({"status": "error", "message": "Insertion failed due to duplicate keys."}), 500
        except Exception as e:
            logger.warning(f"Error inserting data for user {user_id}: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
        
        # Return a success message with inserted IDs
        return jsonify({"status": "success", "user_id": user_id}), 200

    else: 
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

@app.route("/upload_ios", methods=['POST'])
def upload_ios():
    user_id = request.json.get("user_id")
    csv_content = request.json.get("csv_content")
    if user_id and csv_content:
        # Correct the CSV header
        csv_header = (
            "timestamp,volume,screen_on_ratio,wifi_connected,wifi_ssid,"
            "network_traffic,Rx_traffic,Tx_traffic,stepcount_sensor,"
            "gpsLat,gpsLon,battery,current app,bluetooth devices,UNK\n" #todo:: we can remove the last occupied column
        )
        # IMU headers: ,accX,accY,accZ,gyroX,gyroY,gyroZ,magneticFieldX,magneticFieldY,magneticFieldZ


        csv_content = csv_header + csv_content
        csv_io = io.StringIO(csv_content)
        csv_reader = csv.DictReader(csv_io)
        csv_data = []
        for row in csv_reader:
            row = {key: value for key, value in row.items() if key is not None and key.strip() != ""}
            timestamp_str = row.get("timestamp", "")
            if not timestamp_str:
                logger.warning("Missing timestamp in row, skipping row")
                continue
            if timestamp_str.isdigit():
                if len(timestamp_str) == 13:
                    ts = int(timestamp_str) / 1000.0  # 毫秒
                elif len(timestamp_str) == 16:
                    ts = int(timestamp_str) / 1000000.0  # 微秒
                else:
                    ts = int(timestamp_str) / 1000.0
                row["timestamp"] = datetime.datetime.utcfromtimestamp(ts)
            else:
                row["timestamp"] = datetime.datetime.fromisoformat(timestamp_str)
            csv_data.append(row)
        
        if not csv_data:
            return jsonify({"status": "error", "message": "No valid CSV data provided"}), 400

        db = get_db()
        user_collection = db[f"uploads_{user_id}"]
        user_collection.create_index("timestamp", unique=True)
        
        existing_timestamps = user_collection.distinct("timestamp", {
            "timestamp": {"$in": [row["timestamp"] for row in csv_data]}
        })
        csv_data = [row for row in csv_data if row["timestamp"] not in existing_timestamps]
        
        try:
            if csv_data:  
                result = user_collection.insert_many(csv_data, ordered=False)
            else:
                logger.info("No new data to insert after filtering duplicates.")
        except BulkWriteError as bwe:
            logger.warning(f"Error inserting data for user {user_id}: {bwe.details}")
            return jsonify({"status": "error", "message": "Insertion failed due to duplicate keys."}), 500
        
        return jsonify({"status": "success", "user_id": user_id}), 200

    else: 
        return jsonify({"status": "error", "message": "unsuccessful"}), 400

@app.route('/register_device_token', methods=['POST'])
def register_device_token():
    """
    客户端调用该接口来注册自己的 device token，
    请求 JSON 中需要包含 user_id 和 device_token 字段。
    """
    try:
        user_id = request.json.get("user_id")
        device_token = request.json.get("device_token")
        if not user_id or not device_token:
            return jsonify({"status": "error", "message": "缺少 user_id 或 device_token"}), 400
        
        db = get_db()
        # 更新或插入 token 信息，记录注册时间（便于后续清理或统计）
        db.device_tokens.update_one(
            {"user_id": user_id, "device_token": device_token},
            {"$set": {"registered_time": datetime.datetime.utcnow()}},
            upsert=True
        )
        return jsonify({"status": "success", "message": "设备 token 注册成功"}), 200
    except Exception as e:
        logger.exception("注册设备 token 时出错")
        return jsonify({"status": "error", "message": str(e)}), 500
    
def background_push_notification():
    """
    后台任务：每 300 秒自动向所有已注册设备发送静默推送通知。
    """
    while True:
        custom_data = {"title": "UPLOAD", "body": "Please upload your data"}
        try:
            db = get_db()
            # 从 device_tokens 集合中查询所有 token
            tokens_cursor = db.device_tokens.find({})
            tokens_list = [doc["device_token"] for doc in tokens_cursor]
            
            if not tokens_list:
                logger.info("暂无注册的设备 token，跳过本次推送。")
            else:
                for token in tokens_list:
                    if send_silent_push(token, custom_data):
                        logger.info(f"向 token {token} 推送成功")
                    else:
                        logger.error(f"向 token {token} 推送失败")
        except Exception as e:
            logger.exception("后台推送过程中出现异常")
        time.sleep(300)

if __name__ == '__main__':
    logger.info("应用启动测试日志")
    app = create_app()
    threading.Thread(target=background_push_notification, daemon=True).start()
    app.run(host="0.0.0.0")
