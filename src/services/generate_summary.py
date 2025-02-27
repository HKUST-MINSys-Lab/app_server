import os
import pandas as pd
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path='/root/app_server/.env')
import logging
import csv
import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# set log path
log_file = "/root/app_server/logs/summary.log"
def generate_summary(user_id, db):
    """
    根据每天数据生成 summary。
    如果当前时间未到晚上 10 点，则返回提示信息：
      "Today's summary is available at 10pm."
    如果前一天有 summary，则附加上 "This is your yesterday summary; you can choose to rate it if not yet done." 并附上对应的昨日 summary。
    如果已过晚上10点，则先检查数据库中是否存在今天的 summary，
      如果存在直接返回，
      否则调用 OpenAI API 生成今天的 summary，并保存后返回。
    """
    from datetime import datetime, time, timedelta
    now = datetime.now()
    today_date = now.date()
    ten_pm = datetime.combine(today_date, time(21, 0, 0))

    # 定义辅助函数：从 CSV 中加载指定 user_id 和日期的 summary
    def load_summary(user_id, date):
        summary_feedback = db.summary_feedbacks.find_one({
            "user_id": user_id,
            "timestamp": str(date)  # 确保格式匹配数据库中存储的格式
        })
        if summary_feedback:
            logger.info(f"Found summary: {summary_feedback}")
            return summary_feedback
        return None
    print(f"data for summary: {extract_data(user_id, db)}")
    if now < ten_pm:
        # 未到晚上 10 点：返回提示信息，并检查是否有昨日的 summary
        message = "Today's summary is available at 10pm."
        # output = {"message": message}
        yesterday = today_date - timedelta(days=1)
        yest_summary = load_summary(user_id, yesterday)
        if yest_summary is not None:
            message += " This is your yesterday summary; you can choose to rate it if not yet done."
            summaryA = message + "\n" + yest_summary.get("summaryA")
            summaryB = message + "\n" + yest_summary.get("summaryB")
        else:
            print("yest_summary is None")
            summaryA = message
            summaryB = message
        return summaryA, summaryB
    else:
        # 已到或超过晚上10点：检查是否已存在今天的 summary
        today_summary = load_summary(user_id, today_date)
        if today_summary is not None:
            return today_summary.get("summaryA"), today_summary.get("summaryB")
        else:
            # 今天的 summary 不存在，则生成 summary
            data = extract_data(user_id, db)
            client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version="2024-06-01",
                azure_endpoint="https://hkust.azure-api.net/"
            )
            messages = [
                {
                    "role": "system",
                    "content": f"""
                        You are an expert in mental health analysis using multimodal sensor data. Given a university student's mobile app data from today, first identify possible actions and scenarios based on the day's sensor data. Then, using both the raw data and the inferred scenarios, analyze the student's mental state.
                        Produce two distinct analyses:
                            • summaryA: A concise overview that describes the possible actions and scenarios, followed by an analysis indicating a positive or stable mental health state.
                            • summaryB: A concise overview that describes the possible actions and scenarios, followed by an analysis indicating potential mental health challenges.
                        Both summaries should clearly differentiate the reasoning paths and focus on:
                            • Deriving possible actions and scenarios from today's sensor data.
                            • Analyzing mental state based on the raw data and these scenarios.
                        Return your response as a valid JSON object with exactly the keys:
                            • "summaryA": [Your first analysis]
                            • "summaryB": [Your second analysis]
                        The data is below:
                        {data}
                        """
                },
            ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=512,
                temperature=0.0
            )
            response = response.model_dump_json(indent=2)
            response = json.loads(response)
            summaries_str = response.get("choices")[0].get("message").get("content")
            if summaries_str.strip().startswith("```json"):
                lines = summaries_str.splitlines()
                if len(lines) >= 3 and lines[0].strip().startswith("```"):
                    summaries_str = "\n".join(lines[1:-1])
            summaries = json.loads(summaries_str)
            summaryA = summaries.get("summaryA")
            summaryB = summaries.get("summaryB")

            # 保存生成的 summary 到 CSV 文件
            new_entry = {"user_id": user_id, "date": str(today_date), "summaryA": summaryA, "summaryB": summaryB}
            db.daily_summaries.insert_one(new_entry)
            return summaryA, summaryB


def extract_data(user_id, db):
    """
    Extracts data from the database for a specific user.
    """
    # 直接从 MongoDB 查询最近一天的数据
    user_collection = db[f"uploads_{user_id}"]
    last_day = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    data_cursor = user_collection.find({"timestamp": {"$gt": last_day}})
    data_list = list(data_cursor)
    if not data_list:
        return json.dumps([])
    data = pd.DataFrame(data_list)
    if 'user_id' not in data.columns:
        data['user_id'] = user_id
    
    # 将 'timestamp' 列转换为日期时间类型
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    data = data.dropna(subset=['timestamp'])
    
    
    # 根据上传接口的字段名称转换数值类型，将所有需要进行数值计算的字段都转换
    numeric_columns = [
        "volume", "screen_on_ratio", "network_traffic", "Rx_traffic", "Tx_traffic",
        "stepcount_sensor", "gpsLat", "gpsLon", "battery",
        "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ",
        "magneticFieldX", "magneticFieldY", "magneticFieldZ"
    ]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # 提取今天的数据
    today = pd.Timestamp.now(tz='Asia/Shanghai').date()

    data = data[data['timestamp'].dt.date == today]
    
    # 对IMU数据进行聚合处理
    imu_acc_cols = ["accX", "accY", "accZ"]
    if set(imu_acc_cols).issubset(data.columns):
        data["acc_norm"] = (data["accX"]**2 + data["accY"]**2 + data["accZ"]**2)**0.5
        data = data.drop(columns=imu_acc_cols)
    
    imu_gyro_cols = ["gyroX", "gyroY", "gyroZ"]
    if set(imu_gyro_cols).issubset(data.columns):
        data["gyro_norm"] = (data["gyroX"]**2 + data["gyroY"]**2 + data["gyroZ"]**2)**0.5
        data = data.drop(columns=imu_gyro_cols)
    
    # 使用 groupby 对 timestamp 按 '4h' 分组，并计算每个时段内各数值型数据的均值和标准差
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    agg_functions = {col: ['mean', 'std'] for col in numeric_cols}
    data = data.groupby(pd.Grouper(key='timestamp', freq='4h')).agg(agg_functions).reset_index()
    
    # 扁平化多级列索引，确保所有列名都是字符串
    data.columns = ['timestamp'] + [
        '_'.join(map(str, col)).strip() if isinstance(col, tuple) else str(col) 
        for col in data.columns[1:]
    ]
    
    # convert the aggregated data to a list of records
    records = data.to_dict(orient='records')
    
    # 对每个记录进行清洗
    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for k, v in record.items():
            if pd.notnull(v):
                if "sleepHours" in k and v == 0:
                    continue
                if isinstance(v, pd.Timestamp):
                    cleaned_record[k] = v.isoformat()
                elif isinstance(v, float):
                    cleaned_record[k] = float("{:.6g}".format(v))
                else:
                    cleaned_record[k] = v
        if len(cleaned_record) == 1 and 'timestamp' in cleaned_record:
            continue
        cleaned_records.append(cleaned_record)
    
    # convert the cleaned records to a JSON string
    data_json = json.dumps(cleaned_records)
    return data_json


if __name__ == "__main__":
    from pymongo import MongoClient
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))
    db = mongo_client.get_database(os.getenv("MONGODB_DB"))
    data = extract_data("ZHANG Liyu", db)
    # parse the data to a json object
    data = json.loads(data)
    print(data)
    summary = generate_summary("ZHANG Liyu", db)
    print(summary)
    print(type(summary))
    print(summary.get("summaryA"))
    print(summary.get("summaryB"))