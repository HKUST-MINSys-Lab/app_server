import json
import logging
import pika
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from model import Model, ModelArgs

device = torch.device('cpu')
log_file = '/data/log/har_output.log'

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

# 全局加载模型，只初始化一次
model = None

def get_db():
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client.get_database(os.getenv("MONGODB_DB"))

def process_message(message):
    """
    处理从队列中接收到的消息。
    消息格式示例：{"user_id": "xxx", "data": [row1, row2, ...]}
    ["timestamp", "acc_X", "acc_Y", "acc_Z", "gyro_X", "gyro_Y", "gyro_Z", "mag_X", "mag_Y", "mag_Z"]
    """
    user_id = message.get("user_id")
    data = message.get("data")
    if not user_id or not data:
        logger.warning("消息格式不正确: %s", message)
        return

    df = pd.DataFrame(data) # note dtype
    logger.info("用户 %s 的数据: %s", user_id, len(df))
    
    if len(df) == 0:
        logger.warning("没有有效的输入数据, user_id: %s", user_id)
        return

    from pymongo import UpdateOne

    try:
        df, cost_time = infer(model, df)  # df: 包含 "timestamp" 和 "activity" 列
    except Exception as e:
        logger.error("模型预测失败: %s", e)
        raise e

    logger.info("用户 %s 预测完成，耗时: %s 秒", user_id, cost_time)

    # 批量更新 MongoDB 中的记录
    try:
        db = get_db()
        imu_collection = db[f"imu_{user_id}"]
        
        # 构建批量更新操作
        bulk_ops = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            activity = row["activity"]
            bulk_ops.append(UpdateOne({"timestamp": ts}, {"$set": {"activity": activity}}))
        
        if bulk_ops:
            result = imu_collection.bulk_write(bulk_ops, ordered=False)
            logger.info("批量更新完成：修改 %s 条记录", result.modified_count)
        else:
            logger.info("无更新操作")
    except Exception as e:
        logger.error("更新数据库失败: %s", e)
        raise e

def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        logger.info("接收到消息: %s", message)
        process_message(message)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error("处理消息失败: %s", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def main():
    global model
    # 初始化模型只做一次
    load_path = os.getenv("MODEL_PATH")
    model_args = json.load(open(os.path.join(load_path, 'args.json')))['model_args']
    model_args = ModelArgs.from_dict(model_args)
    model = Model(6, args=model_args)
    pretrained_mdl = torch.load(os.path.join(load_path, 'checkpoint.pth'), map_location='cpu')
    msg = model.load_state_dict(pretrained_mdl['model'], strict=False)
    logger.info("加载模型: %s", msg)
    
    model.to(device) # device is cpu
    logger.info("模型初始化完成。")
    
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='har_queue', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='har_queue', on_message_callback=callback)

    logger.info("HAR 消费者启动，等待消息...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("HAR 消费者被终止。")
        channel.stop_consuming()
    connection.close()

if __name__ == "__main__":
    main()
