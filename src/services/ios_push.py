import os
import logging
from apns2.client import APNsClient
from apns2.credentials import TokenCredentials
from apns2.payload import Payload
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# 从环境变量中加载 APNs 配置（需要提前配置这些环境变量）
APNS_AUTH_KEY_PATH = os.getenv("APNS_AUTH_KEY_PATH")  # 认证密钥文件路径（例如：AuthKey_XXXXXXXXXX.p8）
APNS_KEY_ID = os.getenv("APNS_KEY_ID")                # Key ID
APNS_TEAM_ID = os.getenv("APNS_TEAM_ID")              # Team ID
APNS_TOPIC = os.getenv("APNS_TOPIC")                  # 应用的 Bundle Identifier
APNS_USE_SANDBOX = os.getenv("APNS_USE_SANDBOX", "True").lower() in ("true", "1", "yes")

# 初始化 APNs 客户端
credentials = TokenCredentials(auth_key_path=APNS_AUTH_KEY_PATH, auth_key_id=APNS_KEY_ID, team_id=APNS_TEAM_ID)
apns_client = APNsClient(credentials, use_sandbox=APNS_USE_SANDBOX)

def send_silent_push(device_token: str, custom_data: dict) -> bool:
    """
    发送静默推送通知到指定 iOS 设备

    参数:
        device_token: iOS 设备的 Device Token
        custom_data: 自定义数据，字典形式，将作为 payload 的一部分

    返回:
        True 如果发送成功，否则返回 False
    """
    try:
        # 构造 payload，设置 content_available 为 True，确保为静默推送，不包含 alert/sound
        payload = Payload(content_available=True, custom=custom_data)
        # 发送通知，APNS Topic 需与 APP 的 Bundle Identifier 匹配
        response = apns_client.send_notification(device_token, payload, topic=APNS_TOPIC)
        logger.info(f"Push notification sent. Device token: {device_token}, Response: {response}")
        return True
    except Exception as e:
        logger.exception("发送静默推送通知失败")
        return False 