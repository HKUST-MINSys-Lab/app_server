from flask import Blueprint, request, jsonify
from src.services.ios_push import send_silent_push
import logging

logger = logging.getLogger(__name__)
ios_push_bp = Blueprint("ios_push", __name__)

@ios_push_bp.route("/send_silent_push", methods=["POST"])
def send_silent_push_route():
    """
    接收请求数据并向指定 iOS 设备发送静默推送通知。

    请求体示例：
    {
        "device_token": "xxx",
        "custom_data": {
            "customData": "..."
        }
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "无效请求"}), 400

    device_token = data.get("device_token")
    custom_data = data.get("custom_data", {})

    if not device_token:
        return jsonify({"status": "error", "message": "缺少 device_token 参数"}), 400

    if send_silent_push(device_token, custom_data):
        return jsonify({"status": "success", "message": "静默推送通知发送成功"}), 200
    else:
        return jsonify({"status": "error", "message": "静默推送通知发送失败"}), 500 