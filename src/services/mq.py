import pika # message queue
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")

def send_to_queue(message):
    """
    使用 pika 发送消息到 har_queue 队列，队列设置为持久化
    """
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue='har_queue', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='har_queue',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # 消息持久化
            )
        )
        connection.close()
        logging.info("Message sent to queue: %s", message)
    except Exception as e:
        logging.error("Error sending message to queue: %s", e)
        raise