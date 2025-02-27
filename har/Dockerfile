FROM python:3.9-slim
WORKDIR /app

# 复制依赖文件并安装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 默认启动命令（注意，这里默认启动的是 har_service.py，
# 如果你希望启动 har.py，请确保代码文件名称与之匹配）
CMD ["python", "har_service.py"]
