FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Dataset đặt trong ./data (mount hoặc COPY), cấu hình ở src/config.py
# Chạy:  docker build -t btl-ml . && docker run --rm -v "$PWD/results:/app/results" btl-ml
CMD ["python", "src/run_all.py"]
