FROM python:3.10-slim

WORKDIR /app

# Cài các gói hệ thống cần thiết cho OpenCV và Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cài thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY ap.py .
COPY static/ ./static/
COPY runs/segment/success1/weights/best.pt ./runs/segment/success1/weights/best.pt

EXPOSE 5000
CMD ["python", "ap.py"]
