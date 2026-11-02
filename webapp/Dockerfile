FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 && \
    pip install -r /app/webapp/requirements.txt

COPY node_graph /app/node_graph
COPY webapp /app/webapp
WORKDIR /app
EXPOSE 10000
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "10000"]
