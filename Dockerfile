FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 系统依赖：bcrypt/cryptography/matplotlib 等可能需要的编译工具与字体
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        fonts-dejavu-core \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# 静态资源目录（图表会落盘到 static/charts/）
RUN mkdir -p static/charts

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
