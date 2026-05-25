# 1. 使用官方輕量化的 Python 3.10 Linux 鏡像作為地基
FROM python:3.10-slim

# 2. 設定貨櫃內部的預設工作目錄
WORKDIR /app

# 3. 先把套件清單複製進去並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 把後端所有的程式碼（main.py、.env 等）通通複製進貨櫃
COPY . .

# 5. 暴露出 8080 端口（GCP Cloud Run 預設通訊孔）
EXPOSE 8080

# 6. 貨櫃開機時的終極啟動指令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]