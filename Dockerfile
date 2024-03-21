FROM czsc/python:3.11-slim-buster
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 80
CMD ["streamlit", "run", "CZSC.py", "--server.port", "80", "--server.enableCORS", "false", "--server.headless", "true", "--theme.base", "dark"]
