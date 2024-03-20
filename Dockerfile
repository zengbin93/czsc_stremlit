FROM czsc/python:3.11-slim-buster

WORKDIR /app
ENV OPENAI_API_KEY ""
ENV OPENAI_API_BASE ""
ENV ACTIVELOOP_TOKEN ""

COPY . .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 80
CMD ["streamlit", "run", "CZSC.py"]
