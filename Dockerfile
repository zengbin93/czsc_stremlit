FROM python:3.11-slim-buster
WORKDIR /app
ENV OPENAI_API_KEY ""
ENV OPENAI_API_BASE ""
ENV ACTIVELOOP_TOKEN ""
COPY . /app
RUN apt-get -y update && \
    apt-get -y install wget gcc make && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr&& \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz
RUN pip install ta-lib  && pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 80
CMD ["streamlit", "run", "CZSC.py", "--server.port", "80", "----server.maxUploadSize", "1024"]