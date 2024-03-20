# czsc_stremlit
使用 streamlit 进行可视化呈现

#### 本地环境运行
```shell
# PS: 本地环境运行需提前安装好ta-lib库(根据自己的操作系统自行Google)

# 1. 安装依赖
pip install ta-lib && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 创建.streamlit/secrets.toml配置文件
touch .streamlit/secrets.toml
# 文件内容如下，填入自己的参数即可
ts_token = "xxx"
OPENAI_API_KEY = "xxx"
OPENAI_API_BASE = "xxx"
ACTIVELOOP_TOKEN = "xxx"
allow_gpt4_users = ['xxx@xxx.com']

# 3. 启动服务
streamlit run CZSC.py
```

#### Docker方式运行
```shell
# 构建Docker镜像
docker build -t czsc-stremlit:0.0.1 .

# 运行镜像
docker run -d 
    --volume /opt/.streamlit/secrets.toml:/app/.streamlit/secrets.toml \
    --env OPENAI_API_KEY='xxx' \
    --env OPENAI_API_BASE='xxx' \
    --env ACTIVELOOP_TOKEN='xxx' \
    --name czsc-stremlit 
    czsc-stremlit:0.0.1
```
