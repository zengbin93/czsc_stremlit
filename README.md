# czsc_stremlit
使用 streamlit 进行可视化呈现

#### 构建Docker镜像(注意后面的".")
docker build -t czsc-stremlit:0.0.1 .
#### 根据镜像创建容器
docker run -d -p 8501:8501 --name czsc-stremlit czsc-stremlit:0.0.1

docker run -d -p 8501:8501 --env OPENAI_API_KEY='xxx' --env OPENAI_API_BASE='' --env ACTIVELOOP_TOKEN='' --name czsc-stremlit czsc-stremlit:0.0.1

#### 查看容器日志
docker logs --tail=100 -f czsc-stremlit
#### 进入容器命令行
docker exec -it czsc-stremlit bash
#### 停止容器
docker stop czsc-stremlit
#### 重启容器
docker restart czsc-stremlit
#### 启动容器
docker start czsc-stremlit
#### 查看容器状态
docker ps -a