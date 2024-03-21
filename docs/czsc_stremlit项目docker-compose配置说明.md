# czsc_stremlit 项目 docker-compose 配置说明

## 下载项目

从 `https://github.com/zengbin93/czsc_stremlit.git` 克隆或下载最新的项目代码。

## 创建镜像

进入 `czsc_stremlit` 项目目录，基于项目中的 `Dockerfile` 在本地创建镜像。

```shell
docker build -t czsc-stremlit:0.0.1 .
```

命令执行成功后，本地镜像仓库中会有创建好的镜像，可通过以下命令查看镜像列表

```shell
# docker images

REPOSITORY      TAG       IMAGE ID       CREATED              SIZE
czsc-stremlit   0.0.1     839591025820   About a minute ago   1.74GB
...
```



创建好的镜像也可以push到阿里云等云服务提供商提供的免费的镜像仓库中（私有或公开均可），方便从其他主机中进行调用。

阿里云镜像仓库服务说明文档可参考 `https://help.aliyun.com/zh/acr`

## 启动服务

**注意**： `docker-compose.yml` 文件中配置了一个volumes，将宿主机中的一个目录映射到各个czsc-stremlit容器中。可以根据需要调整宿主机目录的位置，避免因为宿主机目录不存在导致服务启动失败。更详细的说明请参考 docker-compose.yml文件说明 的 卷定义 一节。

进入 `docker-compose.yml` 和 `nginx.conf` 文件所在的目录后，执行下面命令，启动服务：

```shell
docker-compose up -d

λ docker-compose up -d
[+] Running 4/6
 - Network czsc_stremlit-docker-compose_default       Created   0.9s 
 - Volume "czsc_stremlit-docker-compose_data-volume"  Created   0.9s 
 ✔ Container czsc-stremlit_3                          Started   0.6s 
 ✔ Container czsc-stremlit_1                          Started   0.3s 
 ✔ Container czsc-stremlit_2                          Started   0.5s 
 ✔ Container czsc_stremlit-docker-compose-nginx-1     Started   0.8s 
```



## docker-compose.yml文件说明

```yaml
version: '3.7'

x-environment: &common-environment
  OPENAI_API_KEY: "OPENAI_API_KEY_VALUE"
  OPENAI_API_BASE: "OPENAI_API_BASE_VALUE"
  ACTIVELOOP_TOKEN: "ACTIVELOOP_TOKEN_VALUE"

services:
  czsc-stremlit_1:
    image: czsc-stremlit:0.0.1
    container_name: czsc-stremlit_1
    environment:
      <<: *common-environment
    volumes:
      - data-volume:/app/market-data
    expose:
      - "80"

  czsc-stremlit_2:
    image: czsc-stremlit:0.0.1
    container_name: czsc-stremlit_2
    environment:
      <<: *common-environment
    volumes:
      - data-volume:/app/market-data
    expose:
      - "80"

  czsc-stremlit_3:
    image: czsc-stremlit:0.0.1
    container_name: czsc-stremlit_3
    environment:
      <<: *common-environment
    volumes:
      - data-volume:/app/market-data
    expose:
      - "80"

  nginx:
    image: nginx:latest
    ports:
      - "8500:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - czsc-stremlit_1
      - czsc-stremlit_2
      - czsc-stremlit_3

volumes:
  data-volume:
    driver: local
    driver_opts:
      type: 'none'
      o: 'bind'
      device: 'D:/data/czsc_stremlit-data'
```

这个 `docker-compose` 文件包含了多个关键部分，每个部分都对应于容器化应用中的一个特定配置。

### 基础配置

- `version: '3.7'`: 指定了使用的 `docker-compose` 文件版本。这个版本号决定了文件中可以使用哪些特性和语法。

### 自定义环境变量

- `x-environment`: 使用 YAML 的锚点和别名功能定义了一组通用环境变量，可以在文件的多个部分中重复使用。这组环境变量包括 `OPENAI_API_KEY`、`OPENAI_API_BASE` 和 `ACTIVELOOP_TOKEN`，每个变量都被设置为一个具体的值。

### 服务定义

- `services`: 定义了 `docker-compose` 文件中要管理的所有服务（容器）。

#### czsc-stremlit 服务

- 每个 `czsc-stremlit` 服务（`czsc-stremlit_1`、`czsc-stremlit_2`、`czsc-stremlit_3`）都配置了以下共同的属性：
  - `image`: 指定服务使用的 Docker 镜像及其版本。
  - `container_name`: 为运行的容器指定一个唯一的名称。
  - `environment`: 引用了之前定义的通用环境变量。
  - `volumes`: 定义了一个卷挂载，将名为 `data-volume` 的卷挂载到容器内的 `/app/market-data` 目录。可以根据需求调整挂载文件映射到容器内部的路径，以满足挂载行情数据等文件的需求。
  - `expose`: 指定容器暴露的端口号，这里是 80 端口。这允许在同一网络内的其他服务连接到这个端口。这么定义的端口仅为docker-compose创建的虚拟网络中容器的内部端口，该端口并未暴露到宿主机上。

#### nginx 服务

- `nginx`: 配置了作为反向代理服务器的 nginx 服务。
  - `image`: 使用 Docker Hub 上最新版本的 nginx 镜像。
  - `ports`: 将宿主机的 8500 端口映射到容器的 80 端口，允许外部访问。**可以根据需求调整 `8500` 端口的值。**
  - `volumes`: 将宿主机上的 `nginx.conf` 文件挂载到容器内的 `/etc/nginx/nginx.conf`，用于配置 nginx。
  - `depends_on`: 指定 nginx 服务启动依赖于 `czsc-stremlit` 服务的启动。

### 卷定义

- `volumes`: 在顶层定义了命名卷 `data-volume`。
  - `driver`: 指定卷使用的驱动为 `local`。
  - `driver_opts`: 通过驱动选项将宿主机上的特定目录（`D:/data/czsc_stremlit-data`）作为 bind mount 挂载到容器内。可以根据实际需求修改本项目的值。

通过这个 `docker-compose` 文件，你可以一次性配置和启动三个 `czsc-stremlit` 服务实例和一个 nginx 反向代理服务。每个 `czsc-stremlit` 实例都会使用相同的环境变量和数据卷配置，而 nginx 服务则负责将外部请求通过 8500 端口转发到这些 `czsc-stremlit` 服务上，实现负载均衡和反向代理的功能。

## NGINX配置文件说明

因为需要提供负载均衡服务，需要定义一个NGINX配置文件：

```nginx
events { }

http {
    upstream streamlit {
        server czsc-stremlit_1:80;
        server czsc-stremlit_2:80;
        server czsc-stremlit_3:80;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://streamlit;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 86400; # can be adjusted
        }
    }
}
```

这个 `nginx.conf` 文件配置了 Nginx 作为反向代理服务器，用于分发请求到后端的 `czsc-stremlit` 服务实例。下面是文件中各项配置的逐一说明，可用作配置文件的说明文档：

### 基础配置

- `events { }`: 这个块用于配置影响 Nginx 服务器或与客户端连接相关的事件。在这个示例中，它被留空，表示使用 Nginx 的默认事件处理模型设置。

### HTTP 服务器配置

- `http`: 这个块包含了用于 HTTP 服务器的配置指令。

#### Upstream 定义

- `upstream streamlit`: 定义了一个名为 `streamlit` 的上游服务器组，这是负载均衡器的目标。包括三个 `czsc-stremlit` 服务实例，每个服务都监听在 80 端口。
  - `server czsc-stremlit_1:80;`
  - `server czsc-stremlit_2:80;`
  - `server czsc-stremlit_3:80;`: 这三行分别指定了上游组中的服务器和它们的端口号。

#### 服务器配置

- `server`: 这个块定义了一个 Nginx 服务器，用于处理传入的 HTTP 请求。
  - `listen 80;`: 指定 Nginx 监听在 80 端口上，等待传入的 HTTP 请求。

#### Location 块

- `location /`: 这个块指定了当请求匹配根路径 (`/`) 时的处理规则。
  - `proxy_pass http://streamlit;`: 指示 Nginx 将请求转发给名为 `streamlit` 的上游服务器组。
  - `proxy_http_version 1.1;`: 设置代理传递的 HTTP 协议版本为 1.1。
  - `proxy_set_header Upgrade $http_upgrade;`: 设置 `Upgrade` HTTP 头以支持 WebSocket 连接。
  - `proxy_set_header Connection "upgrade";`: 指示升级为 WebSocket 连接。
  - `proxy_set_header Host $host;`: 传递原始请求的主机头信息。
  - `proxy_set_header X-Real-IP $remote_addr;`: 传递客户端的真实 IP 地址。
  - `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`: 添加客户端 IP 地址到 `X-Forwarded-For` 头，用于追踪请求源。
  - `proxy_set_header X-Forwarded-Proto $scheme;`: 设置协议（如 http 或 https）到 `X-Forwarded-Proto` 头，用于标识请求使用的协议。
  - `proxy_read_timeout 86400;`: 设置读取超时时间为 86400 秒，以保持 WebSocket 连接不被过早关闭。

通过这些配置，Nginx 将能够接受来自客户端的 HTTP 和 WebSocket 请求，并根据配置将它们转发到后端的 `czsc-stremlit` 服务实例。这样做实现了请求的负载均衡，并允许 WebSocket 连接能够正常工作。

## 调整工作容器数量

要调整 `czsc-stremlit` 容器的数量并保持整个系统的正常运行，用户需要对 `docker-compose.yml` 和 `nginx.conf` 文件进行更新。

### 调整 `docker-compose.yml`

1. **增加 `czsc-stremlit` 服务实例：** 首先，复制现有的一个 `czsc-stremlit` 服务定义块，修改其 `container_name` 以反映新的服务编号（例如，`czsc-stremlit_4`），确保每个服务都有一个唯一的容器名称。

2. **更新 `nginx` 服务依赖：** 如果你添加了新的 `czsc-stremlit` 服务实例，记得在 `nginx` 服务的 `depends_on` 列表中添加新服务的引用，以确保 `nginx` 服务在所有 `czsc-stremlit` 实例启动后再启动。

### 更新 `nginx.conf`

1. **增加新的上游服务器：** 在 `nginx.conf` 文件中的 `upstream streamlit` 块，添加新的 `czsc-stremlit` 服务实例的 `server` 指令行，使用新容器的名称和端口（假设所有 `czsc-stremlit` 容器都暴露相同的端口，如 `80`）。

### 示例步骤

假设你想添加第四个 `czsc-stremlit` 容器：

1. **在 `docker-compose.yml` 中添加新的服务定义：**

```yaml
  czsc-stremlit_4:
    image: czsc-stremlit:0.0.1
    container_name: czsc-stremlit_4
    environment:
      <<: *common-environment
    volumes:
      - data-volume:/app/market-data
    expose:
      - "80"
```

2. **在 `nginx` 服务的 `depends_on` 中添加对新服务的引用：**

```yaml
  nginx:
    depends_on:
      - czsc-stremlit_1
      - czsc-stremlit_2
      - czsc-stremlit_3
      - czsc-stremlit_4
```

3. **在 `nginx.conf` 的 `upstream streamlit` 块中添加新的服务器：**

```nginx
    upstream streamlit {
        server czsc-stremlit_1:80;
        server czsc-stremlit_2:80;
        server czsc-stremlit_3:80;
        server czsc-stremlit_4:80;  # 新增加的服务
    }
```

### 应用更新

**应用 `docker-compose.yml` 的更改：** 在完成 `docker-compose.yml` 和 `nginx.conf` 文件的更新后，运行以下命令来重新启动服务，以便应用配置更改：

```bash
docker-compose down
docker-compose up -d
```

这会先停止并删除当前运行的所有服务容器，然后根据更新后的配置文件重新启动它们。

通过遵循这些步骤，可以根据需要灵活地调整 `czsc-stremlit` 容器的数量，无论是扩展还是缩减服务实例。

