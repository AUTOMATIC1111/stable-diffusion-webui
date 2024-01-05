# sd-tool-server

`sd tool server` 是AIGC(图刷刷)相关功能在python实现的接口服务作为图刷刷后端的补充， 之所以不在图刷刷实现是因为图刷刷后端为`golang`而`python（SD）`处于队列调度之后，会导致部分功能等待。

## 功能接口
- `/tools/v1/face/reco`: 检测数字分身的训练集质量
- `/tools/v1/face/reco/main`: 检测数字分身的主图质量

## 快速开始


- 安装依赖：
```shell
pip install -r requirements.txt

```
**注意**：
insightface>=0.2部署在GPU上必须手动安装`onnxruntime-GPU`以启用GPU推理，或安装`onnxruntime`以仅使用CPU推理。

- 配置环境变量：
```shell
StorageEndponit=OSS或者OBS的endpoint
StorageSK=OSS或者OBS的sk
StorageAK=OSS或者OBS的ak
StorageBucket=OSS或者OBS的bucket

```
- 运行服务:
```shell
python3 main.py 
```
