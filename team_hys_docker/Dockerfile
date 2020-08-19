# 指定python版本，这里使用3.6

# 指定镜像源地址可能更快，或 FROM python:3.6

FROM hub.c.163.com/library/python:3.6

# 安装需要的python包，国内指定某些源，下载比较快，此处使用清华源（可以指定你用的版本，如tensorflow=1.0）

RUN pip install --upgrade pip

RUN pip list

RUN pip --default-timeout=1000 install tensorflow==2.0 numpy pandas scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
# 这个dockerfile文件同目录下，会用到的目录，复制到docker环境，统一放在目录下 xx/，其中xx为docker容器新建的目录，以后再docker中运行，需要的文件都需要放到docker容器中

COPY ./scaler.pkl team_hys_docker/
COPY ./models/ team_hys_docker/models
COPY ./data/ team_hys_docker/data
COPY ./script.py team_hys_docker/
COPY ./models.py team_hys_docker/
COPY ./utils.py team_hys_docker/
COPY ./preprocess.py team_hys_docker/
# 如果你有多个目录，也需要依次拷贝到docker 容器中
# 使用/* 的格式在docker中可能会造成错误
# 设置工作目录为xx

WORKDIR team_hys_docker
# 运行xx目录下面的model_demo.py, docker只能有一个CMD命令，因此我们执行了一系列命令，保证运行完毕后，sleep 2h，使得docker容器还在

CMD echo start | tee start_ && python script.py && echo done |tee done_ && sleep 2h
# 注意，model_demo.py不要用绝对路径，要使用复制到docker后的相对路径

# 比如 COPY ./data xx/data

# 且当前docker的工作目录是xx

# 那么python文件里面用 pd.read_csv('./data/test_sensor.csv') 可读取到文件