# README

**Note: NVIDIA Docker Required**

To utilize GPU capabilities within this Docker container, ensure the NVIDIA Docker runtime is installed on the host machine. This is crucial for GPU-accelerated tasks. Refer to NVIDIA's official documentation for installation details.

## Build 

```bash
docker build -t ykli_surgripe:latest .
```

## Run
```bash
docker run -it --gpus all --shm-size=16G --name ykli_surgripe_test -v /your/dataset/path/Dataset:/home/clean-pvnet/Dataset ykli_surgripe:latest
```


