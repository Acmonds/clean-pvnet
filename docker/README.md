# README

## Build 

```bash
docker build -t ykli_surgripe:latest .
```

## Run
```bash
docker run -it --gpus all --shm-size=16G --name ykli_surgripe_test -v /home/liyangke2014/clean-pvnet/Dataset:/home/clean-pvnet/Dataset ykli_surgripe:latest
```


