# Environment setup

Set env variables

```
$ export DATA_DIR="/path/to/data"

# example
$ export DATA_DIR=/mnt/nfs/coco
```

Start a docker container

```
$ docker compose up -d dethub
```

Install dethub

```
$ docker compose exec dethub pip install .
# optional install
$ docker compose exec dethub pip install -r docker/dev.txt
```
