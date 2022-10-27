# make appropriate changes before running the scripts
# START OF CONFIG =====================================================================================================
IMAGE=masud/dl-aio
CONTAINER=masud_PL_domainadp
AVAILABLE_GPUS='0'
LOCAL_JUPYTER_PORT=18888
LOCAL_TENSORBOARD_PORT=18006
VSCODE_PORT=18443
MATLAB_PORT=18080
PASSWORD=hellomoto
WORKSPACE=/home/masud/Desktop/github/PL_domainadp/
# END OF CONFIG  ======================================================================================================

docker-resume:
	docker start -ai $(CONTAINER)

docker-run:
	docker run --restart=always --env=DISPLAY --gpus '"device=$(AVAILABLE_GPUS)"' -it -e PASSWORD=$(PASSWORD) -e JUPYTER_TOKEN=$(PASSWORD) -p $(VSCODE_PORT):8443 -p $(LOCAL_JUPYTER_PORT):8888 -p \
		$(LOCAL_TENSORBOARD_PORT):6006 -v $(WORKSPACE):/notebooks --name $(CONTAINER) $(IMAGE)
		
docker-stop:
	docker stop $(CONTAINER)

docker-shell:
	docker exec -it $(CONTAINER) bash

docker-clean:
	docker rm $(CONTAINER)

docker-build:
	docker build -t $(IMAGE) -f Dockerfile .
	
docker-rebuild:
	docker build -t $(IMAGE) -f Dockerfile --no-cache --pull .

docker-push:
	docker push $(IMAGE)

docker-tensorboard:
	docker exec -it $(CONTAINER) tensorboard --logdir=logs

docker-vscode:
	docker exec -it $(CONTAINER) code-server --bind-addr 0.0.0.0:8443 --auth password --disable-telemetry /notebooks

docker-matlab-run:
	docker run --gpus '"device=$(AVAILABLE_GPUS)"' -it -e PASSWORD=$(PASSWORD) -p $(MATLAB_PORT):6080 --shm-size=512M -e MLM_LICENSE_FILE=<port id>@<location> -v $(shell pwd):/notebooks --name matlab_test nvcr.io/partners/matlab:r2020a
# Replace the port and location for network license
