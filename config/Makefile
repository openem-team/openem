BUILD_TOOLS=../externals/build_tools
USER=$(shell whoami)

build: 
	make openem_image

openem_image: Dockerfile.gen
	cd .. && docker build -t cvisionai/openem -f config/Dockerfile.gen . || exit 255

openem_lite: inference.gen
	cd .. && docker build -t cvisionai/openem_lite:$(USER) -f config/inference.gen . || exit 255

openem_cpu: cpu.gen
	cd .. && docker build -t cvisionai/openem_cpu:$(USER) -f config/cpu.gen . || exit 255

${BUILD_TOOLS}/makocc.py:
	cd .. && git submodule update --init

${BUILD_TOOLS}/version.sh:
	cd .. && git submodule update --init

.PHONY: ${BUILD_TOOLS}/version.py
${BUILD_TOOLS}/version.py: ${BUILD_TOOLS}/version.sh
	./${BUILD_TOOLS}/version.sh > ${BUILD_TOOLS}/version.py

Dockerfile.gen: Dockerfile.mako ${BUILD_TOOLS}/makocc.py ${BUILD_TOOLS}/version.py 
	./../externals/build_tools/makocc.py -o $@ $<

# Built wheels if on ARM
ifeq ($(shell uname -p), aarch64)
inference.gen: inference.mako ${BUILD_TOOLS}/makocc.py ${BUILD_TOOLS}/version.py
	./../externals/build_tools/makocc.py -o $@ $<
	make -C arm_packages
else
inference.gen: inference.mako ${BUILD_TOOLS}/makocc.py ${BUILD_TOOLS}/version.py
	./../externals/build_tools/makocc.py -o $@ $<
endif

cpu.gen: cpu.mako ${BUILD_TOOLS}/makocc.py ${BUILD_TOOLS}/version.py
	./../externals/build_tools/makocc.py -o $@ $<

ifeq ($(work_dir), )
extra_mounts=
else
extra_mounts=-v $(work_dir):/working --env deploy_dir=/working/deploy
endif

container_name=openem_$(USER)
ifeq ($(data_dir), )

else
extra_mounts+=-v $(data_dir):/data
container_name=openem_$(USER)_$(shell basename $(data_dir))
endif


ifndef openem_gpu
openem_gpu=0
endif
ifeq ($(shell uname -p), aarch64)
docker_cmd=nvidia-docker run
else
docker_cmd=docker run --gpus device=$(openem_gpu)
endif

ifndef openem_container_name
openem_container_name=$(container_name)_gpu_$(openem_gpu)
endif

inference_bash:
	$(docker_cmd) --name $(openem_container_name) --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`/../deploy_python:/deploy_python $(extra_mounts) cvisionai/openem_lite:$(USER)

inference_cpu:
	docker run --name $(container_name)_cpu --rm -ti --shm-size=1g -v `pwd`/../deploy_python:/deploy_python $(extra_mounts) cvisionai/openem_cpu:$(USER)


publish_lite:
	docker tag cvisionai/openem_lite:$(USER) cvisionai/openem_lite:latest
	docker push cvisionai/openem_lite:latest
