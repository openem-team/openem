BUILD_TOOLS=../externals/build_tools
build: 
	make openem_image

openem_image: Dockerfile.gen
	cd .. && docker build -t cvisionai/openem -f config/Dockerfile.gen . || exit 255

openem_lite: inference.gen
	cd .. && docker build -t cvisionai/openem_lite -f config/inference.gen . || exit 255

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

ifeq ($(work_dir), )
extra_mounts=
else
extra_mounts=-v $(work_dir):/working --env deploy_dir=/working
endif

ifeq ($(data_dir), )

else
extra_mounts+=-v $(data_dir):/data
endif



ifeq ($(shell uname -p), aarch64)
docker_cmd=nvidia-docker run
else
docker_cmd=docker --gpus device=0
endif

inference_bash:
	$(docker_cmd) --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`/../deploy_python:/deploy_python $(extra_mounts) -p 10001:10001 cvisionai/openem_lite:latest
