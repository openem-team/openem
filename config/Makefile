BUILD_TOOLS=../externals/build_tools
build: 
	make openem_image

openem_image: Dockerfile.gen
	cd .. && docker build -t cvisionai/openem -f config/Dockerfile.gen . || exit 255

openem_inference: inference.gen
	cd .. && docker build -t cvisionai/openem_inference -f config/inference.gen . || exit 255

${BUILD_TOOLS}/makocc.py:
	cd .. && git submodule update --init

${BUILD_TOOLS}/version.sh:
	cd .. && git submodule update --init

.PHONY: ${BUILD_TOOLS}/version.py
${BUILD_TOOLS}/version.py: ${BUILD_TOOLS}/version.sh
	./${BUILD_TOOLS}/version.sh > ${BUILD_TOOLS}/version.py

Dockerfile.gen: Dockerfile.mako ${BUILD_TOOLS}/makocc.py ${BUILD_TOOLS}/version.py 
	./../externals/build_tools/makocc.py -o $@ $<

inference.gen: inference.mako ${BUILD_TOOLS}/makocc.py ${BUILD_TOOLS}/version.py
	./../externals/build_tools/makocc.py -o $@ $<

ifeq ($(deploy_dir), )
extra_mounts=
else
extra_mounts=-v $(deploy_dir):/deploy_dir
endif

inference_bash:
	nvidia-docker run --rm -ti -v `pwd`/../deploy_python:/deploy_python $(extra_mounts) cvisionai/openem_inference:latest