build: 
	make openem_image

openem_image: Dockerfile.gen
	cd .. && docker build -t cvisionai/openem -f config/Dockerfile.gen . || exit 255

Dockerfile.gen: Dockerfile.mako
	./../externals/build_tools/makocc.py -o $@ $<
