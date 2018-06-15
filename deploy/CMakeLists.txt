find_package(Tensorflow REQUIRED)
find_package(OpenCV REQUIRED)

if(MSVC)
  # These are needed on Windows to avoid compilation issues with Tensorflow
  add_definitions(-DNOMINMAX)
  add_definitions(-DCOMPILER_MSVC)
endif()

add_subdirectory(src)
