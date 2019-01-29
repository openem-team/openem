find_package(Threads REQUIRED)
add_library(tensorflow SHARED IMPORTED)
set_target_properties(tensorflow PROPERTIES
  IMPORTED_LOCATION
  /tensorflow/lib/libtensorflow_cc.so
  INTERFACE_INCLUDE_DIRECTORIES
  "/tensorflow/include/bazel-genfiles;/tensorflow/include"
  INTERFACE_LINK_LIBRARIES
  "/tensorflow/lib/libtensorflow_framework.so;${CMAKE_THREAD_LIBS_INIT}"
)
