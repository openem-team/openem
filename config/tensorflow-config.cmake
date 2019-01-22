find_package(Threads REQUIRED)
add_library(tensorflow SHARED IMPORTED)
set_target_properties(tensorflow PROPERTIES
  IMPORTED_LOCATION
  /tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
  INTERFACE_INCLUDE_DIRECTORIES
  "/tensorflow/bazel-genfiles;/tensorflow"
  INTERFACE_LINK_LIBRARIES
  "/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so;${CMAKE_THREAD_LIBS_INIT}"
)
