add_library(tensorflow SHARED IMPORTED)
set_target_properties(tensorflow PROPERTIES
  IMPORTED_LOCATION
  /tensorflow/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
  INTERFACE_INCLUDE_DIRECTORIES
  "/tensorflow/bazel-genfiles;/tensorflow"
)
