if(MSVC)
  # These are needed on Windows to avoid compilation issues with Tensorflow
  add_definitions(-DNOMINMAX)
  add_definitions(-DCOMPILER_MSVC)
endif()

add_subdirectory(find_ruler)
add_subdirectory(detect)
