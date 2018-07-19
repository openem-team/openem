if(MSVC)
  # These are needed on Windows to avoid compilation issues with Tensorflow
  add_definitions(-DNOMINMAX)
  add_definitions(-DCOMPILER_MSVC)
endif()

add_subdirectory(cc)

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(BUILD_CSHARP)
  add_subdirectory(csharp)
endif()
