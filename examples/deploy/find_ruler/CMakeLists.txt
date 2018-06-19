if(MSVC)
  # These are needed on Windows to avoid compilation issues with Tensorflow
  add_definitions(-DNOMINMAX)
  add_definitions(-DCOMPILER_MSVC)
endif()
add_executable(find_ruler main.cc)
target_include_directories(find_ruler PUBLIC
  ${PROJECT_SOURCE_DIR}/deploy/include)
target_link_libraries(find_ruler openem)
install(TARGETS find_ruler DESTINATION examples/deploy/find_ruler)

