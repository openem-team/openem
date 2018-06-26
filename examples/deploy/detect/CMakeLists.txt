add_executable(detect main.cc)
target_include_directories(detect PUBLIC
  ${PROJECT_SOURCE_DIR}/deploy/include)
target_link_libraries(detect openem)
install(TARGETS detect DESTINATION examples/deploy/detect)
