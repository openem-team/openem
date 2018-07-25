add_executable(find_ruler find_ruler.cc)
target_include_directories(find_ruler PUBLIC
  ${PROJECT_SOURCE_DIR}/deploy/include)
target_link_libraries(find_ruler openem)
install(TARGETS find_ruler DESTINATION examples/deploy/cc)

add_executable(detect detect.cc)
target_include_directories(detect PUBLIC
  ${PROJECT_SOURCE_DIR}/deploy/include)
target_link_libraries(detect openem)
install(TARGETS detect DESTINATION examples/deploy/cc)

add_executable(classify classify.cc)
target_include_directories(classify PUBLIC
  ${PROJECT_SOURCE_DIR}/deploy/include)
target_link_libraries(classify openem)
install(TARGETS classify DESTINATION examples/deploy/cc)
