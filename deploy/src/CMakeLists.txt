add_library(openem find_ruler.cc util.cc model.cc)
target_include_directories(openem PUBLIC 
  ${PROJECT_SOURCE_DIR}/deploy/include
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(openem
  tensorflow
  ${OpenCV_LIBS})

