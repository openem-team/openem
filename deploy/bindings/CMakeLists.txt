find_package(SWIG REQUIRED)
include(UseSWIG)
include_directories(../include)
set_source_files_properties(openem.i PROPERTIES CPLUSPLUS ON)

if(BUILD_PYTHON)
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  swig_add_library(openem TYPE MODULE LANGUAGE python SOURCES openem.i)
  swig_link_libraries(openem ${PYTHON_LIBRARIES} openem)
  install(TARGETS _openem DESTINATION modules)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openem.py DESTINATION python)
endif()

if(BUILD_CSHARP)
  swig_add_library(openem_cs 
    TYPE USE_BUILD_SHARED_LIBS 
    LANGUAGE csharp 
    SOURCES openem.i)
  swig_link_libraries(openem_cs openem)
  install(TARGETS openem_cs 
    LIBRARY DESTINATION lib 
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin) 
endif()
