find_package(SWIG REQUIRED)
include(UseSWIG)
include_directories(../include)
set_source_files_properties(openem.i PROPERTIES CPLUSPLUS ON)

if(BUILD_PYTHON)
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  swig_add_library(openem TYPE MODULE LANGUAGE python SOURCES openem.i)
  swig_link_libraries(openem ${PYTHON_LIBRARIES} openem)
  install(TARGETS _openem DESTINATION python)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openem.py DESTINATION python)
endif()

if(BUILD_CSHARP)
  enable_language(CSharp)
  swig_add_library(openem_cs 
    TYPE SHARED 
    LANGUAGE csharp 
    SOURCES openem.i)
  swig_link_libraries(openem_cs openem)
  file(GLOB CS_SRC "${CMAKE_CURRENT_BINARY_DIR}/*.cs")
  install(TARGETS openem_cs 
    LIBRARY DESTINATION csharp/lib
    ARCHIVE DESTINATION csharp/lib
    RUNTIME DESTINATION csharp/bin)
  install(FILES ${CS_SRC} DESTINATION csharp/src)
endif()
