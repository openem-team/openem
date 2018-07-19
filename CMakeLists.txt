cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(openem)

option(BUILD_DOCS     "Whether to enable documentation target."          ON)
option(BUILD_DEPLOY   "Whether to build the deployment library."         ON)
option(BUILD_BINDINGS "Whether to build bindings to deployment library." ON)
option(BUILD_EXAMPLES "Whether to build example executables."            ON)

if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(BUILD_DEPLOY)
  add_subdirectory(deploy)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
