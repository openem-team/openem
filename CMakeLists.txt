cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(openem)

option(BUILD_DOCS   "Whether to enable documentation target."  ON)
option(BUILD_DEPLOY "Whether to build the deployment library." ON)

if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(BUILD_DEPLOY)
  add_subdirectory(deploy)
endif()
