cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

project(openem LANGUAGES CXX)

option(BUILD_DOCS     "Whether to enable documentation target."          ON)
option(BUILD_DEPLOY   "Whether to build the deployment library."         ON)
option(BUILD_PYTHON   "Whether to build python bindings."                ON)
option(BUILD_CSHARP   "Whether to build C# bindings."                    ON)
option(BUILD_EXAMPLES "Whether to build example executables."            ON)

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /MP")
else()
  set(BUILD_CSHARP OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CXX_STANDARD 11)

# Not Ideal, but let's do this for now -- screwed up Eigen didn't go in
# include.
include_directories(BEFORE /opt/include /opt)
link_directories(/opt/lib)

if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(BUILD_DEPLOY)
  add_subdirectory(deploy)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

install(FILES LICENSE.md DESTINATION .)
install(DIRECTORY train DESTINATION .)
