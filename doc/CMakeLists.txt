find_package(Doxygen)
if(${DOXYGEN_FOUND})
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    COMMENT "Generating library reference documentation with Doxygen."
    VERBATIM)
endif()

