function(enable_doxygen)
  set(DOXYGEN_GENERATE_XML YES)
  set(DOXYGEN_CALLER_GRAPH YES)
  set(DOXYGEN_CALL_GRAPH YES)
  set(DOXYGEN_EXTRACT_ALL YES)
  find_package(Doxygen REQUIRED dot)
  doxygen_add_docs(doxygen-docs ${PROJECT_SOURCE_DIR}/include)
endfunction()

enable_doxygen()

find_package(Sphinx REQUIRED)
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

add_custom_target(
  sphinx ALL
  DEPENDS doxygen-docs
  COMMAND
    ${SPHINX_EXECUTABLE} -b html
    -Dbreathe_projects.ddp-pinocchio=${CMAKE_CURRENT_BINARY_DIR}/xml
    ${SPHINX_SOURCE} ${SPHINX_BUILD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating documentation with Sphinx"
)
