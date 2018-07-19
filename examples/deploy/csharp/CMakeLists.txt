enable_language(CSharp)
include(CSharpUtilities)

file(GLOB SWIG_SRC "${PROJECT_BINARY_DIR}/deploy/bindings/*.cs")
add_executable(find_ruler_cs find_ruler.cs ${SWIG_SRC})
csharp_set_windows_forms_properties(find_ruler.cs)
set_property(TARGET find_ruler_cs PROPERTY VS_DOTNET_REFERENCES
  "System"
  "System.Windows.Forms"
  "System.Drawing")
install(TARGETS find_ruler_cs DESTINATION examples/deploy/csharp)

#add_executable(find_ruler_cs detect.cs)
#target_link_libraries(detect_cs openem_cs)
#install(TARGETS detect_cs DESTINATION examples/deploy/csharp)
