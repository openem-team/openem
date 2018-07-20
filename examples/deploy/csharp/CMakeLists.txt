enable_language(CSharp)
include(CSharpUtilities)
file(GLOB SWIG_SRC "${PROJECT_BINARY_DIR}/deploy/bindings/*.cs")
csharp_set_windows_forms_properties(
  find_ruler.cs
  detect.cs)

add_executable(find_ruler_cs find_ruler.cs ${SWIG_SRC})
set_property(TARGET find_ruler_cs PROPERTY VS_DOTNET_REFERENCES
  "System"
  "System.Windows.Forms"
  "System.Drawing")
install(TARGETS find_ruler_cs DESTINATION examples/deploy/csharp)

add_executable(detect_cs detect.cs ${SWIG_SRC})
set_property(TARGET detect_cs PROPERTY VS_DOTNET_REFERENCES
  "System"
  "System.Windows.Forms"
  "System.Drawing")
install(TARGETS detect_cs DESTINATION examples/deploy/csharp)
