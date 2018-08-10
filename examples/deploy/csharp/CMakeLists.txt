enable_language(CSharp)
include(CSharpUtilities)
file(GLOB SWIG_SRC "${PROJECT_BINARY_DIR}/deploy/bindings/*.cs")
set(SWIG_SRC
  ${PROJECT_BINARY_DIR}/deploy/bindings/VideoWriter.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VideoReader.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorVectorVectorFloat.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorVectorRect.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorVectorFloat.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorUint8.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorRect.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorImage.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorDetection.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorVectorDetection.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorClassification.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorVectorClassification.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorFloat.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/VectorDouble.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/SWIGTYPE_p_void.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/SWIGTYPE_p_unsigned_char.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/RulerMaskFinder.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Rect.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/PairIntInt.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/openemPINVOKE.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/openem.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Image.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Detection.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Classification.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/ErrorCode.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Detector.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Color.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Codec.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/ArrayFloat3.cs
  ${PROJECT_BINARY_DIR}/deploy/bindings/Classifier.cs)
add_custom_target(swig_source_gen 
  COMMAND ""
  DEPENDS openem_cs 
  BYPRODUCTS ${SWIG_SRC})
csharp_set_windows_forms_properties(
  find_ruler.cs
  detect.cs)

add_executable(find_ruler_cs find_ruler.cs ${SWIG_SRC})
add_dependencies(find_ruler_cs openem_cs)
set_property(TARGET find_ruler_cs PROPERTY VS_DOTNET_REFERENCES "System")
install(TARGETS find_ruler_cs DESTINATION examples/deploy/csharp)

add_executable(detect_cs detect.cs ${SWIG_SRC})
add_dependencies(detect_cs openem_cs)
set_property(TARGET detect_cs PROPERTY VS_DOTNET_REFERENCES "System")
install(TARGETS detect_cs DESTINATION examples/deploy/csharp)

add_executable(classify_cs classify.cs ${SWIG_SRC})
add_dependencies(classify_cs openem_cs)
set_property(TARGET classify_cs PROPERTY VS_DOTNET_REFERENCES "System")
install(TARGETS classify_cs DESTINATION examples/deploy/csharp)

add_executable(video_cs video.cs ${SWIG_SRC})
add_dependencies(video_cs openem_cs)
set_property(TARGET video_cs PROPERTY VS_DOTNET_REFERENCES "System")
install(TARGETS video_cs DESTINATION examples/deploy/csharp)

