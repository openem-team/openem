enable_language(CSharp)

add_executable(find_ruler_cs find_ruler.cs)
target_link_libraries(find_ruler_cs openem_cs)
install(TARGETS find_ruler_cs DESTINATION examples/deploy/csharp)

#add_executable(find_ruler_cs detect.cs)
#target_link_libraries(detect_cs openem_cs)
#install(TARGETS detect_cs DESTINATION examples/deploy/csharp)
