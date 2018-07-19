%module openem

%ignore openem::Image::operator=(Image&&);
%ignore openem::Image::operator=(const Image&);
%ignore openem::Image::Image(Image&&);

%{
#include "error_codes.h"
#include "image.h"
#include "find_ruler.h"
#include "detect.h"
%}

%include "pointer.i"
%include "stdint.i"
%include "std_array.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_vector.i"
namespace std {
  %template(vector_double) vector<double>;
  %template(vector_uint8) vector<uint8_t>;
  %template(pair_int_int) pair<int, int>;
  %template(rect) array<int, 4>;
  %template(vector_rect) vector<array<int, 4>>;
  %template(vector_vector_rect) vector<vector<array<int, 4>>>;
  %template(vector_image) vector<openem::Image>;
};

%include "error_codes.h"
%include "image.h"
%include "find_ruler.h"
%include "detect.h"

