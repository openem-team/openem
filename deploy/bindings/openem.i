%module openem

%ignore openem::Image::operator=(Image&&);
%ignore openem::Image::operator=(const Image&);
%ignore openem::Image::Image(Image&&);

%{
#include "error_codes.h"
#include "image.h"
#include "video.h"
#include "find_ruler.h"
#include "detect.h"
#include "classify.h"
#include "count.h"
%}

%include "cpointer.i"
%include "stdint.i"
%include "std_array.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_vector.i"
namespace std {
  %template(VectorDouble) vector<double>;
  %template(VectorFloat) vector<float>;
  %template(VectorInt) vector<int>;
  %template(VectorVectorFloat) vector<vector<float>>;
  %template(VectorVectorVectorFloat) vector<vector<vector<float>>>;
  %template(VectorUint8) vector<uint8_t>;
  %template(PairIntInt) pair<int, int>;
  %template(Rect) array<int, 4>;
  %template(Color) array<uint8_t, 3>;
  %template(VectorRect) vector<array<int, 4>>;
  %template(VectorVectorRect) vector<vector<array<int, 4>>>;
  %template(VectorImage) vector<openem::Image>;
  %template(VectorDetection) vector<openem::detect::Detection>;
  %template(VectorVectorDetection) vector<vector<openem::detect::Detection>>;
  %template(ArrayFloat3) array<float, 3>;
  %template(VectorClassification) vector<openem::classify::Classification>;
  %template(VectorVectorClassification) vector<vector<openem::classify::Classification>>;
};

%include "error_codes.h"
%include "image.h"
%include "video.h"
%include "find_ruler.h"
%include "detect.h"
%include "classify.h"
%include "count.h"

