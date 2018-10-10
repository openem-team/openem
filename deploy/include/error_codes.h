/// @file
/// @brief Contains definitions of error codes.
/// @copyright Copyright (C) 2018 CVision AI.
/// @license This file is part of OpenEM, released under GPLv3.
//  OpenEM is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OPENEM_DEPLOY_ERROR_CODES_H_
#define OPENEM_DEPLOY_ERROR_CODES_H_

namespace openem {

/// All openem error codes.
enum ErrorCode {
  kSuccess = 0,        ///< No error.
  kErrorLoadingModel,  ///< Failed to read tensorflow model.
  kErrorGraphDims,     ///< Wrong number of dimensions in graph input node.
  kErrorNoShape,       ///< No shape attribute in graph input node.
  kErrorTfSession,     ///< Failed to create tensorflow session.
  kErrorTfGraph,       ///< Failed to create tensorflow graph.
  kErrorBadInit,       ///< Attempted to use uninitialized object.
  kErrorNotContinuous, ///< cv::Mat expected to be continuous.
  kErrorRunSession,    ///< Failed to run session.
  kErrorReadingImage,  ///< Failed to read image.
  kErrorImageSize,     ///< Invalid image data size for given dimensions.
  kErrorNumChann,      ///< Invalid number of image channels.
  kErrorVidFileOpen,   ///< Failed to open video file.
  kErrorVidNotOpen,    ///< Tried to read from unopened video.
  kErrorVidFrame,      ///< Failed to read video frame.
  kErrorLenMismatch,   ///< Mismatch in sequence lengths.
  kErrorNumInputDims,  ///< Unexpected number of input dimensions.
  kErrorBadSeqLength   ///< Wrong sequence length for input.
};

} // namespace openem

#endif // OPENEM_DEPLOY_ERROR_CODES_H_

