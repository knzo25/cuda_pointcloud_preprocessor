#ifndef CUDA_POINTCLOUD_PREPROCESSOR__POINT_TYPES_HPP_
#define CUDA_POINTCLOUD_PREPROCESSOR__POINT_TYPES_HPP_

#include <cstdint>

namespace cuda_pointcloud_preprocessor
{

// Note: We can not use PCL nor uniform initialization here because of thrust
struct OutputPointType
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint8_t return_type;
  std::uint16_t channel;
};

struct InputPointType
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint8_t return_type;
  std::uint16_t channel;
  float azimuth;
  float elevation;
  float distance;
  std::uint32_t time_stamp;
};

}  // namespace cuda_pointcloud_preprocessor

#endif  // CUDA_POINTCLOUD_PREPROCESSOR__POINT_TYPES_HPP_
