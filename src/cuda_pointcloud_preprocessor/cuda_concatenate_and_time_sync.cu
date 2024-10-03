
#include <cuda_pointcloud_preprocessor/cuda_concatenate_and_time_sync.hpp>

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <locale>
#include <numeric>
#include <vector>

namespace cuda_pointcloud_preprocessor
{

__global__ void synchronizeAndConcatenate(
  const OutputPointType * input_points, int num_points, TransformStruct transform,
  OutputPointType * output_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    float x = input_points[idx].x;
    float y = input_points[idx].y;
    float z = input_points[idx].z;

    output_points[idx].x =
      transform.m11 * x + transform.m12 * y + transform.m13 * z + transform.translation_x;
    output_points[idx].y =
      transform.m21 * x + transform.m22 * y + transform.m23 * z + transform.translation_y;
    output_points[idx].z =
      transform.m31 * x + transform.m32 * y + transform.m33 * z + transform.translation_z;
    output_points[idx].intensity = input_points[idx].intensity;
  }
}

CudaSyncAndConcatenate::CudaSyncAndConcatenate()
{
  sensor_msgs::msg::PointField x_field, y_field, z_field, intensity_field, return_type_field,
    channel_field, azimuth_field, elevation_field, distance_field, time_stamp_field;
  x_field.name = "x";
  x_field.offset = 0;
  x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  x_field.count = 1;

  y_field.name = "y";
  y_field.offset = 4;
  y_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  y_field.count = 1;

  z_field.name = "z";
  z_field.offset = 8;
  z_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  z_field.count = 1;

  intensity_field.name = "intensity";
  intensity_field.offset = 12;
  intensity_field.datatype = sensor_msgs::msg::PointField::UINT8;
  intensity_field.count = 1;

  return_type_field.name = "return_type";
  return_type_field.offset = 13;
  return_type_field.datatype = sensor_msgs::msg::PointField::UINT8;
  return_type_field.count = 1;

  channel_field.name = "channel";
  channel_field.offset = 14;
  channel_field.datatype = sensor_msgs::msg::PointField::UINT16;
  channel_field.count = 1;

  static_assert(sizeof(OutputPointType) == 16, "OutputPointType size is not 16 bytes");
  static_assert(offsetof(OutputPointType, x) == 0);
  static_assert(offsetof(OutputPointType, y) == 4);
  static_assert(offsetof(OutputPointType, z) == 8);
  static_assert(offsetof(OutputPointType, intensity) == 12);
  static_assert(offsetof(OutputPointType, return_type) == 13);
  static_assert(offsetof(OutputPointType, channel) == 14);

  point_fields_.push_back(x_field);
  point_fields_.push_back(y_field);
  point_fields_.push_back(z_field);
  point_fields_.push_back(intensity_field);
  point_fields_.push_back(return_type_field);
  point_fields_.push_back(channel_field);
}

void CudaSyncAndConcatenate::setInputTopics(const std::vector<std::string> & input_topics)
{
  input_topics_ = input_topics;
  for (const auto & topic : input_topics) {
    concat_struct_map_[topic];
    cudaStreamCreate(&concat_struct_map_[topic].stream);
    cudaEventCreate(&concat_struct_map_[topic].event);
  }
}

void CudaSyncAndConcatenate::setPointcloud(
  const std::string & topic_name,
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pointcloud_msg_ptr)
{
  ConcatStruct & concat_struct = concat_struct_map_[topic_name];
  assert(pointcloud_msg_ptr->point_step == sizeof(OutputPointType));
  concat_struct.pointcloud_msg_ptr = pointcloud_msg_ptr;
}

std::unique_ptr<cuda_blackboard::CudaPointCloud2> CudaSyncAndConcatenate::getConcatenatedCloud(
  std::unordered_map<std::string, TransformStruct> & transforms_maps)
{
  int num_concatenated_points = 0;

  for (const auto & [name, concat_struct] : concat_struct_map_) {
    if (transforms_maps.count(name) == 0) {
      continue;
    }

    num_concatenated_points +=
      concat_struct.pointcloud_msg_ptr->width * concat_struct.pointcloud_msg_ptr->height;
    // std::cout << "Concatenating " << name << " with " << concat_struct.pointcloud_msg_ptr->width
    // * concat_struct.pointcloud_msg_ptr->height << " points" << std::endl << std::flush;
  }

  if (num_concatenated_points > buffer_capacity_ || !buffer_) {
    buffer_capacity_ = static_cast<std::size_t>(
      1.1 * num_concatenated_points);  // over-allocate to avoid reallocation
    buffer_ = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    cudaMalloc(
      reinterpret_cast<void **>(&buffer_->data),
      buffer_capacity_ * sizeof(OutputPointType) * sizeof(uint8_t));
  }

  buffer_->width = num_concatenated_points;
  buffer_->height = 1;
  buffer_->point_step = sizeof(OutputPointType);
  buffer_->row_step = num_concatenated_points * sizeof(OutputPointType);
  buffer_->fields = point_fields_;
  buffer_->is_bigendian = false;
  buffer_->is_dense = true;
  ;

  int start_index = 0;

  for (auto it : concat_struct_map_) {
    ConcatStruct & concat_struct = it.second;

    if (
      concat_struct.pointcloud_msg_ptr == nullptr || transforms_maps.count(it.first) == 0 ||
      concat_struct.pointcloud_msg_ptr->width == 0) {
      continue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid =
      (concat_struct.pointcloud_msg_ptr->width * concat_struct.pointcloud_msg_ptr->height +
       threadsPerBlock - 1) /
      threadsPerBlock;

    synchronizeAndConcatenate<<<blocksPerGrid, threadsPerBlock, 0, concat_struct.stream>>>(
      reinterpret_cast<const OutputPointType *>(concat_struct.pointcloud_msg_ptr->data),
      concat_struct.pointcloud_msg_ptr->width * concat_struct.pointcloud_msg_ptr->height,
      transforms_maps[it.first], reinterpret_cast<OutputPointType *>(buffer_->data) + start_index);

    start_index +=
      concat_struct.pointcloud_msg_ptr->width * concat_struct.pointcloud_msg_ptr->height;
  }

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    std::cout << "kernel launch failed with error " << cudaGetErrorString(cudaerr) << std::endl
              << std::flush;
  }

  // Release the input pointclouds
  for (auto it : concat_struct_map_) {
    it.second.pointcloud_msg_ptr.reset();
  }

  return std::move(buffer_);
}

void CudaSyncAndConcatenate::preallocate()
{
  buffer_ = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  cudaMalloc(
    reinterpret_cast<void **>(&buffer_->data),
    buffer_capacity_ * sizeof(OutputPointType) * sizeof(uint8_t));
}

}  // namespace cuda_pointcloud_preprocessor
