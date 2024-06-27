
#ifndef CUDA_POINTCLOUD_PREPROCESSOR__CUDA_CONCATENATE_AND_TIME_SYNC_HPP_
#define CUDA_POINTCLOUD_PREPROCESSOR__CUDA_CONCATENATE_AND_TIME_SYNC_HPP_

#include "cuda_pointcloud_preprocessor/point_types.hpp"

#include <cuda_blackboard/cuda_pointcloud2.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <thrust/device_vector.h>

#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cuda_pointcloud_preprocessor
{

struct TransformStruct
{
  float translation_x;
  float translation_y;
  float translation_z;
  float m11;
  float m12;
  float m13;
  float m21;
  float m22;
  float m23;
  float m31;
  float m32;
  float m33;
};

struct ConcatStruct
{
  cudaStream_t stream;
  cudaEvent_t event;
  std::shared_ptr<const cuda_blackboard::CudaPointCloud2> pointcloud_msg_ptr;
};

class CudaSyncAndConcatenate
{
public:
  CudaSyncAndConcatenate();
  ~CudaSyncAndConcatenate() {}

  void setInputTopics(const std::vector<std::string> & input_topics);
  void setPointcloud(
    const std::string & topic_name,
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pointcloud_msg_ptr);

  std::unique_ptr<cuda_blackboard::CudaPointCloud2> getConcatenatedCloud(
    std::unordered_map<std::string, TransformStruct> & transforms_maps);
  void preallocate();

private:
  std::vector<std::string> input_topics_;
  std::unordered_map<std::string, ConcatStruct> concat_struct_map_;
  std::vector<sensor_msgs::msg::PointField> point_fields_;

  std::unique_ptr<cuda_blackboard::CudaPointCloud2> buffer_{};
  std::size_t buffer_capacity_{0};

  std::mutex mutex_;
};

}  // namespace cuda_pointcloud_preprocessor

#endif  // CUDA_POINTCLOUD_PREPROCESSOR__CUDA_CONCATENATE_AND_TIME_SYNC_HPP_
