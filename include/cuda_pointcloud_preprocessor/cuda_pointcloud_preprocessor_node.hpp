#ifndef CUDA_POINTCLOUD_PREPROCESSOR__CUDA_POINTCLOUD_PREPROCESSOR_NODE_HPP_
#define CUDA_POINTCLOUD_PREPROCESSOR__CUDA_POINTCLOUD_PREPROCESSOR_NODE_HPP_

#include "cuda_pointcloud_preprocessor/cuda_pointcloud_preprocessor.hpp"
#include "cuda_pointcloud_preprocessor/point_types.hpp"

#include <autoware_point_types/types.hpp>
#include <cuda_blackboard/cuda_adaptation.hpp>
#include <cuda_blackboard/cuda_blackboard_publisher.hpp>
#include <cuda_blackboard/cuda_blackboard_subscriber.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#define CHECK_OFFSET(structure1, structure2, field)             \
  static_assert(                                                \
    offsetof(structure1, field) == offsetof(structure2, field), \
    "Offset of " #field " in " #structure1 " does not match expected offset.")

namespace cuda_pointcloud_preprocessor
{

static_assert(sizeof(InputPointType) == sizeof(autoware_point_types::PointXYZIRCAEDT));
static_assert(sizeof(OutputPointType) == sizeof(autoware_point_types::PointXYZIRC));

CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, x);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, y);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, z);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, intensity);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, return_type);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, channel);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, azimuth);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, elevation);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, distance);
CHECK_OFFSET(InputPointType, autoware_point_types::PointXYZIRCAEDT, time_stamp);

CHECK_OFFSET(OutputPointType, autoware_point_types::PointXYZIRCAEDT, x);
CHECK_OFFSET(OutputPointType, autoware_point_types::PointXYZIRCAEDT, y);
CHECK_OFFSET(OutputPointType, autoware_point_types::PointXYZIRCAEDT, z);
CHECK_OFFSET(OutputPointType, autoware_point_types::PointXYZIRCAEDT, intensity);
CHECK_OFFSET(OutputPointType, autoware_point_types::PointXYZIRCAEDT, return_type);
CHECK_OFFSET(OutputPointType, autoware_point_types::PointXYZIRCAEDT, channel);

class CudaPointcloudPreprocessorNode : public rclcpp::Node
{
public:
  explicit CudaPointcloudPreprocessorNode(const rclcpp::NodeOptions & node_options);
  ~CudaPointcloudPreprocessorNode() = default;

private:
  bool getTransform(
    const std::string & target_frame, const std::string & source_frame,
    tf2::Transform * tf2_transform_ptr);

  // Callback
  void pointcloudCallback(const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> cuda_msg);
  void twistCallback(
    const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr pointcloud_msg);
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg);

  tf2_ros::Buffer tf2_buffer_;
  tf2_ros::TransformListener tf2_listener_;

  std::string base_frame_;
  bool imu_tranform_valid_{false};
  std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> twist_queue_;
  std::deque<geometry_msgs::msg::Vector3Stamped> angular_velocity_queue_;

  // Subscriber
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_sub_{};
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_{};

  // CUDA pub & sub
  std::unique_ptr<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>> pub_;
  std::unique_ptr<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>> sub_;

  std::unique_ptr<CudaPointcloudPreprocessor> cuda_pointcloud_preprocessor_;
};

}  // namespace cuda_pointcloud_preprocessor

#endif  // CUDA_POINTCLOUD_PREPROCESSOR__CUDA_POINTCLOUD_PREPROCESSOR_NODE_HPP_
