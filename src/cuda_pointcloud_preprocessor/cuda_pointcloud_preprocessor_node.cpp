
#include "cuda_pointcloud_preprocessor/cuda_pointcloud_preprocessor_node.hpp"

#include "cuda_pointcloud_preprocessor/point_types.hpp"

#include <nebula_common/point_types.hpp>  // needed during development to make sure we use the correct type. This should be removed in the future

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <vector>

namespace cuda_pointcloud_preprocessor
{
using sensor_msgs::msg::PointCloud2;

CudaPointcloudPreprocessorNode::CudaPointcloudPreprocessorNode(
  const rclcpp::NodeOptions & node_options)
: Node("cuda_pointcloud_preprocessor", node_options),
  tf2_buffer_(this->get_clock()),
  tf2_listener_(tf2_buffer_)
{
  std::cout << "CudaPointcloudPreprocessorNode" << std::endl;
  RCLCPP_WARN(get_logger(), "CudaPointcloudPreprocessorNode: start of constructor");
  using std::placeholders::_1;

  // Parameters
  base_frame_ = static_cast<std::string>(declare_parameter("base_frame", "base_link"));

  RingOutlierFilterParameters ring_outlier_filter_parameters;
  ring_outlier_filter_parameters.distance_ratio =
    static_cast<float>(declare_parameter("distance_ratio", 1.03));
  ring_outlier_filter_parameters.object_length_threshold =
    static_cast<float>(declare_parameter("object_length_threshold", 0.1));
  ring_outlier_filter_parameters.num_points_threshold =
    static_cast<int>(declare_parameter("num_points_threshold", 4));

  CropBoxParameters self_crop_box_parameters, mirror_crop_box_parameters;
  self_crop_box_parameters.min_x = static_cast<float>(declare_parameter("self_crop.min_x", -1.0));
  self_crop_box_parameters.min_y = static_cast<float>(declare_parameter("self_crop.min_y", -1.0));
  self_crop_box_parameters.min_z = static_cast<float>(declare_parameter("self_crop.min_z", -1.0));
  self_crop_box_parameters.max_x = static_cast<float>(declare_parameter("self_crop.max_x", 1.0));
  self_crop_box_parameters.max_y = static_cast<float>(declare_parameter("self_crop.max_y", 1.0));
  self_crop_box_parameters.max_z = static_cast<float>(declare_parameter("self_crop.max_z", 1.0));

  mirror_crop_box_parameters.min_x =
    static_cast<float>(declare_parameter("mirror_crop.min_x", -1.0));
  mirror_crop_box_parameters.min_y =
    static_cast<float>(declare_parameter("mirror_crop.min_y", -1.0));
  mirror_crop_box_parameters.min_z =
    static_cast<float>(declare_parameter("mirror_crop.min_z", -1.0));
  mirror_crop_box_parameters.max_x =
    static_cast<float>(declare_parameter("mirror_crop.max_x", 1.0));
  mirror_crop_box_parameters.max_y =
    static_cast<float>(declare_parameter("mirror_crop.max_y", 1.0));
  mirror_crop_box_parameters.max_z =
    static_cast<float>(declare_parameter("mirror_crop.max_z", 1.0));

  bool use_3d_undistortion = static_cast<bool>(declare_parameter("use_3d_undistortion", false));

  // Subscriber
  sub_ =
    std::make_unique<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/input/pointcloud", false,
      std::bind(&CudaPointcloudPreprocessorNode::pointcloudCallback, this, _1));

  // Publisher
  pub_ =
    std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/output/pointcloud");
  twist_sub_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "~/input/twist", 10,
    std::bind(&CudaPointcloudPreprocessorNode::twistCallback, this, std::placeholders::_1));

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "~/input/imu", 10,
    std::bind(&CudaPointcloudPreprocessorNode::imuCallback, this, std::placeholders::_1));

  cuda_pointcloud_preprocessor_ = std::make_unique<CudaPointcloudPreprocessor>();
  cuda_pointcloud_preprocessor_->setRingOutlierFilterParameters(ring_outlier_filter_parameters);
  cuda_pointcloud_preprocessor_->setCropBoxParameters(
    self_crop_box_parameters, mirror_crop_box_parameters);
  cuda_pointcloud_preprocessor_->set3DUndistortion(use_3d_undistortion);
}

bool CudaPointcloudPreprocessorNode::getTransform(
  const std::string & target_frame, const std::string & source_frame,
  tf2::Transform * tf2_transform_ptr)
{
  if (target_frame == source_frame) {
    tf2_transform_ptr->setOrigin(tf2::Vector3(0.0, 0.0, 0.0));
    tf2_transform_ptr->setRotation(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));
    return true;
  }

  try {
    const auto transform_msg =
      tf2_buffer_.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    tf2::convert(transform_msg.transform, *tf2_transform_ptr);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(get_logger(), "%s", ex.what());
    RCLCPP_ERROR(
      get_logger(), "Please publish TF %s to %s", target_frame.c_str(), source_frame.c_str());

    tf2_transform_ptr->setOrigin(tf2::Vector3(0.0, 0.0, 0.0));
    tf2_transform_ptr->setRotation(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));
    return false;
  }
  return true;
}

void CudaPointcloudPreprocessorNode::twistCallback(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr twist_msg_ptr)
{
  // RCLCPP_INFO_STREAM(get_logger(), "Adding twist with stamp " << twist_msg_ptr->header.stamp.sec
  // << "." << twist_msg_ptr->header.stamp.nanosec);
  twist_queue_.push_back(*twist_msg_ptr);

  while (!twist_queue_.empty()) {
    // for replay rosbag
    if (
      rclcpp::Time(twist_queue_.front().header.stamp) > rclcpp::Time(twist_msg_ptr->header.stamp)) {
      twist_queue_.pop_front();
    } else if (  // NOLINT
      rclcpp::Time(twist_queue_.front().header.stamp) <
      rclcpp::Time(twist_msg_ptr->header.stamp) - rclcpp::Duration::from_seconds(1.0)) {
      twist_queue_.pop_front();
    }
    break;
  }
}

void CudaPointcloudPreprocessorNode::imuCallback(
  const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
{
  /// RCLCPP_INFO_STREAM(get_logger(), "Adding imu with stamp " << imu_msg->header.stamp.sec << "."
  /// << imu_msg->header.stamp.nanosec);
  tf2::Transform tf2_imu_link_to_base_link{};
  getTransform(base_frame_, imu_msg->header.frame_id, &tf2_imu_link_to_base_link);
  geometry_msgs::msg::TransformStamped::SharedPtr tf_base2imu_ptr =
    std::make_shared<geometry_msgs::msg::TransformStamped>();
  tf_base2imu_ptr->transform.rotation = tf2::toMsg(tf2_imu_link_to_base_link.getRotation());

  geometry_msgs::msg::Vector3Stamped angular_velocity;
  angular_velocity.vector = imu_msg->angular_velocity;

  geometry_msgs::msg::Vector3Stamped transformed_angular_velocity;
  tf2::doTransform(angular_velocity, transformed_angular_velocity, *tf_base2imu_ptr);
  transformed_angular_velocity.header = imu_msg->header;
  angular_velocity_queue_.push_back(transformed_angular_velocity);

  while (!angular_velocity_queue_.empty()) {
    // for replay rosbag
    if (
      rclcpp::Time(angular_velocity_queue_.front().header.stamp) >
      rclcpp::Time(imu_msg->header.stamp)) {
      angular_velocity_queue_.pop_front();
    } else if (  // NOLINT
      rclcpp::Time(angular_velocity_queue_.front().header.stamp) <
      rclcpp::Time(imu_msg->header.stamp) - rclcpp::Duration::from_seconds(1.0)) {
      angular_velocity_queue_.pop_front();
    }
    break;
  }
}

void CudaPointcloudPreprocessorNode::pointcloudCallback(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> input_pointcloud_msg)
{
  static_assert(
    sizeof(InputPointType) == sizeof(nebula::drivers::NebulaPoint),
    "PointStruct and PointXYZIRADT must have the same size");

  auto start_system = std::chrono::system_clock::now();

  // Make sure that the first twist is newer than the first point
  InputPointType first_point;
  cudaMemcpy(
    &first_point, input_pointcloud_msg->data, sizeof(InputPointType), cudaMemcpyDeviceToHost);
  double first_point_stamp = input_pointcloud_msg->header.stamp.sec +
                             input_pointcloud_msg->header.stamp.nanosec * 1e-9 +
                             first_point.time_stamp * 1e-9;

  while (twist_queue_.size() > 1 &&
         rclcpp::Time(twist_queue_.front().header.stamp).seconds() < first_point_stamp) {
    // RCLCPP_INFO_STREAM(get_logger(), "Deleting twist with stamp " <<
    // twist_queue_[0].header.stamp.sec << "." << twist_queue_[0].header.stamp.nanosec);
    twist_queue_.pop_front();
  }

  while (angular_velocity_queue_.size() > 1 &&
         rclcpp::Time(angular_velocity_queue_.front().header.stamp).seconds() < first_point_stamp) {
    // RCLCPP_INFO_STREAM(get_logger(), "Deleting angular velocity with stamp " <<
    // angular_velocity_queue_[0].header.stamp.sec << "." <<
    // angular_velocity_queue_[0].header.stamp.nanosec);
    angular_velocity_queue_.pop_front();
  }

  // Obtain the base link to input pointcloud transform
  geometry_msgs::msg::TransformStamped transform_msg;

  try {
    transform_msg = tf2_buffer_.lookupTransform(
      base_frame_, input_pointcloud_msg->header.frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(get_logger(), "%s", ex.what());
    return;
  }

  // RCLCPP_INFO(get_logger(), "Processing pointcloud %s with %d points",
  // input_pointcloud_msg->header.frame_id.c_str(),
  // input_pointcloud_msg->height*input_pointcloud_msg->width);
  auto output_pointcloud_ptr = cuda_pointcloud_preprocessor_->process(
    *input_pointcloud_msg, transform_msg, twist_queue_, angular_velocity_queue_);
  output_pointcloud_ptr->header.frame_id = base_frame_;

  // Publish
  pub_->publish(std::move(output_pointcloud_ptr));

  // Preallocate for next iteration
  cuda_pointcloud_preprocessor_->preallocateOutput();

  // /auto end_system = std::chrono::system_clock::now();
  // RCLCPP_INFO(get_logger(), "System Processing time: %f", 1e-6 *
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_system - start_system).count());
}

}  // namespace cuda_pointcloud_preprocessor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cuda_pointcloud_preprocessor::CudaPointcloudPreprocessorNode)
