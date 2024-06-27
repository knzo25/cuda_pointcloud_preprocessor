
#ifndef CUDA_POINTCLOUD_PREPROCESSOR__CUDA_CONCATENATE_AND_TIME_SYNC_NODE_HPP_
#define CUDA_POINTCLOUD_PREPROCESSOR__CUDA_CONCATENATE_AND_TIME_SYNC_NODE_HPP_

#include <autoware/universe_utils/ros/debug_publisher.hpp>
#include <autoware/universe_utils/system/stop_watch.hpp>
#include <autoware_point_types/types.hpp>
#include <cuda_blackboard/cuda_blackboard_publisher.hpp>
#include <cuda_blackboard/cuda_blackboard_subscriber.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <cuda_pointcloud_preprocessor/cuda_concatenate_and_time_sync.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <point_cloud_msg_wrapper/point_cloud_msg_wrapper.hpp>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tier4_debug_msgs/msg/int32_stamped.hpp>
#include <tier4_debug_msgs/msg/string_stamped.hpp>

#include <message_filters/pass_through.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace cuda_pointcloud_preprocessor
{
using autoware_point_types::PointXYZIRC;
using point_cloud_msg_wrapper::PointCloud2Modifier;

/** \brief @b CudaPointCloudConcatenateAndSyncNode is a special form of data
 * synchronizer: it listens for a set of input PointCloud messages on the same topic,
 * checks their timestamps, and concatenates their fields together into a single
 * PointCloud output message.
 * \author Radu Bogdan Rusu
 */
class CudaPointCloudConcatenateAndSyncNode : public rclcpp::Node
{
public:
  // typedef sensor_msgs::msg::PointCloud2 PointCloud2;

  /** \brief constructor. */
  explicit CudaPointCloudConcatenateAndSyncNode(const rclcpp::NodeOptions & node_options);

  /** \brief constructor.
   * \param queue_size the maximum queue size
   */
  CudaPointCloudConcatenateAndSyncNode(const rclcpp::NodeOptions & node_options, int queue_size);

  /** \brief Empty destructor. */
  virtual ~CudaPointCloudConcatenateAndSyncNode() {}

private:
  /** \brief The output PointCloud publisher. */
  std::shared_ptr<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>
    pub_output_;

  /** \brief The maximum number of messages that we can store in the queue. */
  int maximum_queue_size_ = 3;

  double timeout_sec_ = 0.1;

  bool publish_synchronized_pointcloud_;
  bool keep_input_frame_in_synchronized_pointcloud_;
  std::string synchronized_pointcloud_postfix_;

  std::set<std::string> not_subscribed_topic_names_;

  /** \brief A vector of subscriber. */
  std::vector<
    std::shared_ptr<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>>
    filters_;

  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_twist_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;

  rclcpp::TimerBase::SharedPtr timer_;
  diagnostic_updater::Updater updater_{this};

  CudaSyncAndConcatenate cuda_sync_and_concatenate_;

  const std::string input_twist_topic_type_;

  /** \brief Output TF frame the concatenated points should be transformed to. */
  std::string output_frame_;

  /** \brief Input point cloud topics. */
  // XmlRpc::XmlRpcValue input_topics_;
  std::vector<std::string> input_topics_;

  /** \brief TF listener object. */
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> twist_ptr_queue_;

  std::map<std::string, std::shared_ptr<const cuda_blackboard::CudaPointCloud2>> cloud_stdmap_;
  std::map<std::string, std::shared_ptr<const cuda_blackboard::CudaPointCloud2>> cloud_stdmap_tmp_;
  std::mutex mutex_;
  double newest_stamp_;

  std::vector<double> input_offset_;
  std::map<std::string, double> offset_map_;

  Eigen::Matrix4f computeTransformToAdjustForOldTimestamp(
    const rclcpp::Time & old_stamp, const rclcpp::Time & new_stamp);
  std::unique_ptr<cuda_blackboard::CudaPointCloud2> combineClouds();
  void publish();

  void setPeriod(const int64_t new_period);
  void cloud_callback(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & input_ptr,
    const std::string & topic_name);
  void twist_callback(const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr input);
  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr input);
  void timer_callback();

  void checkConcatStatus(diagnostic_updater::DiagnosticStatusWrapper & stat);
  std::string replaceSyncTopicNamePostfix(
    const std::string & original_topic_name, const std::string & postfix);

  /** \brief processing time publisher. **/
  std::unique_ptr<autoware::universe_utils::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_;
  std::unique_ptr<autoware::universe_utils::DebugPublisher> debug_publisher_;
};

}  // namespace cuda_pointcloud_preprocessor

#endif  // CUDA_POINTCLOUD_PREPROCESSOR__CUDA_CONCATENATE_AND_TIME_SYNC_NODE_HPP_
