

#include "cuda_pointcloud_preprocessor/cuda_concatenate_and_time_sync_node.hpp"

#include <pcl_ros/transforms.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define DEFAULT_SYNC_TOPIC_POSTFIX \
  "_synchronized"  // default postfix name for synchronized pointcloud

//////////////////////////////////////////////////////////////////////////////////////////////

namespace cuda_pointcloud_preprocessor
{

std::string getLidarName(const std::string & input)
{
  size_t startPos = input.find_last_of('/', input.find_last_of('/') - 1);
  if (startPos == std::string::npos) {
    return std::string("");  // Return an empty string if '/' is not found
  }
  // Increment startPos to move past the '/' character
  ++startPos;

  // Find the next '/' after the starting position to determine the end of the position string
  size_t endPos = input.find('/', startPos);
  if (endPos == std::string::npos) {
    return std::string("");  // Return an empty string if '/' is not found
  }

  // Extract and return the position string
  return std::string(input.substr(startPos, endPos - startPos));
}

CudaPointCloudConcatenateAndSyncNode::CudaPointCloudConcatenateAndSyncNode(
  const rclcpp::NodeOptions & node_options)
: Node("cuda_point_cloud_concatenator_component", node_options),
  input_twist_topic_type_(declare_parameter<std::string>("input_twist_topic_type", "twist"))
{
  // initialize debug tool
  {
    using autoware::universe_utils::DebugPublisher;
    using autoware::universe_utils::StopWatch;

    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, "concatenate_data_synchronizer");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  // Set parameters
  {
    output_frame_ = static_cast<std::string>(declare_parameter("output_frame", ""));
    if (output_frame_.empty()) {
      RCLCPP_ERROR(get_logger(), "Need an 'output_frame' parameter to be set before continuing!");
      return;
    }
    declare_parameter("input_topics", std::vector<std::string>());
    input_topics_ = get_parameter("input_topics").as_string_array();
    if (input_topics_.empty()) {
      RCLCPP_ERROR(get_logger(), "Need a 'input_topics' parameter to be set before continuing!");
      return;
    }
    if (input_topics_.size() == 1) {
      RCLCPP_ERROR(get_logger(), "Only one topic given. Need at least two topics to continue.");
      return;
    }

    // Optional parameters
    maximum_queue_size_ = static_cast<int>(declare_parameter("max_queue_size", 5));
    timeout_sec_ = static_cast<double>(declare_parameter("timeout_sec", 0.1));

    input_offset_ = declare_parameter("input_offset", std::vector<double>{});
    if (!input_offset_.empty() && input_topics_.size() != input_offset_.size()) {
      RCLCPP_ERROR(get_logger(), "The number of topics does not match the number of offsets.");
      return;
    }

    // Check if publish synchronized pointcloud
    publish_synchronized_pointcloud_ = declare_parameter("publish_synchronized_pointcloud", true);
    keep_input_frame_in_synchronized_pointcloud_ =
      declare_parameter("keep_input_frame_in_synchronized_pointcloud", true);
    synchronized_pointcloud_postfix_ =
      declare_parameter("synchronized_pointcloud_postfix", "pointcloud");
  }

  // Initialize not_subscribed_topic_names_
  {
    for (const std::string & e : input_topics_) {
      not_subscribed_topic_names_.insert(e);
    }
  }

  // Initialize offset map
  {
    for (size_t i = 0; i < input_offset_.size(); ++i) {
      offset_map_[input_topics_[i]] = input_offset_[i];
    }
  }

  // tf2 listener
  {
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
  }

  // Output Publishers
  {
    // pub_output_ = this->create_publisher<PointCloud2>(
    //   "output", rclcpp::SensorDataQoS().keep_last(maximum_queue_size_));
    pub_output_ =
      std::make_shared<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
        *this, "output");
  }

  // Subscribers
  {
    RCLCPP_DEBUG_STREAM(
      get_logger(), "Subscribing to " << input_topics_.size() << " user given topics as inputs:");
    for (auto & input_topic : input_topics_) {
      RCLCPP_DEBUG_STREAM(get_logger(), " - " << input_topic);
    }

    // Subscribe to the filters
    filters_.resize(input_topics_.size());

    // First input_topics_.size () filters are valid
    for (size_t d = 0; d < input_topics_.size(); ++d) {
      cloud_stdmap_.insert(std::make_pair(input_topics_[d], nullptr));
      cloud_stdmap_tmp_ = cloud_stdmap_;

      // CAN'T use auto type here.
      std::function<void(const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> msg)> cb =
        std::bind(
          &CudaPointCloudConcatenateAndSyncNode::cloud_callback, this, std::placeholders::_1,
          input_topics_[d]);

      filters_[d].reset();
      // filters_[d] = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      //   input_topics_[d], rclcpp::SensorDataQoS().keep_last(maximum_queue_size_), cb);
      filters_[d] = std::make_shared<
        cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
        *this, input_topics_[d], false, cb);
    }

    if (input_twist_topic_type_ == "twist") {
      auto twist_cb = std::bind(
        &CudaPointCloudConcatenateAndSyncNode::twist_callback, this, std::placeholders::_1);
      sub_twist_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
        "~/input/twist", rclcpp::QoS{100}, twist_cb);
    } else if (input_twist_topic_type_ == "odom") {
      auto odom_cb = std::bind(
        &CudaPointCloudConcatenateAndSyncNode::odom_callback, this, std::placeholders::_1);
      sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "~/input/odom", rclcpp::QoS{100}, odom_cb);
    } else {
      RCLCPP_ERROR_STREAM(
        get_logger(), "input_twist_topic_type is invalid: " << input_twist_topic_type_);
      throw std::runtime_error("input_twist_topic_type is invalid: " + input_twist_topic_type_);
    }
  }

  // Set timer
  {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(timeout_sec_));
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns,
      std::bind(&CudaPointCloudConcatenateAndSyncNode::timer_callback, this));
  }

  // Diagnostic Updater
  {
    updater_.setHardwareID("concatenate_data_checker");
    updater_.add("concat_status", this, &CudaPointCloudConcatenateAndSyncNode::checkConcatStatus);
  }

  // Cuda wrapper
  cuda_sync_and_concatenate_.setInputTopics(input_topics_);
}

//////////////////////////////////////////////////////////////////////////////////////////////

std::string CudaPointCloudConcatenateAndSyncNode::replaceSyncTopicNamePostfix(
  const std::string & original_topic_name, const std::string & postfix)
{
  std::string replaced_topic_name;
  // separate the topic name by '/' and replace the last element with the new postfix
  size_t pos = original_topic_name.find_last_of("/");
  if (pos == std::string::npos) {
    // not found '/': this is not a namespaced topic
    RCLCPP_WARN_STREAM(
      get_logger(),
      "The topic name is not namespaced. The postfix will be added to the end of the topic name.");
    return original_topic_name + postfix;
  } else {
    // replace the last element with the new postfix
    replaced_topic_name = original_topic_name.substr(0, pos) + "/" + postfix;
  }

  // if topic name is the same with original topic name, add postfix to the end of the topic name
  if (replaced_topic_name == original_topic_name) {
    RCLCPP_WARN_STREAM(
      get_logger(), "The topic name "
                      << original_topic_name
                      << " have the same postfix with synchronized pointcloud. We use the postfix "
                         "to the end of the topic name.");
    replaced_topic_name = original_topic_name + DEFAULT_SYNC_TOPIC_POSTFIX;
  }
  return replaced_topic_name;
}

/**
 * @brief compute transform to adjust for old timestamp
 *
 * @param old_stamp
 * @param new_stamp
 * @return Eigen::Matrix4f: transformation matrix from new_stamp to old_stamp
 */
Eigen::Matrix4f CudaPointCloudConcatenateAndSyncNode::computeTransformToAdjustForOldTimestamp(
  const rclcpp::Time & old_stamp, const rclcpp::Time & new_stamp)
{
  // return identity if no twist is available
  if (twist_ptr_queue_.empty()) {
    RCLCPP_WARN_STREAM_THROTTLE(
      get_logger(), *get_clock(), std::chrono::milliseconds(10000).count(),
      "No twist is available. Please confirm twist topic and timestamp");
    return Eigen::Matrix4f::Identity();
  }

  // return identity if old_stamp is newer than new_stamp
  if (old_stamp > new_stamp) {
    RCLCPP_WARN_STREAM_THROTTLE(
      get_logger(), *get_clock(), std::chrono::milliseconds(10000).count(),
      "old_stamp is newer than new_stamp,");
    return Eigen::Matrix4f::Identity();
  }

  // std::cout << "twist_ptr_queue_.size() = " << twist_ptr_queue_.size() << std::endl;

  auto old_twist_ptr_it = std::lower_bound(
    std::begin(twist_ptr_queue_), std::end(twist_ptr_queue_), old_stamp,
    [](const geometry_msgs::msg::TwistStamped::ConstSharedPtr & x_ptr, const rclcpp::Time & t) {
      return rclcpp::Time(x_ptr->header.stamp) < t;
    });
  old_twist_ptr_it =
    old_twist_ptr_it == twist_ptr_queue_.end() ? (twist_ptr_queue_.end() - 1) : old_twist_ptr_it;

  auto new_twist_ptr_it = std::lower_bound(
    std::begin(twist_ptr_queue_), std::end(twist_ptr_queue_), new_stamp,
    [](const geometry_msgs::msg::TwistStamped::ConstSharedPtr & x_ptr, const rclcpp::Time & t) {
      return rclcpp::Time(x_ptr->header.stamp) < t;
    });
  new_twist_ptr_it =
    new_twist_ptr_it == twist_ptr_queue_.end() ? (twist_ptr_queue_.end() - 1) : new_twist_ptr_it;

  auto prev_time = old_stamp;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  for (auto twist_ptr_it = old_twist_ptr_it; twist_ptr_it != new_twist_ptr_it + 1; ++twist_ptr_it) {
    const double dt =
      (twist_ptr_it != new_twist_ptr_it)
        ? (rclcpp::Time((*twist_ptr_it)->header.stamp) - rclcpp::Time(prev_time)).seconds()
        : (rclcpp::Time(new_stamp) - rclcpp::Time(prev_time)).seconds();

    if (std::fabs(dt) > 0.1) {
      RCLCPP_WARN_STREAM_THROTTLE(
        get_logger(), *get_clock(), std::chrono::milliseconds(10000).count(),
        "Time difference is too large. Cloud not interpolate. Please confirm twist topic and "
        "timestamp");
      break;
    }

    const double dis = (*twist_ptr_it)->twist.linear.x * dt;
    yaw += (*twist_ptr_it)->twist.angular.z * dt;
    x += dis * std::cos(yaw);
    y += dis * std::sin(yaw);
    prev_time = (*twist_ptr_it)->header.stamp;
  }
  Eigen::AngleAxisf rotation_x(0, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rotation_y(0, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rotation_z(yaw, Eigen::Vector3f::UnitZ());
  Eigen::Translation3f translation(x, y, 0);
  Eigen::Matrix4f rotation_matrix = (translation * rotation_z * rotation_y * rotation_x).matrix();
  return rotation_matrix;
}

std::unique_ptr<cuda_blackboard::CudaPointCloud2>
CudaPointCloudConcatenateAndSyncNode::combineClouds()
{
  // Step1. gather stamps and sort it
  std::vector<rclcpp::Time> pc_stamps;
  for (const auto & e : cloud_stdmap_) {
    if (e.second) {
      if (e.second->height * e.second->width == 0) {
        continue;
      }

      pc_stamps.push_back(rclcpp::Time(e.second->header.stamp));
    }
  }
  if (pc_stamps.empty()) {
    return nullptr;
  }
  // sort stamps and get oldest stamp
  std::sort(pc_stamps.begin(), pc_stamps.end());
  std::reverse(pc_stamps.begin(), pc_stamps.end());
  const auto oldest_stamp = pc_stamps.back();
  newest_stamp_ = pc_stamps.front().seconds();

  std::unordered_map<std::string, TransformStruct> transforms_map;
  // Step2. Calculate compensation transform and concatenate with the oldest stamp
  for (const auto & e : cloud_stdmap_) {
    if (e.second != nullptr) {
      if (e.second->height * e.second->width == 0) {
        continue;
      }

      // calculate transforms to oldest stamp
      Eigen::Matrix4f adjust_to_old_data_transform = Eigen::Matrix4f::Identity();
      rclcpp::Time transformed_stamp = rclcpp::Time(e.second->header.stamp);

      for (const auto & stamp : pc_stamps) {
        const auto new_to_old_transform =
          computeTransformToAdjustForOldTimestamp(stamp, transformed_stamp);  // format is old - new
        adjust_to_old_data_transform = new_to_old_transform * adjust_to_old_data_transform;
        transformed_stamp = std::min(transformed_stamp, stamp);
      }

      TransformStruct transform_struct;
      transform_struct.translation_x = adjust_to_old_data_transform(0, 3);
      transform_struct.translation_y = adjust_to_old_data_transform(1, 3);
      transform_struct.translation_z = adjust_to_old_data_transform(2, 3);
      transform_struct.m11 = adjust_to_old_data_transform(0, 0);
      transform_struct.m12 = adjust_to_old_data_transform(0, 1);
      transform_struct.m13 = adjust_to_old_data_transform(0, 2);
      transform_struct.m21 = adjust_to_old_data_transform(1, 0);
      transform_struct.m22 = adjust_to_old_data_transform(1, 1);
      transform_struct.m23 = adjust_to_old_data_transform(1, 2);
      transform_struct.m31 = adjust_to_old_data_transform(2, 0);
      transform_struct.m32 = adjust_to_old_data_transform(2, 1);
      transform_struct.m33 = adjust_to_old_data_transform(2, 2);

      transforms_map[e.first] = transform_struct;
    } else {
      not_subscribed_topic_names_.insert(e.first);
    }
  }

  const double preconcat_latency_ms =
    (rclcpp::Time(this->get_clock()->now()).seconds() - newest_stamp_) * 1e3;
  debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
    "debug/preconcat_latency_ms", preconcat_latency_ms);

  auto concatenated_poincloud = cuda_sync_and_concatenate_.getConcatenatedCloud(transforms_map);
  concatenated_poincloud->header.frame_id = output_frame_;
  concatenated_poincloud->header.stamp = oldest_stamp;

  return concatenated_poincloud;
}

void CudaPointCloudConcatenateAndSyncNode::publish()
{
  stop_watch_ptr_->toc("processing_time", true);

  not_subscribed_topic_names_.clear();

  auto concatenated_pointcloud_msg_ptr = combineClouds();

  const double concat_latency_ms =
    (rclcpp::Time(this->get_clock()->now()).seconds() - newest_stamp_) * 1e3;
  debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
    "debug/concat_latency_ms", concat_latency_ms);

  builtin_interfaces::msg::Time pointcloud_stamp;

  // publish concatenated pointcloud
  if (
    concatenated_pointcloud_msg_ptr &&
    concatenated_pointcloud_msg_ptr->height * concatenated_pointcloud_msg_ptr->width > 0) {
    pointcloud_stamp = concatenated_pointcloud_msg_ptr->header.stamp;
    pub_output_->publish(std::move(concatenated_pointcloud_msg_ptr));

    const double publisher_latency_ms =
      (rclcpp::Time(this->get_clock()->now()).seconds() - newest_stamp_) * 1e3;
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/publisher_latency_ms", publisher_latency_ms);
  } else {
    RCLCPP_WARN(this->get_logger(), "concat_cloud_ptr is nullptr, skipping pointcloud publish.");
  }

  updater_.force_update();

  cloud_stdmap_ = cloud_stdmap_tmp_;
  std::for_each(std::begin(cloud_stdmap_tmp_), std::end(cloud_stdmap_tmp_), [](auto & e) {
    e.second = nullptr;
  });
  // add processing time for debug
  if (debug_publisher_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);

    auto now = rclcpp::Time(this->get_clock()->now());
    double stamp_seconds = now.seconds();

    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/test_latency", stamp_seconds - newest_stamp_);
  }

  cuda_sync_and_concatenate_.preallocate();
}

void CudaPointCloudConcatenateAndSyncNode::setPeriod(const int64_t new_period)
{
  if (!timer_) {
    return;
  }
  int64_t old_period = 0;
  rcl_ret_t ret = rcl_timer_get_period(timer_->get_timer_handle().get(), &old_period);
  if (ret != RCL_RET_OK) {
    rclcpp::exceptions::throw_from_rcl_error(ret, "Couldn't get old period");
  }
  ret = rcl_timer_exchange_period(timer_->get_timer_handle().get(), new_period, &old_period);
  if (ret != RCL_RET_OK) {
    rclcpp::exceptions::throw_from_rcl_error(ret, "Couldn't exchange_period");
  }
}

void CudaPointCloudConcatenateAndSyncNode::cloud_callback(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & input_ptr,
  const std::string & topic_name)
{
  std::string lidar_name = getLidarName(topic_name);

  const double start_latency_ms = (rclcpp::Time(this->get_clock()->now()).seconds() -
                                   rclcpp::Time(input_ptr->header.stamp).seconds()) *
                                  1e3;
  debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
    "debug/" + lidar_name + "/start_latency_ms", start_latency_ms);

  std::lock_guard<std::mutex> lock(mutex_);

  if (input_ptr->height * input_ptr->width == 0) {
    RCLCPP_WARN_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000, "Empty sensor points!");
  }

  const bool is_already_subscribed_this = (cloud_stdmap_[topic_name] != nullptr);
  const bool is_already_subscribed_tmp = std::any_of(
    std::begin(cloud_stdmap_tmp_), std::end(cloud_stdmap_tmp_),
    [](const auto & e) { return e.second != nullptr; });

  if (is_already_subscribed_this) {
    cuda_sync_and_concatenate_.setPointcloud(topic_name, input_ptr);
    cloud_stdmap_tmp_[topic_name] = input_ptr;

    if (!is_already_subscribed_tmp) {
      auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timeout_sec_));
      try {
        setPeriod(period.count());
      } catch (rclcpp::exceptions::RCLError & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
      }
      timer_->reset();
    }
  } else {
    cloud_stdmap_[topic_name] = input_ptr;
    cuda_sync_and_concatenate_.setPointcloud(topic_name, input_ptr);

    const bool is_subscribed_all = std::all_of(
      std::begin(cloud_stdmap_), std::end(cloud_stdmap_),
      [](const auto & e) { return e.second != nullptr; });

    if (is_subscribed_all) {
      for (const auto & e : cloud_stdmap_tmp_) {
        if (e.second != nullptr) {
          cloud_stdmap_[e.first] = e.second;
        }
      }
      std::for_each(std::begin(cloud_stdmap_tmp_), std::end(cloud_stdmap_tmp_), [](auto & e) {
        e.second = nullptr;
      });

      timer_->cancel();
      publish();
    } else if (offset_map_.size() > 0) {
      timer_->cancel();
      auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timeout_sec_ - offset_map_[topic_name]));
      try {
        setPeriod(period.count());
      } catch (rclcpp::exceptions::RCLError & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
      }
      timer_->reset();
    }
  }
}

void CudaPointCloudConcatenateAndSyncNode::timer_callback()
{
  using std::chrono_literals::operator""ms;
  timer_->cancel();
  if (mutex_.try_lock()) {
    publish();
    mutex_.unlock();
  } else {
    try {
      std::chrono::nanoseconds period = 10ms;
      setPeriod(period.count());
    } catch (rclcpp::exceptions::RCLError & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
    }
    timer_->reset();
  }
}

void CudaPointCloudConcatenateAndSyncNode::twist_callback(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr input)
{
  // if rosbag restart, clear buffer
  if (!twist_ptr_queue_.empty()) {
    if (rclcpp::Time(twist_ptr_queue_.front()->header.stamp) > rclcpp::Time(input->header.stamp)) {
      twist_ptr_queue_.clear();
    }
  }

  // pop old data
  while (!twist_ptr_queue_.empty()) {
    if (
      rclcpp::Time(twist_ptr_queue_.front()->header.stamp) + rclcpp::Duration::from_seconds(1.0) >
      rclcpp::Time(input->header.stamp)) {
      break;
    }
    twist_ptr_queue_.pop_front();
  }

  auto twist_ptr = std::make_shared<geometry_msgs::msg::TwistStamped>();
  twist_ptr->header = input->header;
  twist_ptr->twist = input->twist.twist;
  twist_ptr_queue_.push_back(twist_ptr);
}

void CudaPointCloudConcatenateAndSyncNode::odom_callback(
  const nav_msgs::msg::Odometry::ConstSharedPtr input)
{
  // if rosbag restart, clear buffer
  if (!twist_ptr_queue_.empty()) {
    if (rclcpp::Time(twist_ptr_queue_.front()->header.stamp) > rclcpp::Time(input->header.stamp)) {
      twist_ptr_queue_.clear();
    }
  }

  // pop old data
  while (!twist_ptr_queue_.empty()) {
    if (
      rclcpp::Time(twist_ptr_queue_.front()->header.stamp) + rclcpp::Duration::from_seconds(1.0) >
      rclcpp::Time(input->header.stamp)) {
      break;
    }
    twist_ptr_queue_.pop_front();
  }

  auto twist_ptr = std::make_shared<geometry_msgs::msg::TwistStamped>();
  twist_ptr->header = input->header;
  twist_ptr->twist = input->twist.twist;
  twist_ptr_queue_.push_back(twist_ptr);
}

void CudaPointCloudConcatenateAndSyncNode::checkConcatStatus(
  diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  for (const std::string & e : input_topics_) {
    const std::string subscribe_status = not_subscribed_topic_names_.count(e) ? "NG" : "OK";
    stat.add(e, subscribe_status);
  }

  const int8_t level = not_subscribed_topic_names_.empty()
                         ? diagnostic_msgs::msg::DiagnosticStatus::OK
                         : diagnostic_msgs::msg::DiagnosticStatus::WARN;
  const std::string message = not_subscribed_topic_names_.empty()
                                ? "Concatenate all topics"
                                : "Some topics are not concatenated";
  stat.summary(level, message);
}
}  // namespace cuda_pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(cuda_pointcloud_preprocessor::CudaPointCloudConcatenateAndSyncNode)
