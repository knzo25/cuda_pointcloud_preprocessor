
#include "cuda_pointcloud_preprocessor/cuda_pointcloud_preprocessor.hpp"
#include "cuda_pointcloud_preprocessor/point_types.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cuda_runtime.h>
#include <tf2/utils.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace cuda_pointcloud_preprocessor
{

__host__ __device__ Eigen::Matrix3f skewSymmetric(const Eigen::Vector3f & v)
{
  Eigen::Matrix3f m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

__host__ __device__ Eigen::Matrix3f leftJacobianSO3(const Eigen::Vector3f & omega)
{
  double theta = omega.norm();
  if (std::abs(theta) < 1e-6) {
    return Eigen::Matrix3f::Identity();
  }

  Eigen::Matrix3f Omega = skewSymmetric(omega);

  Eigen::Matrix3f Omega2 = Omega * Omega;
  double theta2 = theta * theta;
  double theta3 = theta2 * theta;

  // Rodrigues' formula for Jacobian
  return Eigen::Matrix3f::Identity() + ((1 - cos(theta)) / theta2) * Omega +
         ((theta - sin(theta)) / theta3) * Omega2;
}

__host__ __device__ Eigen::Matrix4f transformationMatrixFromVelocity(
  const Eigen::Vector3f & linear_velocity, const Eigen::Vector3f & angular_velocity, double dt)
{
  Eigen::Matrix3f R = Eigen::AngleAxisf(angular_velocity.norm() * dt, angular_velocity.normalized())
                        .toRotationMatrix();
  Eigen::Matrix3f J = leftJacobianSO3(angular_velocity * dt);

  Eigen::Vector3f translation = J * (linear_velocity * dt);

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = translation;

  return transformation;
}

__global__ void transform_points_kernel(
  const InputPointType * input_points, InputPointType * output_points, int num_points,
  TransformStruct transform)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    output_points[idx] = input_points[idx];

    float x = input_points[idx].x;
    float y = input_points[idx].y;
    float z = input_points[idx].z;

    output_points[idx].x = transform.m11 * x + transform.m12 * y + transform.m13 * z + transform.x;
    output_points[idx].y = transform.m21 * x + transform.m22 * y + transform.m23 * z + transform.y;
    output_points[idx].z = transform.m31 * x + transform.m32 * y + transform.m33 * z + transform.z;
  }
}

__global__ void crop_box_kernel(
  InputPointType * d_points, uint32_t * output_mask, int num_points, float min_x, float min_y,
  float min_z, float max_x, float max_y, float max_z)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    float x = d_points[idx].x;
    float y = d_points[idx].y;
    float z = d_points[idx].z;

    if (x > min_x && x < max_x && y > min_y && y < max_y && z > min_z && z < max_z) {
      output_mask[idx] = 0;
    } else {
      output_mask[idx] = 1;
    }
  }
}

__global__ void combine_masks_kernel(
  uint32_t * mask1, uint32_t * mask2, uint32_t * mask3, int num_points, uint32_t * output_mask)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    output_mask[idx] = mask1[idx] & mask2[idx] & mask3[idx];
  }
}

__global__ void extract_input_point_indices_kernel(
  InputPointType * input_points, uint32_t * masks, uint32_t * indices, int num_points,
  InputPointType * output_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points && masks[idx] == 1) {
    output_points[indices[idx] - 1] = input_points[idx];
  }
}

__global__ void extract_output_point_indices_kernel(
  OutputPointType * input_points, uint32_t * masks, uint32_t * indices, int num_points,
  OutputPointType * output_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points && masks[idx] == 1) {
    output_points[indices[idx] - 1] = input_points[idx];
  }
}

__global__ void extract_input_points_to_output_points_indices_kernel(
  InputPointType * input_points, uint32_t * masks, uint32_t * indices, int num_points,
  OutputPointType * output_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points && masks[idx] == 1) {
    InputPointType & input_point = input_points[idx];
    OutputPointType & output_point = output_points[indices[idx] - 1];
    output_point.x = input_point.x;
    output_point.y = input_point.y;
    output_point.z = input_point.z;
    output_point.intensity = input_point.intensity;
    output_point.return_type = input_point.return_type;
    output_point.channel = input_point.channel;
  }
}

__global__ void undistort_2d_kernel(
  InputPointType * input_points, int num_points, TwistStruct2D * twist_structs, int num_twists)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    InputPointType & point = input_points[idx];

    // The twist must always be newer than the point ! (or it was the last twist)
    int twist_index = 0;
    while (twist_index < num_twists && twist_structs[twist_index].stamp_nsec < point.time_stamp) {
      twist_index++;
    }

    twist_index = min(twist_index, num_twists - 1);

    TwistStruct2D twist = twist_structs[twist_index];
    float x = twist.cum_x;
    float y = twist.cum_y;
    float theta = twist.cum_theta;

    double dt_nsec =
      point.time_stamp > twist.last_stamp_nsec ? point.time_stamp - twist.last_stamp_nsec : 0;
    double dt = 1e-9 * (dt_nsec);

    theta += twist.vtheta * dt;
    float d = twist.vx * dt;
    x += d * cos(theta);
    y += d * sin(theta);

    /* point.x += x * cos(theta) - y * sin(theta);
    point.y += x * sin(theta) + y * cos(theta); */
    float distorted_x = point.x;
    float distorted_y = point.y;

    point.x = distorted_x * cos(theta) - distorted_y * sin(theta) + x;
    point.y = distorted_x * sin(theta) + distorted_y * cos(theta) + y;
  }
}

__global__ void undistort_3d_kernel(
  InputPointType * input_points, int num_points, TwistStruct3D * twist_structs, int num_twists)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    InputPointType & point = input_points[idx];

    // The twist must always be newer than the point ! (or it was the last twist)
    int twist_index = 0;
    while (twist_index < num_twists && twist_structs[twist_index].stamp_nsec < point.time_stamp) {
      twist_index++;
    }

    twist_index = min(twist_index, num_twists - 1);

    TwistStruct3D twist = twist_structs[twist_index];
    Eigen::Map<Eigen::Matrix4f> cum_transform_buffer_map(twist.cum_transform_buffer);
    Eigen::Map<Eigen::Vector3f> v_map(twist.v);
    Eigen::Map<Eigen::Vector3f> w_map(twist.w);

    double dt_nsec =
      point.time_stamp > twist.last_stamp_nsec ? point.time_stamp - twist.last_stamp_nsec : 0;
    double dt = 1e-9 * (dt_nsec);

    Eigen::Matrix4f transform =
      cum_transform_buffer_map * transformationMatrixFromVelocity(v_map, w_map, dt);
    Eigen::Vector3f p(point.x, point.y, point.z);
    p = transform.block<3, 3>(0, 0) * p + transform.block<3, 1>(0, 3);

    point.x = p.x();
    point.y = p.y();
    point.z = p.z();
  }
}

__global__ void ring_outlier_filter_kernel(
  const InputPointType * d_points, uint32_t * output_mask, int num_rings, int max_points_per_ring,
  float distance_ratio, float object_length_threshold_squared, int num_points_threshold)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int j = idx / max_points_per_ring;
  int i = idx % max_points_per_ring;

  if (j >= num_rings || i >= max_points_per_ring) {
    return;
  }

  int min_i = max(i - num_points_threshold, 0);
  int max_i = min(i + num_points_threshold, max_points_per_ring);

  int walk_size = 1;
  int left_idx = min_i;
  int right_idx = min_i + 1;

  for (int k = min_i; k < max_i - 1; k++) {
    const InputPointType & left_point = d_points[j * max_points_per_ring + k];
    const InputPointType & right_point = d_points[j * max_points_per_ring + k + 1];

    // Find biggest walk that passes through i
    float azimuth_diff = right_point.azimuth - left_point.azimuth;
    azimuth_diff = azimuth_diff < 0.f ? azimuth_diff + 2 * M_PI : azimuth_diff;

    if (
      max(left_point.distance, right_point.distance) <
        min(left_point.distance, right_point.distance) * distance_ratio &&
      azimuth_diff < 1.f * M_PI / 180.f) {
      // Determined to be included in the same walk
      walk_size++;
      right_idx++;
    } else if (k >= i) {
      break;
    } else {
      walk_size = 1;
      left_idx = k + 1;
      right_idx = k + 2;  // this is safe since we break if k >= i
    }
  }

  const InputPointType & left_point = d_points[j * max_points_per_ring + left_idx];
  const InputPointType & right_point = d_points[j * max_points_per_ring + right_idx - 1];
  const float x = left_point.x - right_point.x;
  const float y = left_point.y - right_point.y;
  const float z = left_point.z - right_point.z;

  output_mask[j * max_points_per_ring + i] =
    ((walk_size > num_points_threshold) ||
     (x * x + y * y + z * z >= object_length_threshold_squared))
      ? 1
      : 0;
}

__global__ void transform_point_type_kernel(
  const InputPointType * device_input_points, int num_points, OutputPointType * device_ouput_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    const InputPointType & input_point = device_input_points[idx];
    OutputPointType & output_point = device_ouput_points[idx];

    output_point.x = input_point.x;
    output_point.y = input_point.y;
    output_point.z = input_point.z;
    output_point.intensity = (float)input_point.intensity;
  }
}

CudaPointcloudPreprocessor::CudaPointcloudPreprocessor()
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

void CudaPointcloudPreprocessor::setCropBoxParameters(
  const CropBoxParameters & self_crop_box_parameters,
  const CropBoxParameters & mirror_crop_box_parameters)
{
  self_crop_box_parameters_ = self_crop_box_parameters;
  mirror_crop_box_parameters_ = mirror_crop_box_parameters;
}

void CudaPointcloudPreprocessor::setRingOutlierFilterParameters(
  const RingOutlierFilterParameters & ring_outlier_parameters)
{
  ring_outlier_parameters_ = ring_outlier_parameters;
}

void CudaPointcloudPreprocessor::set3DUndistortion(bool use_3d_undistortion)
{
  use_3d_undistortion_ = use_3d_undistortion;
}

void CudaPointcloudPreprocessor::preallocateOutput()
{
  output_pointcloud_ptr_ = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  cudaMalloc(
    reinterpret_cast<void **>(&output_pointcloud_ptr_->data),
    max_rings_ * max_points_per_ring_ * sizeof(OutputPointType));
}

void CudaPointcloudPreprocessor::setupTwist2DStructs(
  const cuda_blackboard::CudaPointCloud2 & input_pointcloud_msg,
  const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> & twist_queue,
  const std::deque<geometry_msgs::msg::Vector3Stamped> & angular_velocity_queue)
{
  const InputPointType * device_input_points =
    reinterpret_cast<const InputPointType *>(input_pointcloud_msg.data);
  InputPointType first_point;
  cudaMemcpy(&first_point, &device_input_points[0], sizeof(InputPointType), cudaMemcpyDeviceToHost);

  // Twist preprocessing

  uint64_t pointcloud_stamp_nsec = 1'000'000'000 * input_pointcloud_msg.header.stamp.sec +
                                   input_pointcloud_msg.header.stamp.nanosec;

  thrust::host_vector<TwistStruct2D> host_twist_structs;

  float cum_x = 0;
  float cum_y = 0;
  float cum_theta = 0;
  // All time stamps from now on are in nsec from the "beginning of the pointcloud"
  uint32_t last_stamp_nsec = first_point.time_stamp;

  /* for (int i = 0; i < num_twists; i++) {
    host_twist_structs[i].cum_x = cum_x;
    host_twist_structs[i].cum_y = cum_y;
    host_twist_structs[i].cum_theta = cum_theta;

    uint64_t twist_global_stamp_nsec = 1'000'000'000 *
  static_cast<uint64_t>(twist_queue[i].header.stamp.sec) +
  static_cast<uint64_t>(twist_queue[i].header.stamp.nanosec); assert(twist_global_stamp_nsec >
  pointcloud_stamp_nsec); // by construction uint32_t twist_from_pointcloud_start_nsec =
  twist_global_stamp_nsec - pointcloud_stamp_nsec;

    host_twist_structs[i].stamp_nsec = twist_from_pointcloud_start_nsec;
    host_twist_structs[i].vx = twist_queue[i].twist.twist.linear.x;
    host_twist_structs[i].vtheta = twist_queue[i].twist.twist.angular.z;
    host_twist_structs[i].last_stamp_nsec = last_stamp_nsec;

    double dt_seconds = 1e-9*(host_twist_structs[i].stamp_nsec - last_stamp_nsec);
    last_stamp_nsec = host_twist_structs[i].stamp_nsec;
    cum_theta += twist_queue[i].twist.twist.angular.z * dt_seconds;
    float d = host_twist_structs[i].vx * dt_seconds;
    cum_x += d * cos(cum_theta);
    cum_y += d * sin(cum_theta);
  } */

  std::size_t twist_index = 0;
  std::size_t angular_velocity_index = 0;

  for (; twist_index < twist_queue.size() ||
         angular_velocity_index < angular_velocity_queue.size();) {
    // std::cout << "twist_index: " << twist_index << " angular_velocity_index: " <<
    // angular_velocity_index << std::endl;
    uint64_t twist_stamp, input_twist_global_stamp_nsec, angular_velocity_global_stamp_nsec;
    float vx, vtheta;

    if (twist_index < twist_queue.size()) {
      input_twist_global_stamp_nsec =
        1'000'000'000 * static_cast<uint64_t>(twist_queue[twist_index].header.stamp.sec) +
        static_cast<uint64_t>(twist_queue[twist_index].header.stamp.nanosec);
      vx = twist_queue[twist_index].twist.twist.linear.x;
    } else {
      input_twist_global_stamp_nsec = std::numeric_limits<uint64_t>::max();
      vx = 0.0;
    }

    if (angular_velocity_index < angular_velocity_queue.size()) {
      angular_velocity_global_stamp_nsec =
        1'000'000'000 *
          static_cast<uint64_t>(angular_velocity_queue[angular_velocity_index].header.stamp.sec) +
        static_cast<uint64_t>(angular_velocity_queue[angular_velocity_index].header.stamp.nanosec);
      vtheta = angular_velocity_queue[angular_velocity_index].vector.z;
    } else {
      angular_velocity_global_stamp_nsec = std::numeric_limits<uint64_t>::max();
      vtheta = 0.0;
    }

    if (input_twist_global_stamp_nsec < angular_velocity_global_stamp_nsec) {
      twist_stamp = input_twist_global_stamp_nsec;
      twist_index++;
    } else if (input_twist_global_stamp_nsec > angular_velocity_global_stamp_nsec) {
      twist_stamp = angular_velocity_global_stamp_nsec;
      angular_velocity_index++;
    } else {
      twist_index++;
      angular_velocity_index++;
    }

    TwistStruct2D twist;
    twist.cum_x = cum_x;
    twist.cum_y = cum_y;
    twist.cum_theta = cum_theta;

    uint64_t twist_global_stamp_nsec = twist_stamp;
    assert(twist_global_stamp_nsec > pointcloud_stamp_nsec);  // by construction
    uint32_t twist_from_pointcloud_start_nsec = twist_global_stamp_nsec - pointcloud_stamp_nsec;

    twist.stamp_nsec = twist_from_pointcloud_start_nsec;
    twist.vx = vx;
    twist.vtheta = vtheta;
    twist.last_stamp_nsec = last_stamp_nsec;
    host_twist_structs.push_back(twist);

    double dt_seconds = 1e-9 * (twist.stamp_nsec - last_stamp_nsec);
    last_stamp_nsec = twist.stamp_nsec;
    cum_theta += vtheta * dt_seconds;
    float d = twist.vx * dt_seconds;
    cum_x += d * cos(cum_theta);
    cum_y += d * sin(cum_theta);
  }

  // std::cout << "[" << input_pointcloud_msg.header.frame_id << "] cum_x: " << cum_x << " cum_y: "
  // << cum_y << " cum_theta: " << cum_theta << std::endl;

  /* std::cout << "[" << input_pointcloud_msg.header.frame_id << "] Pointcloud stamp: " <<
  input_pointcloud_msg.header.stamp.sec << "." << input_pointcloud_msg.header.stamp.nanosec <<
  std::endl;

  for (const auto & twist : twist_queue) {
    std::cout << "[" << input_pointcloud_msg.header.frame_id << "] \tTwist stamp: " <<
  twist.header.stamp.sec << "." << twist.header.stamp.nanosec << std::endl;
  }
  for (const auto & angular_velocity : angular_velocity_queue) {
    std::cout << "[" << input_pointcloud_msg.header.frame_id << "] \tAngular velocity stamp: " <<
  angular_velocity.header.stamp.sec << "." << angular_velocity.header.stamp.nanosec << std::endl;
  }
  std::cout << "[" << input_pointcloud_msg.header.frame_id << "] \tTwist queue size: " <<
  twist_queue.size() << std::endl; std::cout << "[" << input_pointcloud_msg.header.frame_id << "]
  \tAngular velocity queue size: " << angular_velocity_queue.size() << std::endl; std::cout << "["
  << input_pointcloud_msg.header.frame_id << "] \tTwist structs size: " << host_twist_structs.size()
  << std::endl << std::endl;
 */
  // Copy to device
  device_twist_2d_structs_ = host_twist_structs;
}

void CudaPointcloudPreprocessor::setupTwist3DStructs(
  const cuda_blackboard::CudaPointCloud2 & input_pointcloud_msg,
  const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> & twist_queue,
  const std::deque<geometry_msgs::msg::Vector3Stamped> & angular_velocity_queue)
{
  const InputPointType * device_input_points =
    reinterpret_cast<const InputPointType *>(input_pointcloud_msg.data);
  InputPointType first_point;
  cudaMemcpy(&first_point, &device_input_points[0], sizeof(InputPointType), cudaMemcpyDeviceToHost);

  // Twist preprocessing

  uint64_t pointcloud_stamp_nsec = 1'000'000'000 * input_pointcloud_msg.header.stamp.sec +
                                   input_pointcloud_msg.header.stamp.nanosec;

  thrust::host_vector<TwistStruct3D> host_twist_structs;

  Eigen::Matrix4f cum_transform = Eigen::Matrix4f::Identity();
  Eigen::Vector3f v = Eigen::Vector3f::Zero();
  Eigen::Vector3f w = Eigen::Vector3f::Zero();

  // All time stamps from now on are in nsec from the "beginning of the pointcloud"
  uint32_t last_stamp_nsec = first_point.time_stamp;

  std::size_t twist_index = 0;
  std::size_t angular_velocity_index = 0;

  for (; twist_index < twist_queue.size() ||
         angular_velocity_index < angular_velocity_queue.size();) {
    // std::cout << "twist_index: " << twist_index << " angular_velocity_index: " <<
    // angular_velocity_index << std::endl;
    uint64_t twist_stamp, input_twist_global_stamp_nsec, angular_velocity_global_stamp_nsec;
    // float vx, vtheta;

    if (twist_index < twist_queue.size()) {
      input_twist_global_stamp_nsec =
        1'000'000'000 * static_cast<uint64_t>(twist_queue[twist_index].header.stamp.sec) +
        static_cast<uint64_t>(twist_queue[twist_index].header.stamp.nanosec);
      v.x() = twist_queue[twist_index].twist.twist.linear.x;
      v.y() = twist_queue[twist_index].twist.twist.linear.y;
      v.z() = twist_queue[twist_index].twist.twist.linear.z;
    } else {
      input_twist_global_stamp_nsec = std::numeric_limits<uint64_t>::max();
      v = Eigen::Vector3f::Zero();
    }

    if (angular_velocity_index < angular_velocity_queue.size()) {
      angular_velocity_global_stamp_nsec =
        1'000'000'000 *
          static_cast<uint64_t>(angular_velocity_queue[angular_velocity_index].header.stamp.sec) +
        static_cast<uint64_t>(angular_velocity_queue[angular_velocity_index].header.stamp.nanosec);
      w.x() = angular_velocity_queue[angular_velocity_index].vector.x;
      w.y() = angular_velocity_queue[angular_velocity_index].vector.y;
      w.z() = angular_velocity_queue[angular_velocity_index].vector.z;
    } else {
      angular_velocity_global_stamp_nsec = std::numeric_limits<uint64_t>::max();
      w = Eigen::Vector3f::Zero();
    }

    if (input_twist_global_stamp_nsec < angular_velocity_global_stamp_nsec) {
      twist_stamp = input_twist_global_stamp_nsec;
      twist_index++;
    } else if (input_twist_global_stamp_nsec > angular_velocity_global_stamp_nsec) {
      twist_stamp = angular_velocity_global_stamp_nsec;
      angular_velocity_index++;
    } else {
      twist_index++;
      angular_velocity_index++;
    }

    TwistStruct3D twist;

    Eigen::Map<Eigen::Matrix4f> cum_transform_buffer_map(twist.cum_transform_buffer);
    Eigen::Map<Eigen::Vector3f> v_map(twist.v);
    Eigen::Map<Eigen::Vector3f> w_map(twist.w);
    cum_transform_buffer_map = cum_transform;

    uint64_t twist_global_stamp_nsec = twist_stamp;
    assert(twist_global_stamp_nsec > pointcloud_stamp_nsec);  // by construction
    uint32_t twist_from_pointcloud_start_nsec = twist_global_stamp_nsec - pointcloud_stamp_nsec;

    twist.stamp_nsec = twist_from_pointcloud_start_nsec;
    v_map = v;
    w_map = w;
    twist.last_stamp_nsec = last_stamp_nsec;
    host_twist_structs.push_back(twist);

    double dt_seconds = 1e-9 * (twist.stamp_nsec - last_stamp_nsec);
    last_stamp_nsec = twist.stamp_nsec;

    /* cum_theta += vtheta * dt_seconds;
    float d = twist.vx * dt_seconds;
    cum_x += d * cos(cum_theta);
    cum_y += d * sin(cum_theta); */

    auto delta_transform = transformationMatrixFromVelocity(v, w, dt_seconds);
    cum_transform = cum_transform * delta_transform;
  }

  // Copy to device
  device_twist_3d_structs_ = host_twist_structs;
}

std::unique_ptr<cuda_blackboard::CudaPointCloud2> CudaPointcloudPreprocessor::process(
  const cuda_blackboard::CudaPointCloud2 & input_pointcloud_msg,
  const geometry_msgs::msg::TransformStamped & transform_msg,
  const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> & twist_queue,
  const std::deque<geometry_msgs::msg::Vector3Stamped> & angular_velocity_queue)
{
  auto frame_id = input_pointcloud_msg.header.frame_id;

  auto num_input_points = input_pointcloud_msg.width * input_pointcloud_msg.height;
  if (
    input_pointcloud_msg.width * input_pointcloud_msg.height > max_rings_ * max_points_per_ring_) {
    max_rings_ = input_pointcloud_msg.height;
    max_points_per_ring_ = input_pointcloud_msg.width;

    device_transformed_points_.resize(num_input_points);
    device_self_crop_mask_.resize(num_input_points);
    device_mirror_crop_mask_.resize(num_input_points);
    device_ring_outlier_mask_.resize(num_input_points);
    device_indices_.resize(num_input_points);

    preallocateOutput();
  }

  tf2::Quaternion rotation_quaternion(
    transform_msg.transform.rotation.x, transform_msg.transform.rotation.y,
    transform_msg.transform.rotation.z, transform_msg.transform.rotation.w);
  tf2::Matrix3x3 rotation_matrix;
  rotation_matrix.setRotation(rotation_quaternion);

  TransformStruct transform_struct;
  transform_struct.x = static_cast<float>(transform_msg.transform.translation.x);
  transform_struct.y = static_cast<float>(transform_msg.transform.translation.y);
  transform_struct.z = static_cast<float>(transform_msg.transform.translation.z);
  transform_struct.m11 = static_cast<float>(rotation_matrix[0][0]);
  transform_struct.m12 = static_cast<float>(rotation_matrix[0][1]);
  transform_struct.m13 = static_cast<float>(rotation_matrix[0][2]);
  transform_struct.m21 = static_cast<float>(rotation_matrix[1][0]);
  transform_struct.m22 = static_cast<float>(rotation_matrix[1][1]);
  transform_struct.m23 = static_cast<float>(rotation_matrix[1][2]);
  transform_struct.m31 = static_cast<float>(rotation_matrix[2][0]);
  transform_struct.m32 = static_cast<float>(rotation_matrix[2][1]);
  transform_struct.m33 = static_cast<float>(rotation_matrix[2][2]);

  // Twist preprocessing
  const InputPointType * device_input_points =
    reinterpret_cast<const InputPointType *>(input_pointcloud_msg.data);
  // uint64_t pointcloud_stamp_nsec = 1'000'000'000 * input_pointcloud_msg.header.stamp.sec +
  // input_pointcloud_msg.header.stamp.nanosec; std::size_t num_twists = std::min<std::size_t>(20,
  // twist_queue.size()); std::size_t num_twists = std::min<std::size_t>(20, twist_queue.size() +
  // angular_velocity_queue.size());

  if (use_3d_undistortion_) {
    setupTwist3DStructs(input_pointcloud_msg, twist_queue, angular_velocity_queue);
  } else {
    setupTwist2DStructs(input_pointcloud_msg, twist_queue, angular_velocity_queue);
  }

  // Obtain raw pointers for the kernels
  TwistStruct2D * device_twist_2d_structs =
    thrust::raw_pointer_cast(device_twist_2d_structs_.data());
  TwistStruct3D * device_twist_3d_structs =
    thrust::raw_pointer_cast(device_twist_3d_structs_.data());
  InputPointType * device_transformed_points =
    thrust::raw_pointer_cast(device_transformed_points_.data());
  uint32_t * device_self_crop_mask = thrust::raw_pointer_cast(device_self_crop_mask_.data());
  uint32_t * device_mirror_crop_mask = thrust::raw_pointer_cast(device_mirror_crop_mask_.data());
  uint32_t * device_ring_outlier_mask = thrust::raw_pointer_cast(device_ring_outlier_mask_.data());
  uint32_t * device_indices = thrust::raw_pointer_cast(device_indices_.data());

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (num_input_points + threadsPerBlock - 1) / threadsPerBlock;

  transform_points_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    device_input_points, device_transformed_points, num_input_points, transform_struct);

  crop_box_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    device_transformed_points, device_self_crop_mask, num_input_points,
    self_crop_box_parameters_.min_x, self_crop_box_parameters_.min_y,
    self_crop_box_parameters_.min_z, self_crop_box_parameters_.max_x,
    self_crop_box_parameters_.max_y, self_crop_box_parameters_.max_z);

  crop_box_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    device_transformed_points, device_mirror_crop_mask, num_input_points,
    mirror_crop_box_parameters_.min_x, mirror_crop_box_parameters_.min_y,
    mirror_crop_box_parameters_.min_z, mirror_crop_box_parameters_.max_x,
    mirror_crop_box_parameters_.max_y, mirror_crop_box_parameters_.max_z);

  if (use_3d_undistortion_ && device_twist_3d_structs_.size() > 0) {
    undistort_3d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      device_transformed_points, num_input_points, device_twist_3d_structs,
      device_twist_3d_structs_.size());
  } else if (!use_3d_undistortion_ && device_twist_2d_structs_.size() > 0) {
    undistort_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      device_transformed_points, num_input_points, device_twist_2d_structs,
      device_twist_2d_structs_.size());
  }

  ring_outlier_filter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    device_transformed_points, device_ring_outlier_mask, max_rings_, max_points_per_ring_,
    ring_outlier_parameters_.distance_ratio,
    ring_outlier_parameters_.object_length_threshold *
      ring_outlier_parameters_.object_length_threshold,
    ring_outlier_parameters_.num_points_threshold);

  combine_masks_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    device_self_crop_mask, device_mirror_crop_mask, device_ring_outlier_mask, num_input_points,
    device_ring_outlier_mask);

  thrust::inclusive_scan(
    thrust::device, device_ring_outlier_mask, device_ring_outlier_mask + num_input_points,
    device_indices);

  uint32_t num_output_points;
  cudaMemcpy(
    &num_output_points, device_indices + num_input_points - 1, sizeof(uint32_t),
    cudaMemcpyDeviceToHost);

  if (num_output_points > 0) {
    extract_input_points_to_output_points_indices_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      device_transformed_points, device_ring_outlier_mask, device_indices, num_input_points,
      reinterpret_cast<OutputPointType *>(output_pointcloud_ptr_->data));
  }

  // Copy the transformed points back
  output_pointcloud_ptr_->row_step = num_output_points * sizeof(OutputPointType);
  output_pointcloud_ptr_->width = num_output_points;
  output_pointcloud_ptr_->height = 1;

  output_pointcloud_ptr_->fields = point_fields_;
  output_pointcloud_ptr_->is_dense = true;
  output_pointcloud_ptr_->is_bigendian = input_pointcloud_msg.is_bigendian;
  output_pointcloud_ptr_->point_step = sizeof(OutputPointType);
  output_pointcloud_ptr_->header.stamp = input_pointcloud_msg.header.stamp;

  return std::move(output_pointcloud_ptr_);
}

}  // namespace cuda_pointcloud_preprocessor
