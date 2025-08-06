// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from conveyorbelt_msgs:srv/ConveyorBeltControl.idl
// generated code does not contain a copyright notice

#ifndef CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__TRAITS_HPP_
#define CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "conveyorbelt_msgs/srv/detail/conveyor_belt_control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace conveyorbelt_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const ConveyorBeltControl_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: power
  {
    out << "power: ";
    rosidl_generator_traits::value_to_yaml(msg.power, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ConveyorBeltControl_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: power
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "power: ";
    rosidl_generator_traits::value_to_yaml(msg.power, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ConveyorBeltControl_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace conveyorbelt_msgs

namespace rosidl_generator_traits
{

[[deprecated("use conveyorbelt_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const conveyorbelt_msgs::srv::ConveyorBeltControl_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  conveyorbelt_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use conveyorbelt_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const conveyorbelt_msgs::srv::ConveyorBeltControl_Request & msg)
{
  return conveyorbelt_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>()
{
  return "conveyorbelt_msgs::srv::ConveyorBeltControl_Request";
}

template<>
inline const char * name<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>()
{
  return "conveyorbelt_msgs/srv/ConveyorBeltControl_Request";
}

template<>
struct has_fixed_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace conveyorbelt_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const ConveyorBeltControl_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ConveyorBeltControl_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ConveyorBeltControl_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace conveyorbelt_msgs

namespace rosidl_generator_traits
{

[[deprecated("use conveyorbelt_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const conveyorbelt_msgs::srv::ConveyorBeltControl_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  conveyorbelt_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use conveyorbelt_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const conveyorbelt_msgs::srv::ConveyorBeltControl_Response & msg)
{
  return conveyorbelt_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>()
{
  return "conveyorbelt_msgs::srv::ConveyorBeltControl_Response";
}

template<>
inline const char * name<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>()
{
  return "conveyorbelt_msgs/srv/ConveyorBeltControl_Response";
}

template<>
struct has_fixed_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<conveyorbelt_msgs::srv::ConveyorBeltControl>()
{
  return "conveyorbelt_msgs::srv::ConveyorBeltControl";
}

template<>
inline const char * name<conveyorbelt_msgs::srv::ConveyorBeltControl>()
{
  return "conveyorbelt_msgs/srv/ConveyorBeltControl";
}

template<>
struct has_fixed_size<conveyorbelt_msgs::srv::ConveyorBeltControl>
  : std::integral_constant<
    bool,
    has_fixed_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>::value &&
    has_fixed_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>::value
  >
{
};

template<>
struct has_bounded_size<conveyorbelt_msgs::srv::ConveyorBeltControl>
  : std::integral_constant<
    bool,
    has_bounded_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>::value &&
    has_bounded_size<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>::value
  >
{
};

template<>
struct is_service<conveyorbelt_msgs::srv::ConveyorBeltControl>
  : std::true_type
{
};

template<>
struct is_service_request<conveyorbelt_msgs::srv::ConveyorBeltControl_Request>
  : std::true_type
{
};

template<>
struct is_service_response<conveyorbelt_msgs::srv::ConveyorBeltControl_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__TRAITS_HPP_
