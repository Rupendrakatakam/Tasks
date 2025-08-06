// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from conveyorbelt_msgs:msg/ConveyorBeltState.idl
// generated code does not contain a copyright notice

#ifndef CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__TRAITS_HPP_
#define CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "conveyorbelt_msgs/msg/detail/conveyor_belt_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace conveyorbelt_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ConveyorBeltState & msg,
  std::ostream & out)
{
  out << "{";
  // member: power
  {
    out << "power: ";
    rosidl_generator_traits::value_to_yaml(msg.power, out);
    out << ", ";
  }

  // member: enabled
  {
    out << "enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.enabled, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ConveyorBeltState & msg,
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

  // member: enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.enabled, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ConveyorBeltState & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace conveyorbelt_msgs

namespace rosidl_generator_traits
{

[[deprecated("use conveyorbelt_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const conveyorbelt_msgs::msg::ConveyorBeltState & msg,
  std::ostream & out, size_t indentation = 0)
{
  conveyorbelt_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use conveyorbelt_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const conveyorbelt_msgs::msg::ConveyorBeltState & msg)
{
  return conveyorbelt_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<conveyorbelt_msgs::msg::ConveyorBeltState>()
{
  return "conveyorbelt_msgs::msg::ConveyorBeltState";
}

template<>
inline const char * name<conveyorbelt_msgs::msg::ConveyorBeltState>()
{
  return "conveyorbelt_msgs/msg/ConveyorBeltState";
}

template<>
struct has_fixed_size<conveyorbelt_msgs::msg::ConveyorBeltState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<conveyorbelt_msgs::msg::ConveyorBeltState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<conveyorbelt_msgs::msg::ConveyorBeltState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__TRAITS_HPP_
