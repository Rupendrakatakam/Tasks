// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from conveyorbelt_msgs:srv/ConveyorBeltControl.idl
// generated code does not contain a copyright notice

#ifndef CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__BUILDER_HPP_
#define CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "conveyorbelt_msgs/srv/detail/conveyor_belt_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace conveyorbelt_msgs
{

namespace srv
{

namespace builder
{

class Init_ConveyorBeltControl_Request_power
{
public:
  Init_ConveyorBeltControl_Request_power()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::conveyorbelt_msgs::srv::ConveyorBeltControl_Request power(::conveyorbelt_msgs::srv::ConveyorBeltControl_Request::_power_type arg)
  {
    msg_.power = std::move(arg);
    return std::move(msg_);
  }

private:
  ::conveyorbelt_msgs::srv::ConveyorBeltControl_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::conveyorbelt_msgs::srv::ConveyorBeltControl_Request>()
{
  return conveyorbelt_msgs::srv::builder::Init_ConveyorBeltControl_Request_power();
}

}  // namespace conveyorbelt_msgs


namespace conveyorbelt_msgs
{

namespace srv
{

namespace builder
{

class Init_ConveyorBeltControl_Response_success
{
public:
  Init_ConveyorBeltControl_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::conveyorbelt_msgs::srv::ConveyorBeltControl_Response success(::conveyorbelt_msgs::srv::ConveyorBeltControl_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::conveyorbelt_msgs::srv::ConveyorBeltControl_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::conveyorbelt_msgs::srv::ConveyorBeltControl_Response>()
{
  return conveyorbelt_msgs::srv::builder::Init_ConveyorBeltControl_Response_success();
}

}  // namespace conveyorbelt_msgs

#endif  // CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__BUILDER_HPP_
