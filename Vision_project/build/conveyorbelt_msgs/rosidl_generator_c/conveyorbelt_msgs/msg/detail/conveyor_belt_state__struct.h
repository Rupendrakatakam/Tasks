// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from conveyorbelt_msgs:msg/ConveyorBeltState.idl
// generated code does not contain a copyright notice

#ifndef CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__STRUCT_H_
#define CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/ConveyorBeltState in the package conveyorbelt_msgs.
typedef struct conveyorbelt_msgs__msg__ConveyorBeltState
{
  /// POWER of the belt (percentage, in +Y direction of belt frame).
  double power;
  /// {true} if the power of the belt can be modified // {false} if the belt is stopped.
  bool enabled;
} conveyorbelt_msgs__msg__ConveyorBeltState;

// Struct for a sequence of conveyorbelt_msgs__msg__ConveyorBeltState.
typedef struct conveyorbelt_msgs__msg__ConveyorBeltState__Sequence
{
  conveyorbelt_msgs__msg__ConveyorBeltState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} conveyorbelt_msgs__msg__ConveyorBeltState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__STRUCT_H_
