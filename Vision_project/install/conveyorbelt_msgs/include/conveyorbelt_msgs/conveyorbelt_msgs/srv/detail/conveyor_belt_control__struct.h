// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from conveyorbelt_msgs:srv/ConveyorBeltControl.idl
// generated code does not contain a copyright notice

#ifndef CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__STRUCT_H_
#define CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/ConveyorBeltControl in the package conveyorbelt_msgs.
typedef struct conveyorbelt_msgs__srv__ConveyorBeltControl_Request
{
  /// Desired conveyor bolt POWER.
  double power;
} conveyorbelt_msgs__srv__ConveyorBeltControl_Request;

// Struct for a sequence of conveyorbelt_msgs__srv__ConveyorBeltControl_Request.
typedef struct conveyorbelt_msgs__srv__ConveyorBeltControl_Request__Sequence
{
  conveyorbelt_msgs__srv__ConveyorBeltControl_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} conveyorbelt_msgs__srv__ConveyorBeltControl_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/ConveyorBeltControl in the package conveyorbelt_msgs.
typedef struct conveyorbelt_msgs__srv__ConveyorBeltControl_Response
{
  /// RESULT: SUCCESS message.
  bool success;
} conveyorbelt_msgs__srv__ConveyorBeltControl_Response;

// Struct for a sequence of conveyorbelt_msgs__srv__ConveyorBeltControl_Response.
typedef struct conveyorbelt_msgs__srv__ConveyorBeltControl_Response__Sequence
{
  conveyorbelt_msgs__srv__ConveyorBeltControl_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} conveyorbelt_msgs__srv__ConveyorBeltControl_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CONVEYORBELT_MSGS__SRV__DETAIL__CONVEYOR_BELT_CONTROL__STRUCT_H_
