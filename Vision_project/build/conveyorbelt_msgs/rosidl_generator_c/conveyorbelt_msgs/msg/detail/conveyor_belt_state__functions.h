// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from conveyorbelt_msgs:msg/ConveyorBeltState.idl
// generated code does not contain a copyright notice

#ifndef CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__FUNCTIONS_H_
#define CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "conveyorbelt_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "conveyorbelt_msgs/msg/detail/conveyor_belt_state__struct.h"

/// Initialize msg/ConveyorBeltState message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * conveyorbelt_msgs__msg__ConveyorBeltState
 * )) before or use
 * conveyorbelt_msgs__msg__ConveyorBeltState__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
bool
conveyorbelt_msgs__msg__ConveyorBeltState__init(conveyorbelt_msgs__msg__ConveyorBeltState * msg);

/// Finalize msg/ConveyorBeltState message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
void
conveyorbelt_msgs__msg__ConveyorBeltState__fini(conveyorbelt_msgs__msg__ConveyorBeltState * msg);

/// Create msg/ConveyorBeltState message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * conveyorbelt_msgs__msg__ConveyorBeltState__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
conveyorbelt_msgs__msg__ConveyorBeltState *
conveyorbelt_msgs__msg__ConveyorBeltState__create();

/// Destroy msg/ConveyorBeltState message.
/**
 * It calls
 * conveyorbelt_msgs__msg__ConveyorBeltState__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
void
conveyorbelt_msgs__msg__ConveyorBeltState__destroy(conveyorbelt_msgs__msg__ConveyorBeltState * msg);

/// Check for msg/ConveyorBeltState message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
bool
conveyorbelt_msgs__msg__ConveyorBeltState__are_equal(const conveyorbelt_msgs__msg__ConveyorBeltState * lhs, const conveyorbelt_msgs__msg__ConveyorBeltState * rhs);

/// Copy a msg/ConveyorBeltState message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
bool
conveyorbelt_msgs__msg__ConveyorBeltState__copy(
  const conveyorbelt_msgs__msg__ConveyorBeltState * input,
  conveyorbelt_msgs__msg__ConveyorBeltState * output);

/// Initialize array of msg/ConveyorBeltState messages.
/**
 * It allocates the memory for the number of elements and calls
 * conveyorbelt_msgs__msg__ConveyorBeltState__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
bool
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__init(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array, size_t size);

/// Finalize array of msg/ConveyorBeltState messages.
/**
 * It calls
 * conveyorbelt_msgs__msg__ConveyorBeltState__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
void
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__fini(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array);

/// Create array of msg/ConveyorBeltState messages.
/**
 * It allocates the memory for the array and calls
 * conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence *
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__create(size_t size);

/// Destroy array of msg/ConveyorBeltState messages.
/**
 * It calls
 * conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
void
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__destroy(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array);

/// Check for msg/ConveyorBeltState message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
bool
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__are_equal(const conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * lhs, const conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * rhs);

/// Copy an array of msg/ConveyorBeltState messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_conveyorbelt_msgs
bool
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__copy(
  const conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * input,
  conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CONVEYORBELT_MSGS__MSG__DETAIL__CONVEYOR_BELT_STATE__FUNCTIONS_H_
