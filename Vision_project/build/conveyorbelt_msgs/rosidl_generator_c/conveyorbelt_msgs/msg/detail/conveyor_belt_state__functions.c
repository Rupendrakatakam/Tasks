// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from conveyorbelt_msgs:msg/ConveyorBeltState.idl
// generated code does not contain a copyright notice
#include "conveyorbelt_msgs/msg/detail/conveyor_belt_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
conveyorbelt_msgs__msg__ConveyorBeltState__init(conveyorbelt_msgs__msg__ConveyorBeltState * msg)
{
  if (!msg) {
    return false;
  }
  // power
  // enabled
  return true;
}

void
conveyorbelt_msgs__msg__ConveyorBeltState__fini(conveyorbelt_msgs__msg__ConveyorBeltState * msg)
{
  if (!msg) {
    return;
  }
  // power
  // enabled
}

bool
conveyorbelt_msgs__msg__ConveyorBeltState__are_equal(const conveyorbelt_msgs__msg__ConveyorBeltState * lhs, const conveyorbelt_msgs__msg__ConveyorBeltState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // power
  if (lhs->power != rhs->power) {
    return false;
  }
  // enabled
  if (lhs->enabled != rhs->enabled) {
    return false;
  }
  return true;
}

bool
conveyorbelt_msgs__msg__ConveyorBeltState__copy(
  const conveyorbelt_msgs__msg__ConveyorBeltState * input,
  conveyorbelt_msgs__msg__ConveyorBeltState * output)
{
  if (!input || !output) {
    return false;
  }
  // power
  output->power = input->power;
  // enabled
  output->enabled = input->enabled;
  return true;
}

conveyorbelt_msgs__msg__ConveyorBeltState *
conveyorbelt_msgs__msg__ConveyorBeltState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  conveyorbelt_msgs__msg__ConveyorBeltState * msg = (conveyorbelt_msgs__msg__ConveyorBeltState *)allocator.allocate(sizeof(conveyorbelt_msgs__msg__ConveyorBeltState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(conveyorbelt_msgs__msg__ConveyorBeltState));
  bool success = conveyorbelt_msgs__msg__ConveyorBeltState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
conveyorbelt_msgs__msg__ConveyorBeltState__destroy(conveyorbelt_msgs__msg__ConveyorBeltState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    conveyorbelt_msgs__msg__ConveyorBeltState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__init(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  conveyorbelt_msgs__msg__ConveyorBeltState * data = NULL;

  if (size) {
    data = (conveyorbelt_msgs__msg__ConveyorBeltState *)allocator.zero_allocate(size, sizeof(conveyorbelt_msgs__msg__ConveyorBeltState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = conveyorbelt_msgs__msg__ConveyorBeltState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        conveyorbelt_msgs__msg__ConveyorBeltState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__fini(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      conveyorbelt_msgs__msg__ConveyorBeltState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

conveyorbelt_msgs__msg__ConveyorBeltState__Sequence *
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array = (conveyorbelt_msgs__msg__ConveyorBeltState__Sequence *)allocator.allocate(sizeof(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__destroy(conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__are_equal(const conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * lhs, const conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!conveyorbelt_msgs__msg__ConveyorBeltState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
conveyorbelt_msgs__msg__ConveyorBeltState__Sequence__copy(
  const conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * input,
  conveyorbelt_msgs__msg__ConveyorBeltState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(conveyorbelt_msgs__msg__ConveyorBeltState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    conveyorbelt_msgs__msg__ConveyorBeltState * data =
      (conveyorbelt_msgs__msg__ConveyorBeltState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!conveyorbelt_msgs__msg__ConveyorBeltState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          conveyorbelt_msgs__msg__ConveyorBeltState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!conveyorbelt_msgs__msg__ConveyorBeltState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
