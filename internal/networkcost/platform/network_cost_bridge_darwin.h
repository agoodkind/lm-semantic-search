#ifndef LMS_NETWORK_COST_BRIDGE_DARWIN_H
#define LMS_NETWORK_COST_BRIDGE_DARWIN_H

#include <stdint.h>

typedef struct {
    int32_t available;
    int32_t expensive;
    int32_t constrained;
} lms_network_cost_result;

lms_network_cost_result lms_network_cost_read(int64_t timeout_nanoseconds);

#endif
