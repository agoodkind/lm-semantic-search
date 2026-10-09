#import "network_cost_bridge_darwin.h"

#import <Network/Network.h>
#import <dispatch/dispatch.h>

lms_network_cost_result lms_network_cost_read(int64_t timeout_nanoseconds) {
    __block lms_network_cost_result observed = {0, 0, 0};
    __block int32_t delivered = 0;

    dispatch_queue_t queue = dispatch_queue_create(
        "io.goodkind.lm-semantic-search.networkcost",
        DISPATCH_QUEUE_SERIAL
    );
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    nw_path_monitor_t monitor = nw_path_monitor_create();

    nw_path_monitor_set_queue(monitor, queue);
    nw_path_monitor_set_update_handler(monitor, ^(nw_path_t path) {
        if (delivered != 0) {
            return;
        }
        delivered = 1;
        nw_path_status_t status = nw_path_get_status(path);
        if (status == nw_path_status_satisfied || status == nw_path_status_satisfiable) {
            observed.available = 1;
        }
        if (nw_path_is_expensive(path)) {
            observed.expensive = 1;
        }
        if (nw_path_is_constrained(path)) {
            observed.constrained = 1;
        }
        dispatch_semaphore_signal(finished);
    });
    nw_path_monitor_start(monitor);

    intptr_t timed_out = dispatch_semaphore_wait(
        finished,
        dispatch_time(DISPATCH_TIME_NOW, timeout_nanoseconds)
    );
    nw_path_monitor_cancel(monitor);

    lms_network_cost_result result = {0, 0, 0};
    if (timed_out == 0) {
        result = observed;
    }

    nw_release(monitor);
    dispatch_release(finished);
    dispatch_release(queue);
    return result;
}
