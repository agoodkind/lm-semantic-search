#ifndef LMS_ONNX_BRIDGE_H
#define LMS_ONNX_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

typedef struct lms_onnx_session lms_onnx_session;

#define LMS_ONNX_ERROR_BUFFER_BYTES 2048

lms_onnx_session *lms_onnx_session_create(
    const char *model_path,
    char *error_buffer
);

void lms_onnx_session_free(lms_onnx_session *session);

int lms_onnx_run(
    lms_onnx_session *session,
    const int64_t *input_ids,
    const int64_t *attention_mask,
    const int64_t *token_type_ids,
    int use_token_type_ids,
    size_t token_count,
    size_t embedding_dimension,
    float *output,
    size_t output_capacity,
    size_t *output_count,
    char *error_buffer,
    size_t error_buffer_length
);

#endif
