#include "onnx_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

struct lms_onnx_session {
    const OrtApi *api;
    OrtEnv *environment;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
};

static int lms_onnx_status_error(
    const OrtApi *api,
    OrtStatus *status,
    char *error_buffer,
    size_t error_buffer_length
) {
    if (status == NULL) {
        return 0;
    }

    if (error_buffer != NULL && error_buffer_length > 0) {
        const char *message = api->GetErrorMessage(status);
        snprintf(error_buffer, error_buffer_length, "%s", message);
    }
    api->ReleaseStatus(status);
    return 1;
}

static int lms_onnx_plain_error(
    const char *message,
    char *error_buffer,
    size_t error_buffer_length
) {
    if (error_buffer != NULL && error_buffer_length > 0) {
        snprintf(error_buffer, error_buffer_length, "%s", message);
    }
    return 1;
}

void lms_onnx_session_free(lms_onnx_session *session) {
    if (session == NULL) {
        return;
    }
    if (session->api != NULL) {
        if (session->memory_info != NULL) {
            session->api->ReleaseMemoryInfo(session->memory_info);
        }
        if (session->session != NULL) {
            session->api->ReleaseSession(session->session);
        }
        if (session->environment != NULL) {
            session->api->ReleaseEnv(session->environment);
        }
    }
    free(session);
}

lms_onnx_session *lms_onnx_session_create(
    const char *model_path,
    char *error_buffer
) {
    if (model_path == NULL || model_path[0] == '\0') {
        lms_onnx_plain_error(
            "invalid ONNX session arguments",
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        );
        return NULL;
    }

    const OrtApiBase *api_base = OrtGetApiBase();
    if (api_base == NULL) {
        lms_onnx_plain_error(
            "OrtGetApiBase returned nil",
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        );
        return NULL;
    }
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    if (api == NULL) {
        lms_onnx_plain_error(
            "ONNX Runtime API version is unavailable",
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        );
        return NULL;
    }

    lms_onnx_session *wrapper = calloc(1, sizeof(*wrapper));
    if (wrapper == NULL) {
        lms_onnx_plain_error(
            "allocate ONNX session",
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        );
        return NULL;
    }
    wrapper->api = api;

    OrtStatus *status = api->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING,
        "lm-semantic-search",
        &wrapper->environment
    );
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        )) {
        lms_onnx_session_free(wrapper);
        return NULL;
    }

    OrtSessionOptions *options = NULL;
    status = api->CreateSessionOptions(&options);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        )) {
        lms_onnx_session_free(wrapper);
        return NULL;
    }

    status = api->SetIntraOpNumThreads(options, 1);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        )) {
        api->ReleaseSessionOptions(options);
        lms_onnx_session_free(wrapper);
        return NULL;
    }

    status = api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        )) {
        api->ReleaseSessionOptions(options);
        lms_onnx_session_free(wrapper);
        return NULL;
    }

    status = api->CreateSession(
        wrapper->environment,
        model_path,
        options,
        &wrapper->session
    );
    api->ReleaseSessionOptions(options);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        )) {
        lms_onnx_session_free(wrapper);
        return NULL;
    }

    status = api->CreateCpuMemoryInfo(
        OrtArenaAllocator,
        OrtMemTypeDefault,
        &wrapper->memory_info
    );
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            LMS_ONNX_ERROR_BUFFER_BYTES
        )) {
        lms_onnx_session_free(wrapper);
        return NULL;
    }

    return wrapper;
}

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
) {
    if (session == NULL || input_ids == NULL || attention_mask == NULL ||
        token_count == 0 || embedding_dimension == 0 || output == NULL ||
        output_count == NULL ||
        (use_token_type_ids != 0 && token_type_ids == NULL)) {
        return lms_onnx_plain_error(
            "invalid ONNX inference arguments",
            error_buffer,
            error_buffer_length
        );
    }

    const OrtApi *api = session->api;
    int64_t shape[] = {1, (int64_t)token_count};
    size_t input_bytes = token_count * sizeof(int64_t);
    OrtValue *input_values[] = {NULL, NULL, NULL};
    OrtValue *result = NULL;
    OrtTensorTypeAndShapeInfo *result_info = NULL;
    OrtStatus *status = NULL;
    int failed = 1;

    status = api->CreateTensorWithDataAsOrtValue(
        session->memory_info,
        (void *)input_ids,
        input_bytes,
        shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_values[0]
    );
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }

    status = api->CreateTensorWithDataAsOrtValue(
        session->memory_info,
        (void *)attention_mask,
        input_bytes,
        shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_values[1]
    );
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }

    if (use_token_type_ids != 0) {
        status = api->CreateTensorWithDataAsOrtValue(
            session->memory_info,
            (void *)token_type_ids,
            input_bytes,
            shape,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_values[2]
        );
        if (lms_onnx_status_error(
                api,
                status,
                error_buffer,
                error_buffer_length
            )) {
            goto cleanup;
        }
    }

    const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    const char *output_names[] = {"last_hidden_state"};
    const OrtValue *const_inputs[] = {
        input_values[0],
        input_values[1],
        input_values[2],
    };
    status = api->Run(
        session->session,
        NULL,
        input_names,
        const_inputs,
        use_token_type_ids != 0 ? 3 : 2,
        output_names,
        1,
        &result
    );
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }

    status = api->GetTensorTypeAndShape(result, &result_info);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }

    size_t dimension_count = 0;
    status = api->GetDimensionsCount(result_info, &dimension_count);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }
    if (dimension_count != 3) {
        lms_onnx_plain_error(
            "ONNX model returned a non-3D embedding tensor",
            error_buffer,
            error_buffer_length
        );
        goto cleanup;
    }

    int64_t dimensions[3] = {0, 0, 0};
    status = api->GetDimensions(result_info, dimensions, 3);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }
    if (dimensions[0] != 1 || dimensions[1] != (int64_t)token_count ||
        dimensions[2] != (int64_t)embedding_dimension) {
        lms_onnx_plain_error(
            "ONNX model returned an unexpected embedding shape",
            error_buffer,
            error_buffer_length
        );
        goto cleanup;
    }

    size_t element_count = 0;
    status = api->GetTensorShapeElementCount(result_info, &element_count);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }
    if (element_count > output_capacity) {
        lms_onnx_plain_error(
            "ONNX output buffer is too small",
            error_buffer,
            error_buffer_length
        );
        goto cleanup;
    }

    float *result_data = NULL;
    status = api->GetTensorMutableData(result, (void **)&result_data);
    if (lms_onnx_status_error(
            api,
            status,
            error_buffer,
            error_buffer_length
        )) {
        goto cleanup;
    }
    memcpy(output, result_data, element_count * sizeof(float));
    *output_count = element_count;
    failed = 0;

cleanup:
    if (result_info != NULL) {
        api->ReleaseTensorTypeAndShapeInfo(result_info);
    }
    if (result != NULL) {
        api->ReleaseValue(result);
    }
    for (size_t i = 0; i < 3; i++) {
        if (input_values[i] != NULL) {
            api->ReleaseValue(input_values[i]);
        }
    }
    return failed;
}
