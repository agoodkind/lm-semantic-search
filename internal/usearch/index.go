// Package usearch provides a small cgo binding to the vendored usearch C API.
package usearch

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/usearch/c
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/../../third_party/usearch/c -I${SRCDIR}/../../third_party/usearch/include -I${SRCDIR}/../../third_party/usearch/numkong/include -I${SRCDIR}/../../third_party/usearch/stringzilla/include
#cgo darwin LDFLAGS: -lc++
#cgo linux LDFLAGS: -lstdc++ -lm
#include <stdlib.h>
#include "usearch.h"

enum usearch_path_operation {
    usearch_path_save = 1,
    usearch_path_load = 2,
    usearch_path_view = 3,
};

typedef struct {
    usearch_index_t index;
    usearch_error_t error;
} usearch_index_result_t;

typedef struct {
    size_t count;
    usearch_error_t error;
} usearch_count_result_t;

typedef struct {
    bool found;
    usearch_error_t error;
} usearch_bool_result_t;

static usearch_index_result_t usearch_init_cosine_f32(
    size_t dimensions,
    size_t connectivity,
    size_t expansion_add,
    size_t expansion_search
) {
    usearch_index_result_t result = {};
    usearch_init_options_t options = {};
    options.metric_kind = usearch_metric_cos_k;
    options.quantization = usearch_scalar_f32_k;
    options.dimensions = dimensions;
    options.connectivity = connectivity;
    options.expansion_add = expansion_add;
    options.expansion_search = expansion_search;
    result.index = usearch_init(&options, &result.error);
    return result;
}

static usearch_error_t usearch_add_f32(
    usearch_index_t index,
    usearch_key_t key,
    void const* vector
) {
    usearch_error_t error = NULL;
    usearch_add(index, key, vector, usearch_scalar_f32_k, &error);
    return error;
}

static usearch_count_result_t usearch_search_f32(
    usearch_index_t index,
    float const* query,
    size_t count,
    void* keys,
    void* distances
) {
    usearch_count_result_t result = {};
    result.count = usearch_search(
        index,
        query,
        usearch_scalar_f32_k,
        count,
        (usearch_key_t*)keys,
        (usearch_distance_t*)distances,
        &result.error
    );
    return result;
}

static usearch_count_result_t usearch_remove_key(
    usearch_index_t index,
    usearch_key_t key
) {
    usearch_count_result_t result = {};
    result.count = usearch_remove(index, key, &result.error);
    return result;
}

static usearch_bool_result_t usearch_contains_key(
    usearch_index_t index,
    usearch_key_t key
) {
    usearch_bool_result_t result = {};
    result.found = usearch_contains(index, key, &result.error);
    return result;
}

static usearch_count_result_t usearch_index_size(usearch_index_t index) {
    usearch_count_result_t result = {};
    result.count = usearch_size(index, &result.error);
    return result;
}

static usearch_error_t usearch_reserve_capacity(
    usearch_index_t index,
    size_t capacity
) {
    usearch_error_t error = NULL;
    usearch_reserve(index, capacity, &error);
    return error;
}

static usearch_error_t usearch_free_index(usearch_index_t index) {
    usearch_error_t error = NULL;
    usearch_free(index, &error);
    return error;
}

static usearch_error_t usearch_path_call(
    usearch_index_t index,
    char const* path,
    int operation
) {
    usearch_error_t error = NULL;
    if (operation == usearch_path_save) {
        usearch_save(index, path, &error);
        return error;
    }
    if (operation == usearch_path_load) {
        usearch_load(index, path, &error);
        return error;
    }
    usearch_view(index, path, &error);
    return error;
}

static usearch_count_result_t usearch_index_dimensions(
    usearch_index_t index
) {
    usearch_count_result_t result = {};
    result.count = usearch_dimensions(index, &result.error);
    return result;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"runtime"
	"sync"
	"unsafe"
)

var errClosed = errors.New("usearch index is closed")

const (
	hnswConnectivity    = 32
	hnswExpansionAdd    = 128
	hnswExpansionSearch = 128
)

// Index owns one usearch dense vector index.
type Index struct {
	handle     C.usearch_index_t
	dimensions int
	mutex      sync.Mutex
}

// New constructs a cosine, float32 usearch index.
func New(dimensions int) (*Index, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("usearch dimensions must be positive: %d", dimensions)
	}
	result := C.usearch_init_cosine_f32(
		C.size_t(dimensions),
		C.size_t(hnswConnectivity),
		C.size_t(hnswExpansionAdd),
		C.size_t(hnswExpansionSearch),
	)
	if err := convertError(result.error); err != nil {
		slog.Error("initialize usearch index failed", "dimensions", dimensions, "err", err)
		return nil, fmt.Errorf("initialize usearch index: %w", err)
	}
	handle := result.index
	if handle == nil {
		return nil, errors.New("initialize usearch index: returned nil handle")
	}
	index := &Index{
		handle:     handle,
		dimensions: dimensions,
		mutex:      sync.Mutex{},
	}
	runtime.SetFinalizer(index, (*Index).Close)
	return index, nil
}

// Add inserts vector under key.
func (index *Index) Add(key uint64, vector []float32) error {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if err := index.validateVectorLocked(vector); err != nil {
		return err
	}
	cError := C.usearch_add_f32(
		index.handle,
		C.usearch_key_t(key),
		unsafe.Pointer(&vector[0]),
	)
	if err := convertError(cError); err != nil {
		slog.Error("add usearch vector failed", "key", key, "err", err)
		return fmt.Errorf("add usearch key %d: %w", key, err)
	}
	return nil
}

// Search returns up to count nearest keys and cosine distances.
func (index *Index) Search(
	vector []float32,
	count int,
) ([]uint64, []float32, error) {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if err := index.validateVectorLocked(vector); err != nil {
		return nil, nil, err
	}
	if count <= 0 {
		return []uint64{}, []float32{}, nil
	}
	keys := make([]uint64, count)
	distances := make([]float32, count)
	result := C.usearch_search_f32(
		index.handle,
		(*C.float)(unsafe.Pointer(&vector[0])),
		C.size_t(count),
		unsafe.Pointer(&keys[0]),
		unsafe.Pointer(&distances[0]),
	)
	if err := convertError(result.error); err != nil {
		slog.Error("search usearch index failed", "count", count, "err", err)
		return nil, nil, fmt.Errorf("search usearch index: %w", err)
	}
	resultCount := int(result.count)
	return keys[:resultCount], distances[:resultCount], nil
}

// Remove deletes the vector under key.
func (index *Index) Remove(key uint64) (bool, error) {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return false, errClosed
	}
	result := C.usearch_remove_key(index.handle, C.usearch_key_t(key))
	if err := convertError(result.error); err != nil {
		slog.Error("remove usearch vector failed", "key", key, "err", err)
		return false, fmt.Errorf("remove usearch key %d: %w", key, err)
	}
	return result.count > 0, nil
}

// Contains reports whether key is present.
func (index *Index) Contains(key uint64) (bool, error) {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return false, errClosed
	}
	result := C.usearch_contains_key(index.handle, C.usearch_key_t(key))
	if err := convertError(result.error); err != nil {
		slog.Error("check usearch key failed", "key", key, "err", err)
		return false, fmt.Errorf("check usearch key %d: %w", key, err)
	}
	return bool(result.found), nil
}

// Size returns the number of indexed vectors.
func (index *Index) Size() (int, error) {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return 0, errClosed
	}
	result := C.usearch_index_size(index.handle)
	if err := convertError(result.error); err != nil {
		slog.Error("read usearch size failed", "err", err)
		return 0, fmt.Errorf("read usearch size: %w", err)
	}
	return int(result.count), nil
}

// Dimensions returns the configured vector width.
func (index *Index) Dimensions() (int, error) {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return 0, errClosed
	}
	return index.dimensions, nil
}

// Reserve grows the index capacity to hold capacity vectors.
func (index *Index) Reserve(capacity int) error {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return errClosed
	}
	if capacity < 0 {
		return fmt.Errorf("usearch capacity must not be negative: %d", capacity)
	}
	cError := C.usearch_reserve_capacity(index.handle, C.size_t(capacity))
	if err := convertError(cError); err != nil {
		slog.Error("reserve usearch capacity failed", "capacity", capacity, "err", err)
		return fmt.Errorf("reserve usearch capacity %d: %w", capacity, err)
	}
	return nil
}

// Save writes the index to path.
func (index *Index) Save(path string) error {
	return index.withPath(path, "save", C.usearch_path_save)
}

// Load copies an index from path into memory.
func (index *Index) Load(path string) error {
	return index.withPath(path, "load", C.usearch_path_load)
}

// View maps an index file at path without copying it into memory.
func (index *Index) View(path string) error {
	return index.withPath(path, "view", C.usearch_path_view)
}

// Close frees the native index. Close is safe to call more than once.
func (index *Index) Close() {
	if index == nil {
		return
	}
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return
	}
	C.usearch_free_index(index.handle)
	index.handle = nil
	runtime.SetFinalizer(index, nil)
}

func (index *Index) validateVectorLocked(vector []float32) error {
	if index.handle == nil {
		return errClosed
	}
	if len(vector) != index.dimensions {
		return fmt.Errorf(
			"usearch vector has %d dimensions, want %d",
			len(vector),
			index.dimensions,
		)
	}
	return nil
}

func (index *Index) withPath(
	path string,
	operation string,
	pathOperation C.int,
) error {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.handle == nil {
		return errClosed
	}
	if path == "" {
		return fmt.Errorf("%s usearch index: path is required", operation)
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cError := C.usearch_path_call(index.handle, cPath, pathOperation)
	if err := convertError(cError); err != nil {
		slog.Error(
			"access usearch index path failed",
			"operation",
			operation,
			"path",
			path,
			"err",
			err,
		)
		return fmt.Errorf("%s usearch index %s: %w", operation, path, err)
	}
	if pathOperation != C.usearch_path_save {
		result := C.usearch_index_dimensions(index.handle)
		if err := convertError(result.error); err != nil {
			slog.Error(
				"read usearch dimensions failed",
				"operation",
				operation,
				"path",
				path,
				"err",
				err,
			)
			return fmt.Errorf("read %s usearch dimensions: %w", operation, err)
		}
		index.dimensions = int(result.count)
	}
	return nil
}

func convertError(cError C.usearch_error_t) error {
	if cError == nil {
		return nil
	}
	return errors.New(C.GoString(cError))
}
