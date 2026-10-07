package onnx

/*
#include <stdlib.h>
#include "onnx_bridge.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"

	"goodkind.io/lm-semantic-search/internal/onnxruntimedist"
)

// ErrRuntimeLibraryUnavailable reports that no candidate path opened the ONNX
// Runtime shared library.
var ErrRuntimeLibraryUnavailable = errors.New("ONNX Runtime shared library is unavailable")

// Lock onnxRuntimesMutex before reading or writing onnxLibraryLoaded.
var onnxLibraryLoaded bool

// RuntimeLibrary is the file the process opened and the version that file
// reports.
type RuntimeLibrary struct {
	Path    string
	Version string
}

// LoadRuntimeLibrary opens the library like NewProviderForModel and reports
// the opened file.
func LoadRuntimeLibrary() (RuntimeLibrary, error) {
	onnxRuntimesMutex.Lock()
	defer onnxRuntimesMutex.Unlock()
	if err := loadRuntimeLibraryLocked(); err != nil {
		return RuntimeLibrary{Path: "", Version: ""}, err
	}
	pathBuffer := make([]byte, onnxErrorBufferBytes)
	versionBuffer := make([]byte, onnxErrorBufferBytes)
	if C.lms_onnx_runtime_info(
		(*C.char)(unsafe.Pointer(&pathBuffer[0])),
		(*C.char)(unsafe.Pointer(&versionBuffer[0])),
	) != 0 {
		err := errors.New("read the opened ONNX Runtime library path")
		slog.Error("read ONNX Runtime library info failed", "err", err)
		return RuntimeLibrary{Path: "", Version: ""}, err
	}
	return RuntimeLibrary{Path: cErrorMessage(pathBuffer), Version: cErrorMessage(versionBuffer)}, nil
}

// Only NewProviderForModel and LoadRuntimeLibrary open the library. After a
// failed load, the next call tries every candidate again.
func loadRuntimeLibraryLocked() error {
	if onnxLibraryLoaded {
		return nil
	}
	candidates, err := runtimeLibraryCandidates()
	if err != nil {
		return err
	}
	failures := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		path := C.CString(candidate)
		errorBuffer := make([]byte, onnxErrorBufferBytes)
		failed := C.lms_onnx_load_runtime(path, (*C.char)(unsafe.Pointer(&errorBuffer[0])))
		C.free(unsafe.Pointer(path))
		if failed == 0 {
			onnxLibraryLoaded = true
			return nil
		}
		failures = append(failures, fmt.Sprintf("%s (%s)", candidate, cErrorMessage(errorBuffer)))
	}
	err = fmt.Errorf("%w: tried %s", ErrRuntimeLibraryUnavailable, strings.Join(failures, "; "))
	slog.Error("load ONNX Runtime library failed", "err", err)
	return err
}

// The installers write the library beside the executable. The SONAME
// candidate resolves through the executable's runpath, which points
// development and test binaries at the staged build library.
func runtimeLibraryCandidates() ([]string, error) {
	names, err := onnxruntimedist.LibraryNamesFor(runtime.GOOS)
	if err != nil {
		slog.Error("resolve ONNX Runtime library name failed", "err", err)
		return nil, fmt.Errorf("resolve ONNX Runtime library name: %w", err)
	}
	candidates := make([]string, 0, 2)
	if executable, executableErr := os.Executable(); executableErr == nil {
		if resolved, resolveErr := filepath.EvalSymlinks(executable); resolveErr == nil {
			executable = resolved
		}
		candidates = append(candidates, filepath.Join(filepath.Dir(executable), names.SONAME))
	}
	if runtime.GOOS == "darwin" {
		return append(candidates, "@rpath/"+names.SONAME), nil
	}
	return append(candidates, names.SONAME), nil
}
