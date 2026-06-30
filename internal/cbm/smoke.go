//go:build darwin && arm64

package cbm

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/cbm/src -I${SRCDIR}/../../third_party/cbm/internal/cbm -I${SRCDIR}/../../third_party/cbm/internal/cbm/vendored/ts_runtime/include
#cgo darwin LDFLAGS: ${SRCDIR}/../../build/libcbm_engine.a -lc++ -lm -lz
#include <stdlib.h>
#include "cbm.h"
#include "mcp/mcp.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type engineServer struct {
	pointer *C.cbm_mcp_server_t
}

func initializeAllocator() {
	C.cbm_alloc_init()
}

func newEngineServer(projectName string) (*engineServer, error) {
	cProjectName := C.CString(projectName)
	defer C.free(unsafe.Pointer(cProjectName))

	pointer := C.cbm_mcp_server_new(cProjectName)
	if pointer == nil {
		return nil, fmt.Errorf("cbm_mcp_server_new returned nil")
	}

	return &engineServer{pointer: pointer}, nil
}

func (server *engineServer) close() {
	if server == nil {
		return
	}
	if server.pointer == nil {
		return
	}

	C.cbm_mcp_server_free(server.pointer)
	server.pointer = nil
}

func (server *engineServer) callTool(toolName string, argumentsJSON string) (string, error) {
	cToolName := C.CString(toolName)
	defer C.free(unsafe.Pointer(cToolName))

	cArgumentsJSON := C.CString(argumentsJSON)
	defer C.free(unsafe.Pointer(cArgumentsJSON))

	rawResponse := C.cbm_mcp_handle_tool(server.pointer, cToolName, cArgumentsJSON)
	if rawResponse == nil {
		return "", fmt.Errorf("%s returned nil", toolName)
	}
	defer C.free(unsafe.Pointer(rawResponse))

	return C.GoString(rawResponse), nil
}
