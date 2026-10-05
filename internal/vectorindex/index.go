// Package vectorindex is an HNSW cosine index over uint64 keys, built on
// github.com/coder/hnsw.
//
// Keys with identical vectors share one graph node, and groups maps that node to
// every such key. In a graph with one node per key, a set of identical vectors
// larger than the neighbor limit links only to itself, and a search can miss
// every other node.
package vectorindex

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"slices"
	"sort"
	"sync"

	"github.com/coder/hnsw"
)

const (
	neighborsPerNode = 32
	searchExpansion  = 128
	levelFactor      = 0.25
	fileFormat       = "lms-vectorindex-1"
)

// ErrFormat reports malformed data even when the file format is recognized.
var ErrFormat = errors.New("unrecognized vector index file")

type vectorDigest [sha256.Size]byte

// Index serializes every operation with one mutex.
type Index struct {
	mutex      sync.Mutex
	graph      *hnsw.Graph[uint64]
	dimensions int
	// groups maps each graph node key to every key that stores the node's
	// vector, in insertion order.
	groups      map[uint64][]uint64
	nodeOfKey   map[uint64]uint64
	nodeOfValue map[vectorDigest]uint64
}

// New rejects a width that is not positive.
func New(dimensions int) (*Index, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("vector index dimensions must be positive: %d", dimensions)
	}
	return newIndex(newGraph(), dimensions), nil
}

func newIndex(graph *hnsw.Graph[uint64], dimensions int) *Index {
	return &Index{
		mutex:       sync.Mutex{},
		graph:       graph,
		dimensions:  dimensions,
		groups:      make(map[uint64][]uint64),
		nodeOfKey:   make(map[uint64]uint64),
		nodeOfValue: make(map[vectorDigest]uint64),
	}
}

func newGraph() *hnsw.Graph[uint64] {
	graph := hnsw.NewGraph[uint64]()
	graph.M = neighborsPerNode
	graph.EfSearch = searchExpansion
	graph.Ml = levelFactor
	graph.Distance = hnsw.CosineDistance
	return graph
}

func digest(vector []float32) vectorDigest {
	encoded := make([]byte, 4*len(vector))
	for position, component := range vector {
		binary.LittleEndian.PutUint32(encoded[4*position:], math.Float32bits(component))
	}
	return sha256.Sum256(encoded)
}

// Add inserts vector under a key that the index does not contain.
func (index *Index) Add(key uint64, vector []float32) error {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if err := index.validateLocked(vector); err != nil {
		return err
	}
	if _, found := index.nodeOfKey[key]; found {
		return fmt.Errorf("vector index already contains key %d", key)
	}
	vectorKey := digest(vector)
	node, found := index.nodeOfValue[vectorKey]
	if !found {
		node = key
		index.graph.Add(hnsw.MakeNode(key, slices.Clone(vector)))
		index.nodeOfValue[vectorKey] = node
	}
	index.groups[node] = append(index.groups[node], key)
	index.nodeOfKey[key] = node
	return nil
}

// Search returns up to count keys in ascending cosine distance, with their
// distances. Keys with one vector are ordered by ascending key.
func (index *Index) Search(vector []float32, count int) ([]uint64, []float32, error) {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if err := index.validateLocked(vector); err != nil {
		return nil, nil, err
	}
	if count <= 0 || index.graph.Len() == 0 {
		return []uint64{}, []float32{}, nil
	}
	nodes := index.graph.Search(vector, count)
	distances := make(map[uint64]float32, len(nodes))
	for _, node := range nodes {
		distances[node.Key] = hnsw.CosineDistance(vector, node.Value)
	}
	sort.Slice(nodes, func(left int, right int) bool {
		if distances[nodes[left].Key] != distances[nodes[right].Key] {
			return distances[nodes[left].Key] < distances[nodes[right].Key]
		}
		return nodes[left].Key < nodes[right].Key
	})
	keys := make([]uint64, 0, count)
	keyDistances := make([]float32, 0, count)
	for _, node := range nodes {
		group := slices.Clone(index.groups[node.Key])
		slices.Sort(group)
		for _, key := range group {
			if len(keys) == count {
				return keys, keyDistances, nil
			}
			keys = append(keys, key)
			keyDistances = append(keyDistances, distances[node.Key])
		}
	}
	return keys, keyDistances, nil
}

// Contains includes keys that share another key's graph node.
func (index *Index) Contains(key uint64) bool {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	_, found := index.nodeOfKey[key]
	return found
}

// Size counts keys, including keys that share a vector.
func (index *Index) Size() int {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	return len(index.nodeOfKey)
}

// Dimensions returns the configured width, also after Close.
func (index *Index) Dimensions() int {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	return index.dimensions
}

// Save writes the vector width to the file header.
// An empty graph does not store its vector width.
func (index *Index) Save(path string) error {
	index.mutex.Lock()
	defer index.mutex.Unlock()
	if index.graph == nil {
		return errors.New("save vector index: index is closed")
	}
	file, err := os.Create(path)
	if err != nil {
		slog.Error("create vector index file failed", "path", path, "err", err)
		return fmt.Errorf("create vector index file %s: %w", path, err)
	}
	writer := bufio.NewWriter(file)
	if err := index.writeLocked(writer); err != nil {
		_ = file.Close()
		slog.Error("write vector index file failed", "path", path, "err", err)
		return fmt.Errorf("write vector index file %s: %w", path, err)
	}
	if err := writer.Flush(); err != nil {
		_ = file.Close()
		slog.Error("flush vector index file failed", "path", path, "err", err)
		return fmt.Errorf("flush vector index file %s: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		slog.Error("sync vector index file failed", "path", path, "err", err)
		return fmt.Errorf("sync vector index file %s: %w", path, err)
	}
	if err := file.Close(); err != nil {
		slog.Error("close vector index file failed", "path", path, "err", err)
		return fmt.Errorf("close vector index file %s: %w", path, err)
	}
	return nil
}

func codecError(operation string, err error) error {
	slog.Error("vector index "+operation+" failed", "err", err)
	return fmt.Errorf("vector index %s: %w", operation, err)
}

func (index *Index) writeLocked(writer io.Writer) error {
	if _, err := io.WriteString(writer, fileFormat); err != nil {
		return codecError("write format", err)
	}
	if err := binary.Write(writer, binary.LittleEndian, int64(index.dimensions)); err != nil {
		return codecError("write dimensions", err)
	}
	if err := index.graph.Export(writer); err != nil {
		return codecError("write graph", err)
	}
	nodes := make([]uint64, 0, len(index.groups))
	for node := range index.groups {
		nodes = append(nodes, node)
	}
	slices.Sort(nodes)
	if err := binary.Write(writer, binary.LittleEndian, int64(len(nodes))); err != nil {
		return codecError("write group count", err)
	}
	for _, node := range nodes {
		group := index.groups[node]
		header := []uint64{node, uint64(len(group))}
		if err := binary.Write(writer, binary.LittleEndian, header); err != nil {
			return codecError("write group header", err)
		}
		if err := binary.Write(writer, binary.LittleEndian, group); err != nil {
			return codecError("write group keys", err)
		}
	}
	return nil
}

// Load returns ErrFormat for an unrecognized file format.
func Load(path string) (*Index, error) {
	file, err := os.Open(path)
	if err != nil {
		slog.Error("open vector index file failed", "path", path, "err", err)
		return nil, fmt.Errorf("open vector index file %s: %w", path, err)
	}
	defer file.Close()
	index, err := read(bufio.NewReader(file))
	if err != nil {
		return nil, fmt.Errorf("read vector index %s: %w: %w", path, ErrFormat, err)
	}
	return index, nil
}

func read(reader *bufio.Reader) (*Index, error) {
	format := make([]byte, len(fileFormat))
	if _, err := io.ReadFull(reader, format); err != nil {
		return nil, codecError("read format", err)
	}
	if string(format) != fileFormat {
		return nil, fmt.Errorf("format %q", format)
	}
	var dimensions int64
	if err := binary.Read(reader, binary.LittleEndian, &dimensions); err != nil {
		return nil, codecError("read dimensions", err)
	}
	if dimensions <= 0 || dimensions > math.MaxInt32 {
		return nil, fmt.Errorf("dimensions %d out of range", dimensions)
	}
	width := int(dimensions)
	graph := newGraph()
	if err := graph.Import(reader); err != nil {
		return nil, codecError("read graph", err)
	}
	if graph.Len() > 0 && graph.Dims() != width {
		return nil, fmt.Errorf("header width %d, vector width %d", width, graph.Dims())
	}
	index := newIndex(graph, width)
	var nodeCount int64
	if err := binary.Read(reader, binary.LittleEndian, &nodeCount); err != nil {
		return nil, codecError("read group count", err)
	}
	if nodeCount != int64(graph.Len()) {
		return nil, fmt.Errorf("%d key groups for %d graph nodes", nodeCount, graph.Len())
	}
	for range nodeCount {
		header := make([]uint64, 2)
		if err := binary.Read(reader, binary.LittleEndian, header); err != nil {
			return nil, codecError("read group header", err)
		}
		node := header[0]
		vector, found := graph.Lookup(node)
		if !found {
			return nil, fmt.Errorf("key group %d without a graph node", node)
		}
		// Read group keys individually instead of allocating from the untrusted length.
		group := make([]uint64, 0)
		for range header[1] {
			var key uint64
			if err := binary.Read(reader, binary.LittleEndian, &key); err != nil {
				return nil, codecError("read group keys", err)
			}
			group = append(group, key)
		}
		index.groups[node] = group
		index.nodeOfValue[digest(vector)] = node
		for _, key := range group {
			index.nodeOfKey[key] = node
		}
	}
	return index, nil
}

// Close does not delete files written by Save.
func (index *Index) Close() {
	if index == nil {
		return
	}
	index.mutex.Lock()
	defer index.mutex.Unlock()
	index.graph = nil
	index.groups = map[uint64][]uint64{}
	index.nodeOfKey = map[uint64]uint64{}
	index.nodeOfValue = map[vectorDigest]uint64{}
}

func (index *Index) validateLocked(vector []float32) error {
	if index.graph == nil {
		return errors.New("vector index is closed")
	}
	if len(vector) != index.dimensions {
		return fmt.Errorf("vector has %d dimensions, want %d", len(vector), index.dimensions)
	}
	return nil
}
