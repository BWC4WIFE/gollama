package vramestimator

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/StackExchange/wmi"
	"github.com/mattn/go-runewidth"
	"github.com/sammcj/gollama/v2/logging"
	"github.com/sammcj/gollama/v2/styles"
	"github.com/sammcj/gollama/v2/utils"
	"golang.org/x/sys/windows"
)

// KVCacheQuantisation represents the quantisation type for the k/v context cache
type KVCacheQuantisation string

// ModelConfig represents the configuration of a model
type ModelConfig struct {
	NumParams             float64 `json:"num_params"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	HiddenSize            int     `json:"hidden_size"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	IntermediateSize      int     `json:"intermediate_size"`
	VocabSize             int     `json:"vocab_size"`
}

// GPUInfo contains GPU information from Windows WMI
type GPUInfo struct {
	Name      string
	Vendor    string
	VRAMGB    float64
	VRAMBytes uint64
}

// WMI VideoController struct for WMI queries
type win32VideoController struct {
	Name       string
	AdapterRAM uint64
}

// BPWValues represents the bits per weight values for different components
type BPWValues struct {
	BPW        float64
	LMHeadBPW  float64
	KVCacheBPW float64
}

// QuantResult represents VRAM estimation results for a specific quantisation
type QuantResult struct {
	QuantType string
	BPW       float64
	Contexts  map[int]ContextVRAM
}

// ContextVRAM contains VRAM usage for different KV cache quantisations
type ContextVRAM struct {
	VRAM     float64
	VRAMQ8_0 float64
	VRAMQ4_0 float64
}

// QuantResultTable represents a table of VRAM estimation results
type QuantResultTable struct {
	ModelID  string
	Results  []QuantResult
	FitsVRAM float64
}

const (
	KVCacheFP16 KVCacheQuantisation = "fp16"
	KVCacheQ8_0 KVCacheQuantisation = "q8_0"
	KVCacheQ4_0 KVCacheQuantisation = "q4_0"
)

const (
	CUDASize = 500 * 1024 * 1024 // 500 MB
)

// GGUFMapping maps GGUF quantisation types to their corresponding bits per weight
var GGUFMapping = map[string]float64{
	"F16":     16,
	"Q8_0":    8.5,
	"Q6_K":    6.59,
	"Q5_K_L":  5.75,
	"Q5_K_M":  5.69,
	"Q5_K_S":  5.54,
	"Q5_0":    5.54,
	"Q4_K_L":  4.9,
	"Q4_K_M":  4.85,
	"Q4_K_S":  4.58,
	"Q4_0":    4.55,
	"IQ4_NL":  4.5,
	"Q3_K_L":  4.27,
	"IQ4_XS":  4.25,
	"Q3_K_M":  3.91,
	"IQ3_M":   3.7,
	"IQ3_S":   3.5,
	"Q3_K_S":  3.5,
	"Q2_K":    3.35,
	"IQ3_XS":  3.3,
	"IQ3_XXS": 3.06,
	"IQ2_M":   2.7,
	"IQ2_S":   2.5,
	"IQ2_XS":  2.31,
	"IQ2_XXS": 2.06,
	"IQ1_S":   1.56,
	"Q2":      3.35, // Alias for Q2_K
	"Q3":      3.5,  // Alias for Q3_K_S
	"Q4":      4.55, // Alias for Q4_0
	"Q5":      5.54, // Alias for Q5_0
	"Q6":      6.59, // Alias for Q6_K
	"Q8":      8.5,  // Alias for Q8_0
	"FP16":    16,   // Alias for F16
}

// EXL2Options contains the EXL2 quantisation options
var EXL2Options []float64

var (
	modelConfigCache = make(map[string]ModelConfig)
	cacheMutex       sync.RWMutex
	gpuInfoCache     *GPUInfo
	gpuCacheMutex    sync.RWMutex
	gpuCacheTime     time.Time
	gpuCacheTTL      = 30 * time.Second // Cache GPU info for 30 seconds
)

// VRAM Estimation description
const vramDescription = `
VRAM Estimation Format:
For context sizes ≥ 16K: F16(Q8_0,Q4_0)
- F16: Base model with FP16 KV cache
- Q8_0: Model with Q8_0 KV cache quantisation
- Q4_0: Model with Q4_0 KV cache quantisation

For context sizes < 16K: Single F16 value shown
`

func init() {
	// Initialize EXL2 options from 6.0 down to 2.0 in 0.05 steps
	for i := 6.0; i >= 2.0; i -= 0.05 {
		EXL2Options = append(EXL2Options, math.Round(i*100)/100)
	}
}

// GetWindowsVRAM returns the VRAM of the primary GPU on Windows using WMI
func GetWindowsVRAM() (float64, error) {
	gpuCacheMutex.RLock()
	if gpuInfoCache != nil && time.Since(gpuCacheTime) < gpuCacheTTL {
		vram := gpuInfoCache.VRAMGB
		gpuCacheMutex.RUnlock()
		return vram, nil
	}
	gpuCacheMutex.RUnlock()

	var videoControllers []win32VideoController
	query := "SELECT Name, AdapterRAM FROM Win32_VideoController WHERE Status = 'OK'"
	
	err := wmi.QueryNamespace(query, &videoControllers, "root\\CIMV2")
	if err != nil {
		return 0, fmt.Errorf("WMI query failed: %w", err)
	}
	
	if len(videoControllers) == 0 {
		return 0, fmt.Errorf("no active video controllers found")
	}
	
	// Find the primary GPU (usually first one with VRAM > 0)
	var primaryGPU *win32VideoController
	for i, vc := range videoControllers {
		// Skip virtual adapters
		if isVirtualAdapter(vc.Name) {
			continue
		}
		
		if vc.AdapterRAM > 0 {
			primaryGPU = &videoControllers[i]
			break
		}
	}
	
	if primaryGPU == nil {
		return 0, fmt.Errorf("no physical GPU with VRAM found")
	}
	
	vramGB := float64(primaryGPU.AdapterRAM) / (1024 * 1024 * 1024)
	
	// Cache the result
	gpuCacheMutex.Lock()
	gpuInfoCache = &GPUInfo{
		Name:      primaryGPU.Name,
		VRAMGB:    vramGB,
		VRAMBytes: primaryGPU.AdapterRAM,
	}
	gpuInfoCache.Vendor = detectVendorFromName(primaryGPU.Name)
	gpuCacheTime = time.Now()
	gpuCacheMutex.Unlock()
	
	logging.InfoLogger.Printf("Detected GPU: %s (VRAM: %.2f GB)", primaryGPU.Name, vramGB)
	
	return vramGB, nil
}

// isVirtualAdapter checks if a GPU name indicates a virtual adapter
func isVirtualAdapter(name string) bool {
	nameLower := strings.ToLower(name)
	virtualIndicators := []string{
		"virtual",
		"vmware",
		"virtualbox",
		"qxl",
		"vbox",
		"hyper-v",
		"rdp",
		"remote desktop",
		"microsoft basic display",
		"microsoft remote display",
	}
	
	for _, indicator := range virtualIndicators {
		if strings.Contains(nameLower, indicator) {
			return true
		}
	}
	
	return false
}

// detectVendorFromName determines GPU vendor from device name
func detectVendorFromName(name string) string {
	nameLower := strings.ToLower(name)
	
	switch {
	case strings.Contains(nameLower, "nvidia"):
		return "NVIDIA"
	case strings.Contains(nameLower, "geforce"):
		return "NVIDIA"
	case strings.Contains(nameLower, "quadro"):
		return "NVIDIA"
	case strings.Contains(nameLower, "tesla"):
		return "NVIDIA"
	case strings.Contains(nameLower, "rtx"):
		return "NVIDIA"
	case strings.Contains(nameLower, "gtx"):
		return "NVIDIA"
	case strings.Contains(nameLower, "radeon"):
		return "AMD"
	case strings.Contains(nameLower, "amd"):
		return "AMD"
	case strings.Contains(nameLower, "intel"):
		return "Intel"
	case strings.Contains(nameLower, "iris"):
		return "Intel"
	case strings.Contains(nameLower, "uhd graphics"):
		return "Intel"
	case strings.Contains(nameLower, "hd graphics"):
		return "Intel"
	default:
		return "Unknown"
	}
}

// GetSystemRAM returns the total system RAM in GB on Windows
func GetSystemRAM() (float64, error) {
	var mod = windows.NewLazySystemDLL("kernel32.dll")
	var getMem = mod.NewProc("GetPhysicallyInstalledSystemMemory")
	
	var memKB uint64
	ret, _, err := getMem.Call(uintptr(unsafe.Pointer(&memKB)))
	
	if ret == 0 {
		// Fallback to GlobalMemoryStatusEx
		var memStatus struct {
			Length               uint32
			MemoryLoad           uint32
			TotalPhys            uint64
			AvailPhys            uint64
			TotalPageFile        uint64
			AvailPageFile        uint64
			TotalVirtual         uint64
			AvailVirtual         uint64
			AvailExtendedVirtual uint64
		}
		memStatus.Length = uint32(unsafe.Sizeof(memStatus))
		
		proc := mod.NewProc("GlobalMemoryStatusEx")
		ret, _, err := proc.Call(uintptr(unsafe.Pointer(&memStatus)))
		
		if ret == 0 {
			return 0, fmt.Errorf("failed to get system memory: %v", err)
		}
		
		totalRAM := float64(memStatus.TotalPhys) / (1024 * 1024 * 1024)
		return totalRAM, nil
	}
	
	// Convert KB to GB
	totalRAM := float64(memKB) / 1024 / 1024
	return totalRAM, nil
}

// GetAvailableMemory returns available memory for model loading (prioritizes VRAM)
func GetAvailableMemory() (float64, error) {
	// First try to get GPU VRAM
	vram, err := GetWindowsVRAM()
	if err == nil && vram > 0 {
		logging.InfoLogger.Printf("Using GPU VRAM: %.2f GB", vram)
		
		// For NVIDIA GPUs, we can use more accurate estimation
		gpuCacheMutex.RLock()
		isNVIDIA := gpuInfoCache != nil && gpuInfoCache.Vendor == "NVIDIA"
		gpuCacheMutex.RUnlock()
		
		if isNVIDIA {
			// Reserve 500MB for CUDA/system overhead
			availableVRAM := vram - 0.5
			if availableVRAM < 0.5 {
				availableVRAM = 0.5 // Minimum for very small GPUs
			}
			return availableVRAM, nil
		}
		
		// For non-NVIDIA GPUs, be more conservative
		return vram * 0.8, nil // Use 80% of VRAM
	}
	
	// Fall back to system RAM
	logging.WarnLogger.Printf("GPU VRAM detection failed: %v, falling back to system RAM", err)
	
	ram, err := GetSystemRAM()
	if err != nil {
		return 0, fmt.Errorf("failed to get system RAM: %v", err)
	}
	
	logging.InfoLogger.Printf("Using system RAM: %.2f GB", ram)
	
	// Reserve 4GB for Windows and applications
	availableRAM := ram - 4.0
	if availableRAM < 4.0 {
		availableRAM = 4.0 // Minimum for very small systems
	}
	
	return availableRAM, nil
}

type OllamaModelInfo struct {
	Details struct {
		ParameterSize     string   `json:"parameter_size"`
		QuantizationLevel string   `json:"quantisation_level"`
		Family            string   `json:"family"`
		Families          []string `json:"families"`
	} `json:"details"`
	ModelInfo map[string]interface{} `json:"model_info"`
}

func extractModelInfo(info map[string]interface{}, key string) (float64, bool) {
	for k, v := range info {
		if strings.HasSuffix(k, key) {
			switch val := v.(type) {
			case float64:
				return val, true
			case int64:
				return float64(val), true
			case int:
				return float64(val), true
			}
		}
	}
	return 0, false
}

func FetchOllamaModelInfo(apiURL, modelName string) (*OllamaModelInfo, error) {
	url := fmt.Sprintf("%s/api/show", apiURL)
	payload := []byte(fmt.Sprintf(`{"name": "%s"}`, modelName))

	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return nil, fmt.Errorf("error making request to Ollama API: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama API returned status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading Ollama API response: %v", err)
	}

	logging.DebugLogger.Printf("Raw Ollama API response: %s", string(body))

	var modelInfo OllamaModelInfo
	if err := json.Unmarshal(body, &modelInfo); err != nil {
		return nil, fmt.Errorf("error decoding Ollama API response: %v", err)
	}

	return &modelInfo, nil
}

// EstimateVRAM is the main entry point for VRAM estimation
func EstimateVRAM(modelIdentifier, apiURL string, fitsVRAM float64) error {
	var ollamaModelInfo *OllamaModelInfo
	var err error

	// Check if the modelIdentifier is an Ollama model name
	if strings.Contains(modelIdentifier, ":") {
		ollamaModelInfo, err = FetchOllamaModelInfo(apiURL, modelIdentifier)
		if err != nil {
			return fmt.Errorf("error fetching Ollama model info: %v", err)
		}
	}

	// Generate the quantisation table
	table, err := GenerateQuantTable(modelIdentifier, fitsVRAM, ollamaModelInfo, 65536)
	if err != nil {
		return fmt.Errorf("error generating quantisation table: %v", err)
	}

	// Print the formatted table
	fmt.Println(PrintFormattedTable(table))

	return nil
}

// CalculateVRAMRaw calculates the raw VRAM usage
func CalculateVRAMRaw(config ModelConfig, bpwValues BPWValues, context int, numGPUs int, gqa bool) float64 {
	logging.DebugLogger.Println("Calculating VRAM usage...")

	cudaSize := float64(CUDASize * numGPUs)
	paramsSize := config.NumParams * 1e9 * (bpwValues.BPW / 8)

	kvCacheSize := float64(context*2*config.NumHiddenLayers*config.HiddenSize) * (bpwValues.KVCacheBPW / 8)
	if gqa {
		kvCacheSize *= float64(config.NumKeyValueHeads) / float64(config.NumAttentionHeads)
	}

	bytesPerParam := bpwValues.BPW / 8
	lmHeadBytesPerParam := bpwValues.LMHeadBPW / 8

	headDim := float64(config.HiddenSize) / float64(config.NumAttentionHeads)
	attentionInput := bytesPerParam * float64(context*config.HiddenSize)

	q := bytesPerParam * float64(context) * headDim * float64(config.NumAttentionHeads)
	k := bytesPerParam * float64(context) * headDim * float64(config.NumKeyValueHeads)
	v := bytesPerParam * float64(context) * headDim * float64(config.NumKeyValueHeads)

	softmaxOutput := lmHeadBytesPerParam * float64(config.NumAttentionHeads*context)
	softmaxDropoutMask := float64(config.NumAttentionHeads * context)
	dropoutOutput := lmHeadBytesPerParam * float64(config.NumAttentionHeads*context)

	outProjInput := lmHeadBytesPerParam * float64(context*config.NumAttentionHeads) * headDim
	attentionDropout := float64(context * config.HiddenSize)

	attentionBlock := attentionInput + q + k + softmaxOutput + v + outProjInput + softmaxDropoutMask + dropoutOutput + attentionDropout

	mlpInput := bytesPerParam * float64(context*config.HiddenSize)
	activationInput := bytesPerParam * float64(context*config.IntermediateSize)
	downProjInput := bytesPerParam * float64(context*config.IntermediateSize)
	dropoutMask := float64(context * config.HiddenSize)
	mlpBlock := mlpInput + activationInput + downProjInput + dropoutMask

	layerNorms := bytesPerParam * float64(context*config.HiddenSize*2)
	activationsSize := attentionBlock + mlpBlock + layerNorms

	outputSize := lmHeadBytesPerParam * float64(context*config.VocabSize)

	vramBits := cudaSize + paramsSize + activationsSize + outputSize + kvCacheSize

	return bitsToGB(vramBits)
}

// bitsToGB converts bits to gigabytes
func bitsToGB(bits float64) float64 {
	return bits / math.Pow(2, 30)
}

// DownloadFile downloads a file from a URL and saves it to the specified path
func DownloadFile(url, filePath string, headers map[string]string) error {
	if _, err := os.Stat(filePath); err == nil {
		logging.InfoLogger.Println("File already exists, skipping download")
		return nil
	}

	logging.DebugLogger.Println("Downloading file from:", url)

	client := &http.Client{
		Timeout: 120 * time.Second,
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	for key, value := range headers {
		req.Header.Set(key, value)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("bad status %d: %s", resp.StatusCode, string(body))
	}

	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	out, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// GetHuggingFaceToken retrieves HuggingFace token from environment or file
func GetHuggingFaceToken() string {
	accessToken := os.Getenv("HUGGINGFACE_TOKEN")
	if accessToken == "" {
		accessToken = os.Getenv("HF_TOKEN")
	}
	if accessToken == "" {
		// Try to read from Windows credential manager or file
		tokenPath := filepath.Join(utils.GetHomeDir(), ".huggingface", "token")
		if _, err := os.Stat(tokenPath); err == nil {
			token, err := os.ReadFile(tokenPath)
			if err == nil {
				accessToken = strings.TrimSpace(string(token))
			}
		}
	}
	return accessToken
}

// GetModelConfig retrieves and parses the model configuration
func GetModelConfig(modelID string) (ModelConfig, error) {
	cacheMutex.RLock()
	if config, ok := modelConfigCache[modelID]; ok {
		cacheMutex.RUnlock()
		return config, nil
	}
	cacheMutex.RUnlock()

	// Windows-specific cache path
	baseDir := filepath.Join(os.Getenv("LOCALAPPDATA"), "gollama", "cache", modelID)
	if baseDir == "" {
		baseDir = filepath.Join(utils.GetHomeDir(), ".cache", "gollama", modelID)
	}
	
	configPath := filepath.Join(baseDir, "config.json")
	indexPath := filepath.Join(baseDir, "model.safetensors.index.json")

	configURL := fmt.Sprintf("https://huggingface.co/%s/raw/main/config.json", modelID)
	indexURL := fmt.Sprintf("https://huggingface.co/%s/raw/main/model.safetensors.index.json", modelID)

	headers := make(map[string]string)
	headers["User-Agent"] = "gollama/2.0"

	accessToken := GetHuggingFaceToken()
	if accessToken != "" {
		headers["Authorization"] = "Bearer " + accessToken
	}

	// Download config file
	if err := DownloadFile(configURL, configPath, headers); err != nil {
		return ModelConfig{}, fmt.Errorf("failed to download config: %w", err)
	}

	// Download index file (optional, try but don't fail if not found)
	if err := DownloadFile(indexURL, indexPath, headers); err != nil {
		logging.WarnLogger.Printf("Failed to download index file: %v, using config only", err)
	}

	// Read and parse config file
	configFile, err := os.ReadFile(configPath)
	if err != nil {
		return ModelConfig{}, fmt.Errorf("failed to read config file: %w", err)
	}

	var config ModelConfig
	if err := json.Unmarshal(configFile, &config); err != nil {
		return ModelConfig{}, fmt.Errorf("failed to parse config: %w", err)
	}

	// Try to get total size from index file if available
	if _, err := os.Stat(indexPath); err == nil {
		indexFile, err := os.ReadFile(indexPath)
		if err == nil {
			var index struct {
				Metadata struct {
					TotalSize float64 `json:"total_size"`
				} `json:"metadata"`
			}
			if err := json.Unmarshal(indexFile, &index); err == nil && index.Metadata.TotalSize > 0 {
				config.NumParams = index.Metadata.TotalSize / 2 / 1e9
			}
		}
	}

	// Estimate parameters if not available
	if config.NumParams == 0 {
		// Estimate based on model size naming convention
		config.NumParams = estimateParamsFromModelName(modelID)
	}

	// Set defaults for missing values
	if config.MaxPositionEmbeddings == 0 {
		config.MaxPositionEmbeddings = 2048
	}
	if config.NumHiddenLayers == 0 {
		config.NumHiddenLayers = int(math.Round(config.NumParams * 1e9 / (12 * float64(config.HiddenSize) * float64(config.HiddenSize))))
	}
	if config.NumAttentionHeads == 0 {
		config.NumAttentionHeads = config.HiddenSize / 64
	}
	if config.NumKeyValueHeads == 0 {
		config.NumKeyValueHeads = config.NumAttentionHeads
	}
	if config.IntermediateSize == 0 {
		config.IntermediateSize = 4 * config.HiddenSize
	}
	if config.VocabSize == 0 {
		config.VocabSize = 32000
	}

	// Cache the config
	cacheMutex.Lock()
	modelConfigCache[modelID] = config
	cacheMutex.Unlock()

	logging.DebugLogger.Printf("Loaded model config: %+v", config)
	
	return config, nil
}

// estimateParamsFromModelName estimates parameters from model name
func estimateParamsFromModelName(modelID string) float64 {
	modelIDLower := strings.ToLower(modelID)
	
	// Common patterns in model names
	switch {
	case strings.Contains(modelIDLower, "7b"):
		return 7.0
	case strings.Contains(modelIDLower, "13b"):
		return 13.0
	case strings.Contains(modelIDLower, "34b"):
		return 34.0
	case strings.Contains(modelIDLower, "70b"):
		return 70.0
	case strings.Contains(modelIDLower, "1.5b"):
		return 1.5
	case strings.Contains(modelIDLower, "3b"):
		return 3.0
	case strings.Contains(modelIDLower, "1b"):
		return 1.0
	case strings.Contains(modelIDLower, "350m"):
		return 0.35
	case strings.Contains(modelIDLower, "500m"):
		return 0.5
	default:
		// Try to extract number followed by 'b'
		re := regexp.MustCompile(`(\d+\.?\d*)b`)
		matches := re.FindStringSubmatch(modelIDLower)
		if len(matches) > 1 {
			if val, err := strconv.ParseFloat(matches[1], 64); err == nil {
				return val
			}
		}
		return 7.0 // Default
	}
}

// GetBPWValues calculates the BPW values based on the input
func GetBPWValues(bpw float64, kvCacheQuant KVCacheQuantisation) BPWValues {
	logging.DebugLogger.Println("Calculating BPW values...")
	var lmHeadBPW, kvCacheBPW float64

	if bpw > 6.0 {
		lmHeadBPW = 8.0
	} else {
		lmHeadBPW = 6.0
	}

	switch kvCacheQuant {
	case KVCacheFP16:
		kvCacheBPW = 16
	case KVCacheQ8_0:
		kvCacheBPW = 8
	case KVCacheQ4_0:
		kvCacheBPW = 4
	default:
		kvCacheBPW = 16 // Default to fp16 if not specified
	}

	return BPWValues{
		BPW:        bpw,
		LMHeadBPW:  lmHeadBPW,
		KVCacheBPW: kvCacheBPW,
	}
}

// CalculateVRAM calculates the VRAM usage for a given model and configuration
func CalculateVRAM(modelID string, bpw float64, context int, kvCacheQuant KVCacheQuantisation, ollamaModelInfo *OllamaModelInfo) (float64, error) {
	logging.DebugLogger.Println("Calculating VRAM usage...")

	var config ModelConfig
	var err error

	if ollamaModelInfo != nil {
		// Use Ollama model information
		paramCount, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "parameter_count")
		contextLength, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "context_length")
		blockCount, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "block_count")
		embeddingLength, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "embedding_length")
		headCountKV, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "attention.head_count_kv")
		headCount, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "attention.head_count")
		feedForwardLength, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "feed_forward_length")
		vocabSize, _ := extractModelInfo(ollamaModelInfo.ModelInfo, "vocab_size")

		config = ModelConfig{
			NumParams:             paramCount / 1e9, // Convert to billions
			MaxPositionEmbeddings: int(contextLength),
			NumHiddenLayers:       int(blockCount),
			HiddenSize:            int(embeddingLength),
			NumKeyValueHeads:      int(headCountKV),
			NumAttentionHeads:     int(headCount),
			IntermediateSize:      int(feedForwardLength),
			VocabSize:             int(vocabSize),
		}

		// Estimate missing values
		if config.HiddenSize == 0 {
			config.HiddenSize = int(math.Sqrt(paramCount / 1000))
		}
		if config.NumHiddenLayers == 0 {
			config.NumHiddenLayers = int(math.Round(config.NumParams * 1e9 / (12 * float64(config.HiddenSize) * float64(config.HiddenSize))))
		}
		if config.NumAttentionHeads == 0 {
			config.NumAttentionHeads = config.HiddenSize / 64
		}
		if config.NumKeyValueHeads == 0 {
			config.NumKeyValueHeads = config.NumAttentionHeads
		}
		if config.IntermediateSize == 0 {
			config.IntermediateSize = 4 * config.HiddenSize
		}
		if config.VocabSize == 0 {
			config.VocabSize = 32000
		}

		// Parse BPW from quantisation level if not provided
		if bpw == 0 {
			bpw, err = ParseBPWOrQuant(ollamaModelInfo.Details.QuantizationLevel)
			if err != nil {
				return 0, fmt.Errorf("error parsing BPW from Ollama quantisation level: %w", err)
			}
		}

		logging.DebugLogger.Printf("Processed Ollama Model Config: %+v", config)
	} else {
		// Use Hugging Face model information
		config, err = GetModelConfig(modelID)
		if err != nil {
			return 0, err
		}
	}

	bpwValues := GetBPWValues(bpw, kvCacheQuant)

	if context == 0 {
		if ollamaModelInfo != nil {
			contextLength, found := extractModelInfo(ollamaModelInfo.ModelInfo, "context_length")
			if found {
				context = int(contextLength)
			}
		}
		if context == 0 {
			context = config.MaxPositionEmbeddings
		}
	}
	if context == 0 {
		context = 2048 // Default context if not provided
	}

	vram := CalculateVRAMRaw(config, bpwValues, context, 1, true)
	return math.Round(vram*100) / 100, nil
}

// ParseBPWOrQuant takes a string and returns a float64 BPW value
func ParseBPWOrQuant(input string) (float64, error) {
	if input == "" {
		return 0, fmt.Errorf("empty input")
	}

	// First, try to parse as a float64 (direct BPW value)
	bpw, err := strconv.ParseFloat(input, 64)
	if err == nil {
		return bpw, nil
	}

	// If parsing as float fails, check if it's a valid quantisation type
	input = strings.ToUpper(input) // Convert to uppercase for case-insensitive matching
	if bpw, ok := GGUFMapping[input]; ok {
		return bpw, nil
	}

	// If not found, try to find a close match
	var closestMatch string
	var minDistance int = len(input)
	for key := range GGUFMapping {
		distance := levenshteinDistance(input, key)
		if distance < minDistance {
			minDistance = distance
			closestMatch = key
		}
	}

	if closestMatch != "" {
		return 0, fmt.Errorf("invalid quantisation type: %s. Did you mean %s?", input, closestMatch)
	}

	return 0, fmt.Errorf("invalid quantisation or BPW value: %s", input)
}

// levenshteinDistance calculates the Levenshtein distance between two strings
func levenshteinDistance(s1, s2 string) int {
	s1 = strings.ToUpper(s1)
	s2 = strings.ToUpper(s2)
	m := len(s1)
	n := len(s2)
	d := make([][]int, m+1)
	for i := range d {
		d[i] = make([]int, n+1)
	}
	for i := 0; i <= m; i++ {
		d[i][0] = i
	}
	for j := 0; j <= n; j++ {
		d[0][j] = j
	}
	for j := 1; j <= n; j++ {
		for i := 1; i <= m; i++ {
			if s1[i-1] == s2[j-1] {
				d[i][j] = d[i-1][j-1]
			} else {
				min := d[i-1][j]
				if d[i][j-1] < min {
					min = d[i][j-1]
				}
				if d[i-1][j-1] < min {
					min = d[i-1][j-1]
				}
				d[i][j] = min + 1
			}
		}
	}
	return d[m][n]
}

// GenerateQuantTable generates a table of VRAM estimates for different quantisations
func GenerateQuantTable(modelID string, fitsVRAM float64, ollamaModelInfo *OllamaModelInfo, topContext int) (QuantResultTable, error) {
	if fitsVRAM == 0 {
		var err error
		fitsVRAM, err = GetAvailableMemory()
		if err != nil {
			log.Printf("Failed to get available memory: %v. Using default value.", err)
			fitsVRAM = 8.0 // Conservative default for Windows
		}
		log.Printf("Using %.2f GB as available memory for VRAM estimation", fitsVRAM)
	}

	table := QuantResultTable{ModelID: modelID, FitsVRAM: fitsVRAM}

	// Generate context sizes based on the topContext
	contextSizes := generateContextSizes(topContext)

	if ollamaModelInfo == nil {
		_, err := GetModelConfig(modelID)
		if err != nil {
			return QuantResultTable{}, err
		}
	}

	// Limit to common quantisations for performance
	commonQuants := []string{"Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "FP16"}
	
	for _, quantType := range commonQuants {
		bpw, exists := GGUFMapping[quantType]
		if !exists {
			continue
		}
		
		var result QuantResult
		result.QuantType = quantType
		result.BPW = bpw
		result.Contexts = make(map[int]ContextVRAM)

		for _, context := range contextSizes {
			vramFP16, err := CalculateVRAM(modelID, bpw, context, KVCacheFP16, ollamaModelInfo)
			if err != nil {
				return QuantResultTable{}, err
			}
			vramQ8_0, err := CalculateVRAM(modelID, bpw, context, KVCacheQ8_0, ollamaModelInfo)
			if err != nil {
				return QuantResultTable{}, err
			}
			vramQ4_0, err := CalculateVRAM(modelID, bpw, context, KVCacheQ4_0, ollamaModelInfo)
			if err != nil {
				return QuantResultTable{}, err
			}
			result.Contexts[context] = ContextVRAM{
				VRAM:     vramFP16,
				VRAMQ8_0: vramQ8_0,
				VRAMQ4_0: vramQ4_0,
			}
		}
		table.Results = append(table.Results, result)
	}

	// Sort the results from lowest BPW to highest
	sort.Slice(table.Results, func(i, j int) bool {
		return table.Results[i].BPW < table.Results[j].BPW
	})

	return table, nil
}

// generateContextSizes generates a slice of context sizes based on the topContext
func generateContextSizes(topContext int) []int {
	sizes := []int{2048, 8192}
	current := 16384
	for current <= topContext {
		sizes = append(sizes, current)
		current *= 2
	}
	if current/2 < topContext {
		sizes = append(sizes, topContext)
	}
	return sizes
}

// PrintFormattedTable prints the VRAM estimation table with Windows console-friendly formatting
func PrintFormattedTable(table QuantResultTable) string {
	var buf bytes.Buffer

	// Add the description header
	buf.WriteString(styles.HeaderStyle().Bold(true).Render(vramDescription))
	buf.WriteString("\n")

	// Get context sizes from the first result
	var contextSizes []int
	if len(table.Results) > 0 {
		for context := range table.Results[0].Contexts {
			contextSizes = append(contextSizes, context)
		}
		sort.Ints(contextSizes)
	}

	// Create table header
	header := []string{"QUANT", "BPW"}
	for _, context := range contextSizes {
		if context >= 1024 {
			header = append(header, fmt.Sprintf("%dK", context/1024))
		} else {
			header = append(header, fmt.Sprintf("%d", context))
		}
	}

	// Calculate column widths
	colWidths := make([]int, len(header))
	for i, h := range header {
		colWidths[i] = runewidth.StringWidth(h) + 2
	}

	// Print header
	buf.WriteString("╔")
	for i, width := range colWidths {
		buf.WriteString(strings.Repeat("═", width))
		if i < len(colWidths)-1 {
			buf.WriteString("╦")
		}
	}
	buf.WriteString("╗\n")

	buf.WriteString("║")
	for i, h := range header {
		padding := colWidths[i] - runewidth.StringWidth(h)
		leftPad := padding / 2
		rightPad := padding - leftPad
		buf.WriteString(strings.Repeat(" ", leftPad))
		buf.WriteString(h)
		buf.WriteString(strings.Repeat(" ", rightPad))
		buf.WriteString("║")
	}
	buf.WriteString("\n")

	buf.WriteString("╠")
	for i, width := range colWidths {
		buf.WriteString(strings.Repeat("═", width))
		if i < len(colWidths)-1 {
			buf.WriteString("╬")
		}
	}
	buf.WriteString("╣\n")

	// Print data rows
	for _, result := range table.Results {
		buf.WriteString("║")
		
		// Quant type
		quantCell := fmt.Sprintf(" %-5s", result.QuantType)
		buf.WriteString(quantCell)
		buf.WriteString("║")
		
		// BPW
		bpwCell := fmt.Sprintf(" %-4.2f ", result.BPW)
		buf.WriteString(bpwCell)
		buf.WriteString("║")
		
		// VRAM estimates for each context
		for i, context := range contextSizes {
			vram, ok := result.Contexts[context]
			if !ok {
				buf.WriteString("      -      ║")
				continue
			}

			var cellContent string
			if context >= 16384 {
				fp16Str := getColouredVRAM(vram.VRAM, fmt.Sprintf("%.1f", vram.VRAM), table.FitsVRAM, false)
				q8Str := getColouredVRAM(vram.VRAMQ8_0, fmt.Sprintf("%.1f", vram.VRAMQ8_0), table.FitsVRAM, false)
				q4Str := getColouredVRAM(vram.VRAMQ4_0, fmt.Sprintf("%.1f", vram.VRAMQ4_0), table.FitsVRAM, false)
				cellContent = fmt.Sprintf("%s(%s,%s)", fp16Str, q8Str, q4Str)
			} else {
				cellContent = getColouredVRAM(vram.VRAM, fmt.Sprintf("%.1f", vram.VRAM), table.FitsVRAM, false)
			}
			
			// Pad to column width
			cellWidth := colWidths[i+2] - 2 // +2 for QUANT and BPW columns
			padding := cellWidth - runewidth.StringWidth(cellContent)
			if padding > 0 {
				cellContent += strings.Repeat(" ", padding)
			}
			
			buf.WriteString(" ")
			buf.WriteString(cellContent)
			buf.WriteString(" ║")
		}
		
		buf.WriteString("\n")
	}

	// Print footer
	buf.WriteString("╚")
	for i, width := range colWidths {
		buf.WriteString(strings.Repeat("═", width))
		if i < len(colWidths)-1 {
			buf.WriteString("╩")
		}
	}
	buf.WriteString("╝\n")

	// Add model info and memory constraint
	modelInfo := fmt.Sprintf("📊 VRAM Estimation for Model: %s", table.ModelID)
	if table.FitsVRAM > 0 {
		modelInfo += fmt.Sprintf(" (Memory Constraint: %.1f GB)", table.FitsVRAM)
	}

	return styles.ItemNameStyle(0).Render(fmt.Sprintf("%s\n\n%s", modelInfo, buf.String()))
}

// getColouredVRAM returns VRAM string with appropriate coloring
func getColouredVRAM(vram float64, vramStr string, fitsVRAM float64, useColor bool) string {
	if fitsVRAM > 0 {
		if vram > fitsVRAM {
			if useColor {
				return styles.VRAMExceedsStyle().Render(vramStr)
			}
			return fmt.Sprintf("[%s]", vramStr) // Mark exceeded with brackets
		} else {
			if useColor {
				return styles.VRAMWithinStyle().Render(vramStr)
			}
			return vramStr
		}
	} else {
		if useColor {
			return styles.VRAMUnknownStyle().Render(vramStr)
		}
		return vramStr
	}
}

// CalculateContext calculates the maximum context for a given memory constraint
func CalculateContext(modelID string, memory, bpw float64, kvCacheQuant KVCacheQuantisation, ollamaModelInfo *OllamaModelInfo, topContext int) (int, error) {
	logging.DebugLogger.Println("Calculating context...")

	var maxContext int
	if ollamaModelInfo != nil {
		contextLength, found := extractModelInfo(ollamaModelInfo.ModelInfo, "context_length")
		if found {
			maxContext = int(contextLength)
		} else {
			maxContext = topContext
		}
	} else {
		config, err := GetModelConfig(modelID)
		if err != nil {
			return 0, err
		}
		maxContext = config.MaxPositionEmbeddings
	}

	if topContext < maxContext {
		maxContext = topContext
	}

	minContext := 512
	low, high := minContext, maxContext
	for low < high {
		mid := (low + high + 1) / 2
		vram, err := CalculateVRAM(modelID, bpw, mid, kvCacheQuant, ollamaModelInfo)
		if err != nil {
			return 0, err
		}
		if vram > memory {
			high = mid - 1
		} else {
			low = mid
		}
	}

	context := low
	for context <= maxContext {
		vram, err := CalculateVRAM(modelID, bpw, context, kvCacheQuant, ollamaModelInfo)
		if err != nil {
			return 0, err
		}
		if vram >= memory {
			break
		}
		context += 100
	}

	return context - 100, nil
}

// CalculateBPW calculates the best BPW for a given memory and context constraint
func CalculateBPW(modelID string, memory float64, context int, kvCacheQuant KVCacheQuantisation, quantType string, ollamaModelInfo *OllamaModelInfo) (interface{}, error) {
	logging.DebugLogger.Println("Calculating BPW...")

	switch quantType {
	case "exl2":
		for _, bpw := range EXL2Options {
			vram, err := CalculateVRAM(modelID, bpw, context, kvCacheQuant, ollamaModelInfo)
			if err != nil {
				return nil, err
			}
			if vram < memory {
				return bpw, nil
			}
		}
	case "gguf":
		for name, bpw := range GGUFMapping {
			vram, err := CalculateVRAM(modelID, bpw, context, kvCacheQuant, ollamaModelInfo)
			if err != nil {
				return nil, err
			}
			if vram < memory {
				return name, nil
			}
		}
	default:
		return nil, fmt.Errorf("invalid quantisation type: %s", quantType)
	}

	return nil, fmt.Errorf("no suitable BPW found for the given memory constraint")
}

// GetModelSummary provides a quick summary of VRAM requirements
func GetModelSummary(modelID string, apiURL string) (string, error) {
	var ollamaModelInfo *OllamaModelInfo
	var err error

	if strings.Contains(modelID, ":") {
		ollamaModelInfo, err = FetchOllamaModelInfo(apiURL, modelID)
		if err != nil {
			return "", err
		}
	}

	availableVRAM, err := GetAvailableMemory()
	if err != nil {
		availableVRAM = 8.0 // Default fallback
	}

	var summary strings.Builder
	summary.WriteString(fmt.Sprintf("Model: %s\n", modelID))
	summary.WriteString(fmt.Sprintf("Available VRAM: %.1f GB\n", availableVRAM))
	summary.WriteString("\nRecommended Quantisations:\n")

	// Test common quantisations
	commonQuants := []struct {
		name string
		bpw  float64
	}{
		{"Q4", 4.55},
		{"Q5", 5.54},
		{"Q6", 6.59},
		{"Q8", 8.5},
	}

	for _, quant := range commonQuants {
		vram, err := CalculateVRAM(modelID, quant.bpw, 4096, KVCacheFP16, ollamaModelInfo)
		if err != nil {
			continue
		}

		fits := vram <= availableVRAM
		status := "✓"
		if !fits {
			status = "✗"
		}

		summary.WriteString(fmt.Sprintf("  %s %-4s: %5.1f GB %s\n", 
			status, quant.name, vram, 
			map[bool]string{true: "(Fits)", false: "(Too large)"}[fits]))
	}

	return summary.String(), nil
}

// Add regex import at top
import "regexp"

// Add unsafe import at top
import "unsafe"
