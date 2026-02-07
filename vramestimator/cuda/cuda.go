package cuda

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/StackExchange/wmi"
)

// GPUInfo contains comprehensive GPU information
type GPUInfo struct {
	Name          string
	Vendor        string
	VRAMGB        float64
	VRAMBytes     uint64
	DriverVersion string
	DeviceID      string
	IsPrimary     bool
}

// WMI VideoController struct
type win32VideoController struct {
	Name           string
	AdapterRAM     uint64
	DriverVersion  string
	PNPDeviceID    string
	VideoProcessor string
	Status         string
	AdapterDACType string
}

// WMI ComputerSystem struct for checking system RAM (context)
type win32ComputerSystem struct {
	TotalPhysicalMemory uint64
}

// GetCUDAVRAM returns the total VRAM of all GPUs in GB
func GetCUDAVRAM() (float64, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return 0, fmt.Errorf("failed to get GPU info: %w", err)
	}
	
	if len(gpus) == 0 {
		return 0, fmt.Errorf("no GPUs detected")
	}
	
	var totalVRAM float64
	for _, gpu := range gpus {
		totalVRAM += gpu.VRAMGB
	}
	
	return totalVRAM, nil
}

// GetAllGPUInfo returns detailed information about all GPUs
func GetAllGPUInfo() ([]GPUInfo, error) {
	var videoControllers []win32VideoController
	query := "SELECT Name, AdapterRAM, DriverVersion, PNPDeviceID, VideoProcessor, Status, AdapterDACType FROM Win32_VideoController WHERE Status = 'OK'"
	
	err := wmi.QueryNamespace(query, &videoControllers, "root\\CIMV2")
	if err != nil {
		return nil, fmt.Errorf("WMI query failed: %w", err)
	}
	
	if len(videoControllers) == 0 {
		// Try without status filter
		err = wmi.QueryNamespace("SELECT Name, AdapterRAM, DriverVersion, PNPDeviceID, VideoProcessor FROM Win32_VideoController", 
			&videoControllers, "root\\CIMV2")
		if err != nil {
			return nil, fmt.Errorf("WMI query (unfiltered) failed: %w", err)
		}
	}
	
	if len(videoControllers) == 0 {
		return nil, fmt.Errorf("no video controllers found")
	}
	
	var gpus []GPUInfo
	for i, vc := range videoControllers {
		// Filter out virtual adapters and invalid entries
		if vc.Name == "" || isVirtualAdapter(vc.Name) {
			continue
		}
		
		// Skip Microsoft Basic Display Adapter (usually fallback)
		if strings.Contains(strings.ToLower(vc.Name), "microsoft basic display") {
			continue
		}
		
		vramBytes := vc.AdapterRAM
		// Sometimes WMI reports 0 or very small values for integrated graphics
		// Apply some sanity checks
		if vramBytes < 134217728 { // Less than 128MB
			// Might be reporting shared memory incorrectly
			// Try to estimate based on system RAM for integrated graphics
			if isIntegratedGraphics(vc.Name) {
				systemRAM, _ := getSystemRAM()
				if systemRAM > 0 {
					// Integrated graphics typically use 1-2GB of system RAM
					vramBytes = min(systemRAM/4, 2*1024*1024*1024) // Cap at 2GB
				}
			}
		}
		
		gpu := GPUInfo{
			Name:          strings.TrimSpace(vc.Name),
			VRAMBytes:     vramBytes,
			VRAMGB:        float64(vramBytes) / (1024 * 1024 * 1024),
			DriverVersion: strings.TrimSpace(vc.DriverVersion),
			DeviceID:      strings.TrimSpace(vc.PNPDeviceID),
			IsPrimary:     i == 0, // First adapter is typically primary
		}
		
		// Determine vendor
		gpu.Vendor = detectVendorFromName(gpu.Name)
		if gpu.Vendor == "" {
			gpu.Vendor = detectVendorFromDeviceID(gpu.DeviceID)
		}
		if gpu.Vendor == "" {
			gpu.Vendor = "Unknown"
		}
		
		// Round VRAM to 2 decimal places for readability
		gpu.VRAMGB = roundToTwoDecimals(gpu.VRAMGB)
		
		gpus = append(gpus, gpu)
	}
	
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no valid GPUs found after filtering")
	}
	
	return gpus, nil
}

// Helper functions
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
		"microsoft remote display adapter",
	}
	
	for _, indicator := range virtualIndicators {
		if strings.Contains(nameLower, indicator) {
			return true
		}
	}
	
	return false
}

func isIntegratedGraphics(name string) bool {
	nameLower := strings.ToLower(name)
	integratedIndicators := []string{
		"intel",
		"hd graphics",
		"uhd graphics",
		"iris",
		"amd radeon graphics", // AMD APU graphics
		"vega",
	}
	
	for _, indicator := range integratedIndicators {
		if strings.Contains(nameLower, indicator) {
			return true
		}
	}
	
	return false
}

func getSystemRAM() (uint64, error) {
	var computerSystems []win32ComputerSystem
	err := wmi.Query("SELECT TotalPhysicalMemory FROM Win32_ComputerSystem", &computerSystems)
	if err != nil || len(computerSystems) == 0 {
		return 0, fmt.Errorf("failed to query system RAM: %w", err)
	}
	
	return computerSystems[0].TotalPhysicalMemory, nil
}

func detectVendorFromName(name string) string {
	nameLower := strings.ToLower(name)
	
	// NVIDIA patterns
	nvidiaPatterns := []string{
		"nvidia",
		"geforce",
		"quadro",
		"tesla",
		"rtx",
		"gtx",
		"titan",
		"p40",
		"p100",
		"v100",
		"a100",
		"h100",
	}
	
	// AMD patterns
	amdPatterns := []string{
		"radeon",
		"amd",
		"rx ",
		"firepro",
		"instinct",
		"vega",
		"navi",
		"radeon pro",
	}
	
	// Intel patterns
	intelPatterns := []string{
		"intel",
		"hd graphics",
		"uhd graphics",
		"iris",
		"xe graphics",
		"arc",
	}
	
	for _, pattern := range nvidiaPatterns {
		if strings.Contains(nameLower, pattern) {
			return "NVIDIA"
		}
	}
	
	for _, pattern := range amdPatterns {
		if strings.Contains(nameLower, pattern) {
			return "AMD"
		}
	}
	
	for _, pattern := range intelPatterns {
		if strings.Contains(nameLower, pattern) {
			return "Intel"
		}
	}
	
	return ""
}

func detectVendorFromDeviceID(deviceID string) string {
	if deviceID == "" {
		return ""
	}
	
	// Extract vendor ID from PNP device ID
	// Format: PCI\VEN_XXXX&DEV_XXXX&SUBSYS_XXXXXXXX&REV_XX
	re := regexp.MustCompile(`VEN_([0-9A-Fa-f]{4})`)
	matches := re.FindStringSubmatch(strings.ToUpper(deviceID))
	
	if len(matches) > 1 {
		vendorID := matches[1]
		switch vendorID {
		case "10DE":
			return "NVIDIA"
		case "1002", "1022":
			return "AMD"
		case "8086":
			return "Intel"
		case "1414":
			return "Microsoft" // Microsoft Basic Display Adapter
		}
	}
	
	return ""
}

func roundToTwoDecimals(value float64) float64 {
	rounded, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", value), 64)
	return rounded
}

func min(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

// IsNVIDIAGPUAvailable checks if an NVIDIA GPU is present
func IsNVIDIAGPUAvailable() bool {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return false
	}
	
	for _, gpu := range gpus {
		if gpu.Vendor == "NVIDIA" {
			return true
		}
	}
	
	return false
}

// GetPrimaryGPU returns the primary display GPU
func GetPrimaryGPU() (*GPUInfo, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return nil, err
	}
	
	for _, gpu := range gpus {
		if gpu.IsPrimary {
			return &gpu, nil
		}
	}
	
	// Fallback to first GPU if no primary flag
	if len(gpus) > 0 {
		return &gpus[0], nil
	}
	
	return nil, fmt.Errorf("no GPUs found")
}

// GetGPUCount returns number of physical GPUs
func GetGPUCount() (int, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return 0, err
	}
	
	return len(gpus), nil
}

// GetGPUSummary returns a formatted summary of all GPUs
func GetGPUSummary() (string, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return "", err
	}
	
	var summary strings.Builder
	summary.WriteString(fmt.Sprintf("Found %d GPU(s):\n", len(gpus)))
	
	for i, gpu := range gpus {
		summary.WriteString(fmt.Sprintf("  %d. %s (%s)\n", i+1, gpu.Name, gpu.Vendor))
		summary.WriteString(fmt.Sprintf("     VRAM: %.2f GB, Driver: %s\n", 
			gpu.VRAMGB, gpu.DriverVersion))
		if gpu.IsPrimary {
			summary.WriteString("     [Primary Display Adapter]\n")
		}
	}
	
	return summary.String(), nil
}

// CheckNVIDIADriverVersion returns NVIDIA driver version
func CheckNVIDIADriverVersion() (string, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return "", err
	}
	
	for _, gpu := range gpus {
		if gpu.Vendor == "NVIDIA" && gpu.DriverVersion != "" {
			return gpu.DriverVersion, nil
		}
	}
	
	return "", fmt.Errorf("NVIDIA driver version not available")
}

// GetTotalVRAMBytes returns total VRAM in bytes
func GetTotalVRAMBytes() (uint64, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return 0, err
	}
	
	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.VRAMBytes
	}
	
	return totalVRAM, nil
}
