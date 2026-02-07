package cuda

import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/StackExchange/wmi"
	"golang.org/x/sys/windows"
)

// GPUInfo contains comprehensive GPU information
type GPUInfo struct {
	Name         string
	Vendor       string
	VRAMGB       float64
	VRAMBytes    uint64
	DriverVersion string
	DeviceID     string
}

// WMI VideoController struct for WMI queries
type win32VideoController struct {
	Name           string
	AdapterRAM     uint64
	DriverVersion  string
	PNPDeviceID    string
	VideoProcessor string
}

// GetCUDAVRAM returns the total VRAM of all GPUs in GB (Windows-specific implementation)
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

// GetAllGPUInfo returns detailed information about all GPUs in the system
func GetAllGPUInfo() ([]GPUInfo, error) {
	var gpus []GPUInfo
	
	// Method 1: Try WMI first (most reliable on Windows)
	wmiGPUs, err := getGPUInfoViaWMI()
	if err == nil && len(wmiGPUs) > 0 {
		gpus = append(gpus, wmiGPUs...)
	}
	
	// Method 2: Try DirectX/DXGI if WMI didn't give us VRAM info
	if len(gpus) == 0 || (len(gpus) > 0 && gpus[0].VRAMGB == 0) {
		dxgiGPUs, err := getGPUInfoViaDXGI()
		if err == nil && len(dxgiGPUs) > 0 {
			// If WMI gave us GPU info but no VRAM, merge with DXGI VRAM info
			if len(gpus) > 0 {
				for i := range gpus {
					if i < len(dxgiGPUs) {
						gpus[i].VRAMGB = dxgiGPUs[i].VRAMGB
						gpus[i].VRAMBytes = dxgiGPUs[i].VRAMBytes
					}
				}
			} else {
				gpus = dxgiGPUs
			}
		}
	}
	
	// Method 3: Fallback to NVML for NVIDIA GPUs
	if len(gpus) == 0 {
		nvmlGPUs, err := getGPUInfoViaNVML()
		if err == nil && len(nvmlGPUs) > 0 {
			gpus = nvmlGPUs
		}
	}
	
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPU information available via any detection method")
	}
	
	return gpus, nil
}

// getGPUInfoViaWMI uses Windows Management Instrumentation to get GPU info
func getGPUInfoViaWMI() ([]GPUInfo, error) {
	var videoControllers []win32VideoController
	query := "SELECT Name, AdapterRAM, DriverVersion, PNPDeviceID, VideoProcessor FROM Win32_VideoController"
	
	err := wmi.Query(query, &videoControllers)
	if err != nil {
		return nil, fmt.Errorf("WMI query failed: %w", err)
	}
	
	if len(videoControllers) == 0 {
		return nil, fmt.Errorf("no video controllers found via WMI")
	}
	
	var gpus []GPUInfo
	for _, vc := range videoControllers {
		// Skip invalid entries
		if vc.Name == "" {
			continue
		}
		
		gpu := GPUInfo{
			Name:          vc.Name,
			VRAMBytes:     vc.AdapterRAM,
			VRAMGB:        float64(vc.AdapterRAM) / (1024 * 1024 * 1024),
			DriverVersion: vc.DriverVersion,
			DeviceID:      vc.PNPDeviceID,
		}
		
		// Determine vendor from name
		gpu.Vendor = detectVendorFromName(vc.Name)
		if gpu.Vendor == "" {
			gpu.Vendor = detectVendorFromDeviceID(vc.PNPDeviceID)
		}
		
		gpus = append(gpus, gpu)
	}
	
	return gpus, nil
}

// getGPUInfoViaDXGI uses DirectX Graphics Infrastructure to get GPU info
func getGPUInfoViaDXGI() ([]GPUInfo, error) {
	// Try to load dxgi.dll
	dxgiDLL := syscall.NewLazyDLL("dxgi.dll")
	if dxgiDLL.Load() != nil {
		return nil, fmt.Errorf("dxgi.dll not available")
	}
	
	// For a production implementation, you would use the full DXGI COM API
	// This is a simplified version - in production, use github.com/go-ole/go-ole
	
	// Placeholder: In reality, you'd:
	// 1. Initialize COM
	// 2. Create DXGIFactory
	// 3. Enumerate adapters
	// 4. Query adapter description and dedicated video memory
	
	// For now, we'll return an empty result and let other methods handle it
	return nil, fmt.Errorf("DXGI implementation requires COM integration")
}

// getGPUInfoViaNVML uses NVIDIA Management Library to get GPU info
func getGPUInfoViaNVML() ([]GPUInfo, error) {
	// Try to load nvml.dll
	nvmlDLL := syscall.NewLazyDLL("nvml.dll")
	if nvmlDLL.Load() != nil {
		return nil, fmt.Errorf("nvml.dll not available")
	}
	
	// Note: Full NVML implementation requires the NVIDIA NVML Go bindings
	// or implementing the C API calls via syscall
	// This is a placeholder showing the approach
	
	var gpus []GPUInfo
	
	// In production, you would:
	// 1. Call nvmlInit_v2
	// 2. Call nvmlDeviceGetCount_v2
	// 3. For each device, get handle and query:
	//    - nvmlDeviceGetName
	//    - nvmlDeviceGetMemoryInfo
	//    - nvmlDeviceGetPciInfo (for device ID)
	
	return gpus, nil
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
		return ""
	}
}

// detectVendorFromDeviceID determines GPU vendor from PNP device ID
func detectVendorFromDeviceID(deviceID string) string {
	if strings.Contains(deviceID, "VEN_10DE") {
		return "NVIDIA"
	}
	if strings.Contains(deviceID, "VEN_1002") || strings.Contains(deviceID, "VEN_1022") {
		return "AMD"
	}
	if strings.Contains(deviceID, "VEN_8086") {
		return "Intel"
	}
	return ""
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

// GetPrimaryGPU returns the primary/display GPU
func GetPrimaryGPU() (*GPUInfo, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return nil, err
	}
	
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs found")
	}
	
	// First GPU from WMI is typically the primary display adapter
	return &gpus[0], nil
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

// GetAvailableVRAM estimates available VRAM (simplified - subtracts a buffer)
func GetAvailableVRAM() (float64, error) {
	totalVRAM, err := GetCUDAVRAM()
	if err != nil {
		return 0, err
	}
	
	// Reserve 10% for system use
	availableVRAM := totalVRAM * 0.9
	
	// Ensure minimum available VRAM
	if availableVRAM < 0.5 {
		availableVRAM = 0.5
	}
	
	return availableVRAM, nil
}

// GetGPUCount returns the number of GPUs in the system
func GetGPUCount() (int, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return 0, err
	}
	
	// Filter out GPUs with 0 VRAM (might be virtual adapters)
	var validGPUs int
	for _, gpu := range gpus {
		if gpu.VRAMGB > 0 {
			validGPUs++
		}
	}
	
	return validGPUs, nil
}

// GetNVIDIAVRAM specifically gets VRAM for NVIDIA GPUs only
func GetNVIDIAVRAM() (float64, error) {
	gpus, err := GetAllGPUInfo()
	if err != nil {
		return 0, err
	}
	
	var nvidiaVRAM float64
	for _, gpu := range gpus {
		if gpu.Vendor == "NVIDIA" {
			nvidiaVRAM += gpu.VRAMGB
		}
	}
	
	if nvidiaVRAM == 0 {
		return 0, fmt.Errorf("no NVIDIA GPUs found or unable to read VRAM")
	}
	
	return nvidiaVRAM, nil
}

// Helper function to check DLL availability
func isDLLAvailable(dllName string) bool {
	handle, err := windows.LoadLibrary(dllName)
	if err != nil {
		return false
	}
	windows.FreeLibrary(handle)
	return true
}

// CheckNVIDIADriverVersion returns the NVIDIA driver version if available
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

// Strings utility (since we're using strings.ToLower)
import "strings"

// Note: The actual implementation would need proper error handling,
// caching of results, and potentially COM initialization for DXGI.
