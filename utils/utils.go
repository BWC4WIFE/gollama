package utils

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/sammcj/gollama/v2/logging"
)

func GetHomeDir() string {
	// On Windows, os.UserHomeDir() returns paths like C:\Users\Username
	homeDir, err := os.UserHomeDir()
	if err != nil {
		logging.ErrorLogger.Printf("Failed to get user home directory: %v\n", err)
		return ""
	}
	return homeDir
}

// GetConfigDir returns the directory of the configuration JSON file for Windows.
func GetConfigDir() string {
	// On Windows, standard config location is in AppData
	appData := os.Getenv("APPDATA")
	if appData != "" {
		// Use APPDATA\gollama for configuration
		return filepath.Join(appData, "gollama")
	}
	// Fallback to home directory if APPDATA is not set
	return filepath.Join(GetHomeDir(), ".config", "gollama")
}

// GetConfigPath returns the path to the configuration JSON file for Windows.
func GetConfigPath() string {
	return filepath.Join(GetConfigDir(), "config.json")
}

// IsLocalhost checks if a URL or host string refers to localhost
func IsLocalhost(url string) bool {
	return strings.Contains(strings.ToLower(url), "localhost") || 
		strings.Contains(url, "127.0.0.1") ||
		strings.Contains(strings.ToLower(url), "::1")
}

// JoinPath is a Windows-specific path joiner that ensures proper path separators
func JoinPath(elem ...string) string {
	return filepath.Join(elem...)
}

// NormalizePath converts a path to use Windows separators and cleans it
func NormalizePath(path string) string {
	return filepath.Clean(path)
}
