#include "SafeDrive.h"

int main() {
	int deviceId = 0;
	safeDrive SD(deviceId);
	SD.analyzeStream();
}