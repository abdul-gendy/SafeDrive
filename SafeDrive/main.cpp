///////////////////////////////////////////////////
// Written by: Abdulrahman Elgendy
// Last Update: 12-01-2022
//////////////////////////////////////////////////

#include "SafeDrive.h"

int main() {
	int deviceId = 0;
	safeDrive SD(deviceId);
	SD.analyzeStream();
}