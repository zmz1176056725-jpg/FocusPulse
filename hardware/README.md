# Hardware

The Arduino sketch in this folder is based on the open-source
**PulseSensor Amped** code by World Famous Electronics LLC.

- Upstream: https://github.com/WorldFamousElectronics/PulseSensor_Amped_Arduino
- Website: https://pulsesensor.com

## Wiring

| Pulse Sensor wire | Arduino pin |
| --- | --- |
| Red (VCC) | 3.3V or 5V |
| Black (GND) | GND |
| Purple (signal) | Analog **A0** |

## Upload

Open `PulseSensorAmped_Arduino_1dot2.ino` in the Arduino IDE and upload it to your board.
The sketch samples the pulse waveform at 500 Hz (Timer 2 interrupt) and prints
`S` (raw signal), `B` (BPM) and `Q` (IBI) frames over serial at 115200 baud.
