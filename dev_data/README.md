# Development Dataset

## Context

The data contained in this folder was collected using 7 Movella (Xsens) DOT sensors.
User Manual: https://www.xsens.com/hubfs/Downloads/Manuals/Xsens DOT User Manual.pdf

The data were collected in a specific order with the initial files corresponding to calibration maneuvers and the latter files containing walking data.

### File Order

| #   | Collection      |
| --- | --------------- |
| 1   | npose           |
| 2   | lean forward    |
| 3   | cycle right leg |
| 4   | cycle left leg  |
| 5   | walk 5 steps    |
| 6   | walk 1          |
| 7   | walk 2          |
| 8   | walk 3          |
| 9   | walk 4          |
| 10  | walk 5          |

## Sensor Positioning

The sensors were attached to the person in the following places:

| Sensor # | Location    |
| -------- | ----------- |
| 1        | left foot   |
| 2        | left shank  |
| 3        | left thigh  |
| 4        | pelvis      |
| 5        | right thigh |
| 6        | right shank |
| 7        | right foot  |

Each sensor was oriented so the +X was pointing apporximately upward and +Z axis pointed away from the body (laterally relative to the body). Note this means
the right and left leg sensors had +Z pointing in opposite directions and the pelvis sensor +Z pointed posteriorly.

## Sensor Files Channels

PacketCounter: ignore.
SampleTimeFine: sample counter (sensors were collected at 120Hz).

| Abreviation    | Data                                                 |
| -------------- | ---------------------------------------------------- |
| PacketCounter  | ignore                                               |
| SampleTimeFine | sample counter (sensors were collected at 120Hz).    |
| Quat_w         | quaternion scalar component                          |
| Quat_x         | quaternion vector component x                        |
| Quat_y         | quaternion vector component y                        |
| Quat_z         | quaternion vector component z                        |
| Acc_X          | acceleration x (m/s/s)                               |
| Acc_Y          | acceleration y (m/s/s)                               |
| Acc_Z          | acceleration z (m/s/s)                               |
| Gyr_X          | angular rate x (deg/s)                               |
| Gyr_Y          | angular rate y (deg/s)                               |
| Gyr_Z          | angular rate z (deg/s)                               |
| Mag_X          | magnetic field vector x (arbitrary units: normalized | to earth's magnetic field) |
| Mag_Y          | magnetic field vector y (arbitrary units: normalized | to earth's magnetic field) |
| Mag_Z          | magnetic field vector z (arbitrary units: normalized | to earth's magnetic field) |
| Status         | ignore                                               |

## File number anatomy

Filename: "6_D422CD00533C_20231031.csv"

`[sensor number]_[sensor id code]_[data of collected].csv`
