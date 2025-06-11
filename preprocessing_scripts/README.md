# MIDI Preprocessing Pipeline

This part of the project provides a pipeline for preprocessing MIDI files, specifically tailored for classical music datasets. The pipeline consists of three Python scripts that clean, analyze, and transpose MIDI files to prepare them for machine learning tasks.

## Overview

The preprocessing pipeline performs the following steps:
1. **Cleaning**: Ensures MIDI files have exactly two piano tracks.
2. **Analysis**: Extracts key and scale information from MIDI files.
3. **Transposition**: Transposes MIDI files to a standardized keys of C major or A minor based on their original key and scale.

The scripts utilize the `music21` library for MIDI processing and leverage parallel processing with Python's `multiprocessing` module to handle large datasets efficiently.

## Scripts

### 1. clean_data.py`
- **Purpose**: Cleans MIDI files by ensuring they have exactly two tracks, both assigned to piano.
- **Input**: Raw MIDI files in `data/../data/Classical_Music_Midi_dataset/`.
- **Output**: Cleaned MIDI files in `data/../data/Cleaned_Midi_dataset/`.
- **Key Features**:
  - Checks for exactly two tracks; skips files with a different track count.
  - Assigns piano instruments to both tracks.
  - Uses parallel processing for efficiency.
  - Logs progress and errors to `../outputs/clean_data_log.txt`.

### 2. `analyse_data.py`
- **Purpose**: Analyzes cleaned MIDI files to extract key and scale information.
- **Input**: Cleaned MIDI files in `data/../data/Cleaned_Midi_dataset/`.
- **Output**: JSON file (`../outputs_2_data/key_scales.json`) containing key and scale data for each MIDI file.
- **Key Features**:
  - Uses `music21`’s KrumhanslSchmuckler` algorithm for key estimation.
  - Processes files in parallel.
  - Logs progress and errors to `../outputs_2/analyze_data_log.txt`.

### 3. `transpose_midi.py`
- **Purpose**: Transposes MIDI files to C major (for major scale) or A minor (for minor scale).
- **Input**: 
  - Cleaned MIDI in files in `data/Cleaned_Midi_dataset/`.
  - Key/scale data from `../outputs_2/key_scales.json`.
- **Output**: Transposed MIDI files in `data/../transposed_midi/`.
- **Key Features**:
  - Calculates the shortest transposition interval to the nearest target pitch.
  - Skips files that already exist or lack key/scale information.
  - Uses parallel processing for efficiency.
  - Logs progress and errors to `../outputs_2/transpose_midi_log.txt`.

## Notes

- **Parallel Processing**: The scripts use `multiprocessing` to leverage multiple CPU cores, significantly speeding up processing for large datasets.
- **Error Handling**: Comprehensive logging captures errors and progress for debugging and monitoring.
- **File Overwrite Protection**: The scripts skip processing if output files already exist to avoid redundant work.
- **Dataset Specifics**: The pipeline is designed for classical music MIDI files but can be adapted for other genres by modifying the cleaning and transposition logic.

## Future Improvements

- Add support for additional MIDI track configurations.
- Incorporate more advanced key detection algorithms.
- Allow user-defined target keys for transposition.
- Implement data validation checks before processing.
