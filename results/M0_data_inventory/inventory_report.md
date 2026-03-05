# Data Inventory Report

## euphonia-v1

## 1. Overview

- **Dataset**: euphonia-v1
- **Audio Directory**: `/Users/skinner/Downloads/takeout-E407`
- **Manifest CSV**: `/Users/skinner/Downloads/takeout-E407/listing.csv`
- **Run Timestamp**: 20260205-142601

**Tool Versions**:

- python: 3.13.7
- pandas: 3.0.0
- numpy: 2.3.5
- soundfile: 0.13.1

______________________________________________________________________

## 2. Manifest Integrity

- **Total Rows**: 3,623
- **Duplicate file_name Entries**: 0
- **Empty/Null Transcripts**: 0
- **Empty/Null Filenames**: 0

______________________________________________________________________

## 3. Inventory Summary

- **Total Manifest Rows**: 3,623

- **Unique Files**: 3,623

- **Total Duration**: 4.68 hours (16843.5 seconds)

- **Read Failures**: 0

- **Missing Files**: 0

**File Format Distribution**:

- WAV: 3,623

**Sample Rate Distribution**:

- 44100 Hz: 3,623

**Channel Distribution**:

- 1 channel(s): 3,623

**Duration Distribution**:

- (-0.001, 1.0\] seconds: 0
- (1.0, 3.0\] seconds: 1,054
- (3.0, 10.0\] seconds: 2,393
- (10.0, 30.0\] seconds: 176
- (30.0, 60.0\] seconds: 0
- (60.0, inf\] seconds: 0

______________________________________________________________________

## 4. Transcript Sanity

- **Blank Transcripts**: 0 (0.00%)
- **Very Short Transcripts** (≤2 words): 621 (17.14%)
- **Duplicate Transcripts**: 685 (18.91%)

**Transcript Length Distribution** (characters):

- (-0.001, 10.0\] chars: 487
- (10.0, 50.0\] chars: 2,915
- (50.0, 100.0\] chars: 214
- (100.0, 200.0\] chars: 7
- (200.0, inf\] chars: 0

______________________________________________________________________

## 5. Coarse Silence / Noise (VAD-based)

**Silence Ratio Distribution** (% non-speech frames):

- (-0.001, 0.1\]: 16
- (0.1, 0.2\]: 309
- (0.2, 0.4\]: 2,294
- (0.4, 0.6\]: 892
- (0.6, 1.0\]: 112

**Longest Silence Distribution** (seconds):

- (-0.001, 0.5\]: 503
- (0.5, 1.0\]: 1,726
- (1.0, 2.0\]: 1,338
- (2.0, 5.0\]: 56
- (5.0, inf\]: 0

**RMS dB Distribution**:

- (-inf, -60.0\]: 0
- (-60.0, -40.0\]: 0
- (-40.0, -20.0\]: 1,936
- (-20.0, -10.0\]: 1,678
- (-10.0, inf\]: 9

**Red Flags**:

- Files with silence_ratio > 0.4: *see inventory_files.csv for details*
- Files with longest_silence > 2.0s: *see inventory_files.csv for details*

______________________________________________________________________

## 6. Initial Conclusion

**Major Cleanup Required**: Yes

**Dominant Failure Modes**:

1. High very-short transcript rate (17.1%)

**Recommended Next Steps**:

1. Review `inventory_files.csv` to identify specific problematic files
1. Manually inspect sampled files in `inventory_samples.csv`
1. Design cleanup policy (S1-M1) based on failure modes

______________________________________________________________________

*End of Report*
