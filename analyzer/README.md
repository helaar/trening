# FIT File Analyzer

A comprehensive tool for analyzing Garmin FIT files from cycling, running, and strength training workouts. Calculates advanced metrics, zone distributions, and generates detailed markdown reports.

## Features

### Power-Based Activities (Cycling & Running)
- **Advanced Power Metrics**: Normalized Power (NP), Intensity Factor (IF), Training Stress Score (TSS), Variability Index (VI)
- **Zone Analysis**: Configurable power and heart rate zone distributions
- **Heart Rate Drift**: Analyzes cardiac drift during steady-state efforts
- **Lap Analysis**: Processes existing laps or generates automatic intervals
- **Elevation**: Calculates ascent, descent, and elevation profile

### Strength Training
- **Set Analysis**: Processes individual strength training sets with heart rate data
- **Rep Counting**: Tracks repetitions and weights per set
- **Heart Rate Monitoring**: Average and peak heart rate per set

### General Features
- **Multi-Sport Support**: Cycling, running, and strength training
- **Configurable Settings**: FTP, max HR, and zone definitions via YAML
- **Markdown Reports**: Generates detailed analysis reports
- **Data Visualization**: Comprehensive statistics and zone breakdowns

## Installation

```bash
poetry install
```

## Configuration

Create a `setup.yaml` file with your personal settings:

```yaml
max-hr: 191
hr-zones:
  - Recovery: 0-141
  - Aerobic: 142-152
  - Tempo: 153-162
  - Sub Threshold: 163-170
  - Super Threshold: 171-177
  - Aerobic Capacity: 178-183
  - Anaerobic Capacity: 184-

cycling:
  ftp: 260
  power-zones:
    - Recovery: 0-143
    - Endurance: 144-195
    - Tempo: 196-234
    - Threshold: 235-273
    - VO2 Max: 274-312
    - Anaerobic Capacity: 313-390
    - Neuromuscular: 391-

running:
  ftp: 365
  power-zones:
    - Recovery: 0-255
    - Endurance: 255-290
    - Tempo: 290-330
    - Threshold: 330-385
    - VO2Max: 385-440
    - Anaerobic: 440-

autolap: PT10M
```

## Usage

### Command Line
```bash
# Basic analysis
poetry run fit-analyze --fitfile workout.fit --settings setup.yaml

# With custom FTP
poetry run fit-analyze --fitfile workout.fit --settings setup.yaml --ftp 280

# Analyze heart rate drift for specific segment
poetry run fit-analyze --fitfile workout.fit --settings setup.yaml --drift-start 5:00 --drift-duration 20:00

# Force autolap generation
poetry run fit-analyze --fitfile workout.fit --settings setup.yaml --autolap
```

### Output

The tool generates a detailed markdown report (`<filename>-analyse.md`) containing:

- **Workout Summary**: Sport, duration, distance, device info
- **Power Analysis**: NP, IF, TSS, VI, power statistics
- **Heart Rate Analysis**: HR zones, statistics, drift analysis
- **Lap Breakdown**: Detailed metrics per lap/interval
- **Zone Distribution**: Time spent in each power/HR zone

## Dependencies

- `pandas`: Data processing and time series analysis
- `numpy`: Numerical computations
- `pydantic`: Data validation and modeling
- `fitparse`: FIT file parsing
- `pyyaml`: Configuration file parsing

## File Structure

```
analyzer/
├── src/
│   ├── fit_analyzes.py      # Main analysis script
│   ├── format/
│   │   ├── garmin_fit.py    # FIT file parser
│   │   ├── models.py        # Data models
│   │   └── utils.py         # Utility functions
├── setup.yaml              # Personal configuration
├── plans.yaml              # Workout plans (optional)
└── pyproject.toml          # Poetry configuration
```

## License

Private project for personal fitness data analysis.