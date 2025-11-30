# Strava Integration for Workout Analysis

This module extends the FIT file analyzer to support downloading and analyzing workouts directly from Strava, providing similar analysis capabilities to the FIT file analysis.

## Features

- Download workouts for any given date from Strava
- Comprehensive power analysis (NP, IF, TSS, VI) when power data is available
- Heart rate analysis and zone distribution
- Heart rate drift calculation
- Cadence, speed, and elevation analysis
- Support for both endurance and strength training workouts
- Compatible with existing analysis configuration files

## Setup

### 1. Install Dependencies

```bash
poetry install
```

### 2. Strava API Setup

1. Go to https://www.strava.com/settings/api
2. Create a new application if you don't have one
3. Note your Client ID and Client Secret
4. Generate an access token using Strava's OAuth flow or API explorer
5. **IMPORTANT**: Create the `.env` file in the project root directory (same level as `pyproject.toml`):

```bash
cp .env.example .env
# Edit .env and add your token
```

**File Location**: The `.env` file must be in your project root directory:
```
/your-project-root/
├── .env                 # ← Environment file goes here
├── pyproject.toml
├── src/
│   ├── strava_analyze.py
│   └── strava/
└── README_STRAVA.md
```

### 3. Configuration

The Strava analyzer uses the same configuration files as the FIT analyzer. Create a `settings.yaml` file or use an existing one:

```yaml
# Example settings.yaml
application:
  output-dir: "./output"

cycling:
  ftp: 250
  power-zones:
    - "Zone 1": "0-142"
    - "Zone 2": "143-189"
    - "Zone 3": "190-236"
    - "Zone 4": "237-284"
    - "Zone 5": "285-"

heart-rate:
  max: 190
  lt: 170
  hr-zones:
    - "Zone 1": "0-136"
    - "Zone 2": "137-153"
    - "Zone 3": "154-170"
    - "Zone 4": "171-187"
    - "Zone 5": "188-"

running:
  power-zones:
    - "Zone 1": "0-200"
    - "Zone 2": "201-250"
    - "Zone 3": "251-300"
    - "Zone 4": "301-350"
    - "Zone 5": "351-"
```

## Usage

### Basic Usage

Download and analyze all workouts for a specific date:

```bash
poetry run strava-analyze --date 2024-11-30 --settings settings.yaml
```

### Command Line Options

```bash
poetry run strava-analyze --help

options:
  --date DATE              Date to download workouts for (YYYY-MM-DD format) [required]
  --settings SETTINGS      Path to settings.yaml file
  --ftp FTP               Override FTP value (watts)
  --window WINDOW         Window length for Normalized Power calculation (default: 30s)
  --drift-start START     Start point for heart rate drift analysis (HH:MM:SS, MM:SS or SS)
  --drift-duration DURATION Duration for heart rate drift analysis (HH:MM:SS, MM:SS or SS)
  --autolap               Enable autolap analysis
```

### Examples

```bash
# Analyze workouts for November 30, 2024
poetry run strava-analyze --date 2024-11-30 --settings settings.yaml

# Override FTP for this analysis
poetry run strava-analyze --date 2024-11-30 --settings settings.yaml --ftp 280

# Analyze heart rate drift for the first 20 minutes
poetry run strava-analyze --date 2024-11-30 --settings settings.yaml --drift-start 0 --drift-duration 1200

# Use custom NP window (60 seconds instead of 30)
poetry run strava-analyze --date 2024-11-30 --settings settings.yaml --window 60
```

## Output

The analyzer creates markdown files with detailed analysis results, similar to the FIT file analyzer:

- **Session Information**: Basic workout details (name, sport, duration, distance)
- **Power Analysis**: NP, IF, TSS, VI, and basic statistics (when power data available)
- **Heart Rate Analysis**: Basic statistics and zone distribution
- **Cadence Analysis**: Basic statistics (when available)
- **Zone Analysis**: Time spent in configured power and heart rate zones
- **Heart Rate Drift**: Drift analysis for endurance workouts with power data

### Sample Output File

```
2024-11-30_strava_Morning_Ride-analysis.md
```

## Data Availability

Strava provides different levels of data depending on the workout:

### Always Available
- Basic workout metadata (name, sport, duration, distance)
- Summary statistics (average power, average heart rate, etc.)

### Stream Data (when available)
- Second-by-second power data
- Heart rate data
- Cadence data
- Speed and elevation data
- GPS coordinates

### Limitations

- **Strength Training**: Strava doesn't provide detailed set/rep data in streams like Garmin devices
- **Indoor Activities**: May have limited GPS/elevation data
- **Privacy Settings**: Some data may be unavailable due to athlete privacy settings
- **Third-party Uploads**: Data quality depends on original recording device

## Comparison with FIT File Analysis

| Feature | FIT Files | Strava |
|---------|-----------|--------|
| Power Analysis (NP, IF, TSS) | ✅ | ✅ |
| Heart Rate Analysis | ✅ | ✅ |
| Zone Distribution | ✅ | ✅ |
| Heart Rate Drift | ✅ | ✅ |
| Lap Analysis | ✅ | Limited* |
| Strength Set Analysis | ✅ | ❌ |
| Autolap Generation | ✅ | ❌ |
| High-Resolution Data | ✅ | Depends** |

\* Strava provides basic lap information but not detailed lap-by-lap streams  
\** Resolution depends on recording device and upload settings

## Troubleshooting

### Access Token Issues

```bash
ERROR: STRAVA_ACCESS_TOKEN environment variable not found.
```

**Solution Steps:**

1. **Check file location**: Ensure your `.env` file is in the project root directory (same level as `pyproject.toml`), not in a subdirectory
2. **Check file content**: Verify `.env` contains `STRAVA_ACCESS_TOKEN=your_token_here` (no spaces around the `=`)
3. **Verify token**: Ensure the token is valid and hasn't expired
4. **Check app permissions**: Verify your Strava app has the required scopes (`activity:read_all`)

**Quick debug:**
```bash
# Check if .env exists in the right place
ls -la .env

# Check if the token is set correctly
grep STRAVA_ACCESS_TOKEN .env
```

### No Data Available

```
No detailed stream data available from Strava for this activity.
```

This typically means:
- The activity was recorded without detailed streams
- Privacy settings prevent access to detailed data
- The activity was uploaded from a device that doesn't record streams

### Rate Limiting

Strava has API rate limits. If you encounter rate limiting errors:
- Wait before retrying
- Consider analyzing fewer activities at once
- Check Strava's current rate limit policies

## Development

The Strava integration consists of:

- [`src/strava/client.py`](src/strava/client.py): Strava API client and data parser
- [`src/strava_analyze.py`](src/strava_analyze.py): Main analysis script
- [`src/tools/calculations.py`](src/tools/calculations.py): Shared calculation functions

To extend the analysis or add features:

1. Modify the analysis functions in `strava_analyze.py`
2. Add new calculations to `tools/calculations.py` if they're reusable
3. Update the Strava client if you need additional data from the API

## License

Same as the main project.