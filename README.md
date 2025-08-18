# SD Card Management Tool

A Python package for managing SD cards from action cameras, with specialized tools for BRUV (Baited Remote Underwater Video) deployments and telemetry analysis.

## Features

- **SD Card Management**: Import and organize video files from action cameras
- **BRUV Analysis**: Specialized tools for underwater camera deployments
- **Telemetry Processing**: Extract and analyze accelerometer/gyroscope data from GoPro videos
- **Impact Detection**: Detect seafloor impacts and camera events using machine learning
- **Video Synchronization**: Analyze time synchronization between paired cameras
- **Field Trip Integration**: Works with the [field_trip_bruv](https://github.com/NickMortimer/field_trip_bruv) template for organized project structure

## Installation

### Prerequisites

1. **Install Miniconda** (recommended over Anaconda for smaller footprint):
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Follow the installation instructions for your operating system

### Setup Environment

1. **Create a new conda environment** called `cardmanager`:
   ```bash
   conda create -n cardmanager python=3.11
   ```

2. **Activate the environment**:
   ```bash
   conda activate cardmanager
   ```

3. **Install the package from repository**:
   ```bash
   pip install git+https://github.com/NickMortimer/sdcard.git
   ```

   Or if you have cloned the repository locally:
   ```bash
   cd /path/to/sdcard
   pip install .
   ```

### Optional: Install Telemetry Support

For telemetry processing features, install the optional telemetry dependencies:

```bash
pip install telemetry-parser
```

### Create Field Trip Project

This tool is designed to work with the BRUV field trip template. After installing the package, create a new field trip project:

1. **Use cookiecutter to create a field trip project**:
   ```bash
   cookiecutter https://github.com/NickMortimer/field_trip_bruv
   ```

2. **Follow the prompts** to configure your field trip (trip name, dates, etc.)

3. **Move to your field trip directory**:
   ```bash
   cd your-field-trip-name
   ```

4. **All sdcard and bruv commands should be run from within this field trip directory**

### Verify Installation

Test that the tools are working from within your field trip directory:

```bash
# Test the main SD card tool
sdcard --help

# Test the BRUV analysis tool  
bruv --help
```

## Quick Start

**Important:** All commands below should be run from within your field trip directory created with the cookiecutter template.

### Basic SD Card Import
```bash
# Import videos from SD card (run from field trip directory)
sdcard import /path/to/sd/card /path/to/destination
```

### BRUV Analysis
Note: Videos need to be in a subdirectory called 100GOPRO or 100MEDIA
```bash
# Process BRUV deployment data (run from field trip directory)
bruv extract-hits /path/to/video/files
```

### Extract Telemetry
```bash
# Extract telemetry from GoPro videos (run from field trip directory)
bruv extract-telemetry /path/to/videos
```

## Workflow Overview

1. **Install sdcard package** (one time setup)
2. **Create field trip project** using cookiecutter template
3. **Navigate to field trip directory**
4. **Configure field trip settings** in the generated config files
5. **Run sdcard/bruv commands** from within the field trip directory
6. **Outputs and data** will be organized according to the field trip structure

## Development Setup

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/NickMortimer/sdcard.git
cd sdcard
conda create -n cardmanager-dev python=3.11
conda activate cardmanager-dev
pip install -e .
```

## Requirements

- Python 3.11+
- Typer 0.14.0+ (latest CLI framework with enhanced features)
- NumPy 2.0+
- Pandas 2.2+
- SciPy 1.13+
- OpenCV 4.10+
- Rich 13.7+ (enhanced terminal output, included with Typer[all])
- Click 8.1+ (CLI foundation, dependency of Typer)
- Cookiecutter 2.5+ (for field trip project templates)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
