# TIFF to PNG Conversion Tool

This directory contains utility tools for the YOLO-SAM inference pipeline.

## tiff2png.py

A script to convert TIFF images to PNG format. It supports recursive directory traversal to process nested folders and includes progress tracking with tqdm.

### Requirements

- Python 3.6+
- Pillow (PIL Fork)
- tqdm (for progress bars)

You can install the required packages with:

```bash
pip install pillow tqdm
```

### Features

- Convert TIFF images (.tif, .tiff, .TIF, .TIFF) to PNG format
- Process directories recursively with the `--recursive` flag
- Specify custom output directory
- Maintain directory structure when using output directory
- Progress tracking with tqdm progress bars
- Detailed logging with the `--verbose` flag

### Usage

Basic usage:

```bash
python tiff2png.py /path/to/tiff/directory
```

With recursive flag to process subdirectories:

```bash
python tiff2png.py /path/to/tiff/directory --recursive
```

Specify an output directory:

```bash
python tiff2png.py /path/to/tiff/directory --output /path/to/output/directory
```

Enable verbose output:

```bash
python tiff2png.py /path/to/tiff/directory --verbose
```

### Command-line Arguments

- `directory`: Directory containing TIFF images (required)
- `--recursive`, `-r`: Process subdirectories recursively
- `--output`, `-o`: Output directory for PNG images
- `--verbose`, `-v`: Enable verbose output

### Examples

Convert all TIFF images in the current directory:

```bash
python tiff2png.py .
```

Convert all TIFF images in a directory and its subdirectories:

```bash
python tiff2png.py /data/images --recursive
```

Convert all TIFF images and save them to a specific output directory:

```bash
python tiff2png.py /data/images --output /data/converted
```

When using the `--output` option with `--recursive`, the script will maintain the same directory structure in the output directory.

## PostgreSQL Data Creation Tool

The `postgres_data_create.py` script connects to the MinIO Tracking PostgreSQL database and creates purpose-specific tables to store selected objects to be ingested into the yolo_sam_inference pipeline.

### Purpose

- Creates purpose-specific tables for different experiments/processes
- Allows collecting data from multiple sources into a single table
- Connects to an existing PostgreSQL database containing MinIO object references
- Searches for objects matching a specified partial path
- Creates a target database and table to store the selected objects
- Tracks the processing status of each object

### Commands

The tool now supports multiple commands for managing purpose-specific tables:

```bash
# Create a new purpose table
python -m yolo_sam_inference.tools.postgres_data_create create --table "experiment_x_data" --template "standard"

# Add data to a purpose table
python -m yolo_sam_inference.tools.postgres_data_create add --path "bead/20250226/12%_13um/" --table "experiment_x_data" --purpose "validation"

# List all purpose tables
python -m yolo_sam_inference.tools.postgres_data_create list

# Show summary information for a purpose table
python -m yolo_sam_inference.tools.postgres_data_create summary --table "experiment_x_data"
```

#### Create Command

Creates a new purpose-specific table using a specified template.

```bash
python -m yolo_sam_inference.tools.postgres_data_create create --table "experiment_x_data" --template "standard"
```

Parameters:
- `--table`: (Required) Name of the purpose table to create
- `--template`: (Optional) Template type to use. Options: "standard", "experiment", or "time_series". Default is "standard".

#### Add Command

Adds data from a specific path to a purpose table.

```bash
python -m yolo_sam_inference.tools.postgres_data_create add --path "bead/20250226/12%_13um/" --table "experiment_x_data" --purpose "validation"
```

Parameters:
- `--path`: (Required) Partial MinIO path to search for objects
- `--table`: (Required) Name of the purpose table to add data to
- `--purpose`: (Required) Purpose identifier for this data set (e.g., "training", "validation", "testing")
- `--description`: (Optional) Description of this data set
- `--batch-id`: (Optional) Custom batch identifier. If not provided, a timestamp-based ID is generated.
- `--template`: (Optional) Template type of the table. Options: "standard", "experiment", or "time_series". Default is "standard".

#### List Command

Lists all available purpose tables.

```bash
python -m yolo_sam_inference.tools.postgres_data_create list
```

#### Summary Command

Shows summary information for a specific purpose table.

```bash
python -m yolo_sam_inference.tools.postgres_data_create summary --table "experiment_x_data"
```

Parameters:
- `--table`: (Required) Name of the purpose table to summarize

### Table Templates

#### Standard Template (`standard`)

Basic template for general purpose data collection:

- Standard MinIO object fields (path, size, timestamps, etc.)
- Purpose identifier for categorizing objects
- Optional description field

#### Experiment Template (`experiment`)

Extended template with additional fields for experimental metadata:

- All standard template fields
- Experiment-specific fields (experiment_id, sample_type, magnification)
- Automatically extracts experiment_id and sample_type from the path

#### Time Series Template (`time_series`)

Extended template with additional fields for time-series data:

- All standard template fields
- Time series specific fields (time_point, channel, sequence_id) 
- Automatically extracts time_point from the filename

### Environment Variables

The tool uses the following environment variables for database connection:

**Source Database (MinIO Tracking PostgreSQL):**
- `POSTGRES_HOST`: Host of the PostgreSQL server (default: "postgres")
- `POSTGRES_DB`: Database name (default: "mlflowdb")
- `POSTGRES_USER`: Username (default: "user")
- `POSTGRES_PASSWORD`: Password (default: "password")
- `POSTGRES_PORT`: Port (default: "5432")

**Target Database (YOLO-SAM Inference database):**
- `TARGET_POSTGRES_HOST`: Host (defaults to source host if not specified)
- `TARGET_POSTGRES_DB`: Database name (default: "yolo_sam_inference")
- `TARGET_POSTGRES_USER`: Username (defaults to source user if not specified)
- `TARGET_POSTGRES_PASSWORD`: Password (defaults to source password if not specified)
- `TARGET_POSTGRES_PORT`: Port (defaults to source port if not specified)

### Example Workflow

Here's an example of how to create and populate a purpose-specific table for "experiment_x":

```bash
# Create a new purpose table for experiment_x
python -m yolo_sam_inference.tools.postgres_data_create create --table "experiment_x_data" --template "experiment"

# Add training data from multiple sources
python -m yolo_sam_inference.tools.postgres_data_create add \
  --path "bead/20250226/12%_13um/" \
  --table "experiment_x_data" \
  --purpose "training" \
  --description "Bead dataset for training experiment_x" \
  --template "experiment"

python -m yolo_sam_inference.tools.postgres_data_create add \
  --path "cell/20250226/brightfield/" \
  --table "experiment_x_data" \
  --purpose "training" \
  --description "Cell dataset for training experiment_x" \
  --template "experiment"

# Add validation data
python -m yolo_sam_inference.tools.postgres_data_create add \
  --path "bead/20250227/validation/" \
  --table "experiment_x_data" \
  --purpose "validation" \
  --template "experiment"

# Check the summary of your experiment table
python -m yolo_sam_inference.tools.postgres_data_create summary --table "experiment_x_data"
```

### Notes

- Only `.tiff` files are selected from the source database.
- The tool automatically creates the target database and table if they don't exist.
- Duplicate entries are handled with an upsert operation (INSERT ON CONFLICT UPDATE).
- The source information is already contained in the minio_path field.
- Each table tracks the purpose and description for each object.
- You can add data from multiple paths to the same table.
- The same object can be in multiple purpose tables for different experiments.
- Each purpose table functions as a self-contained collection of data for a specific experiment or process.
- Indexes are automatically created for efficient querying. 