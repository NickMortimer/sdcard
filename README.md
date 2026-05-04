# SD Card Management Tool

Minimal, instrument-agnostic SD card helper for probing, registering, and importing cards to a local store.

## Features

- Probe USB tree and show card/destination throughput estimates
- Register cards by writing a simple `import.yml`
- Import (copy/move) cards into a configurable `card_store`
- Turbo: fan out imports across multiple terminals on Linux

## Utilities module map

The `sdcard.utils` package is split by responsibility:

- `cards_discovery.py`: SD card/USB discovery helpers (`list_sdcards`, `get_available_cards`, `get_usb_info`)
- `import_conflicts.py`: destination comparison, conflict reporting, and overwrite safety prompts
- `import_metadata.py`: import metadata rendering/writing (`import.yml`, destination `README.md`, register helpers)
- `import_transfer.py`: copy/move import orchestration and transfer execution
- `usb.py`: USB topology and throughput analysis used by `probe` and `turbo`

## Installation

```bash
pip install git+https://github.com/NickMortimer/sdcard.git
# or from a local clone
pip install .
```

## Usage

```bash
# List available cards
sdcard list

# List cards with reader manufacturer and speed details
sdcard list --verbose

# Probe ports, cards, and destination speed
sdcard probe

# Register cards (creates import.yml on each card)
sdcard register --all

# Register cards and prompt for optional manufacturer/UHS metadata
sdcard register --all --card-details

# Refresh an existing import.yml using config defaults while keeping the
# chosen instrument and card number, and generating a new token.
sdcard register /path/to/card --config-path /path/to/import.yml --refresh

# Import cards (auto-discovers if none specified)
sdcard import

# Extract EXIF metadata into exif.json.zst files per image directory
sdcard xif /path/to/head-directory

# Use card_store from config.yml in current directory
sdcard xif --card-store

# Use card_store from a specific config file
sdcard xif --card-store --config-path /path/to/config.yml

# Use xif_path from a specific config file (no --card-store needed)
sdcard xif --config-path /path/to/config.yml

# Restrict EXIF extraction to a specific file extension
sdcard xif /path/to/head-directory --ext ARW

# Create thumbnail files in a sibling thumbnails/ directory
sdcard thumbnail /path/to/head-directory

# Restrict thumbnail generation to a specific file extension
sdcard thumbnail /path/to/head-directory --ext ARW

# Mirror the source tree into a separate output root
sdcard thumbnail /path/to/head-directory --output-dir-name /path/to/thumb-root

# Also copy extracted metadata into generated thumbnail files
sdcard thumbnail /path/to/head-directory --copy-meta

# Control exiftool batch size per call
sdcard xif /path/to/head-directory --batch-size 100

# Control threadpool workers for per-directory extraction
sdcard xif /path/to/head-directory --workers 8

# Run concurrent imports on Linux
sdcard turbo

# Download Windows helper binaries into {CATALOG_DIR}/bin
sdcard getbins --config-path /path/to/config.yml

# Run reusable workflow steps from workflows.yml
sdcard workflow --list
sdcard workflow --name quick_xif
sdcard workflow --name setup_tools --config-path /path/to/config.yml

# Open an interactive workflow menu and keep selecting until exit
sdcard workflow --file /path/to/workflows.yml
```

Example workflows file (`workflows.yml`):

```yaml
workflows:
    quick_xif:
        description: Extract EXIF using cached/default config
        commands:
            - xif --config-path {config_path}

    setup_tools:
        description: Ensure local helper binaries exist
        commands:
            - getbins --config-path {config_path}

    import_then_probe:
        description: Import cards then inspect topology
        commands:
            - import
            - probe
```

When registration writes an `import.yml`, it preserves the destination template string from the supplied config, prompts for any top-level config value that offers multiple choices, and then prints the instrument plus destination written for each card.

`sdcard register --refresh` rebuilds the card's `import.yml` from the supplied config, keeps the existing card number and selected instrument, generates a new `import_token`, and applies current config defaults for the remaining fields.

The `xif` command walks the directory tree below the path you give it. For each
directory containing supported image files, it writes an `exif.json.zst` file in
that directory. If that JSON file already exists, the directory is skipped.
Use `--card-store` to resolve the scan root from `card_store` in config instead
of passing a directory. Pair it with `--config-path` to point at a non-default
config file.
When `--config-path` is provided without a directory, `xif` uses `xif_path`
from that config file.
It uses `exiftool`, so ensure `exiftool` is installed and available in `PATH`.
Use `--batch-size` to tune how many images are sent per `exiftool` call.
Use `--workers` to control the threadpool size across directories.
Use `--ext` one or more times to limit processing to specific suffixes such as
`ARW` or `CR3`.

The `thumbnail` command is a second-stage workflow. It walks the same directory
tree, reads the previously generated `exif.json.zst` files, and creates thumbnail
files in a sibling `thumbnails/` directory without re-reading EXIF metadata from
the source files. Use `--output-dir-name` with a simple name to change that
sibling directory name, or pass a path such as `/path/to/thumb-root` to mirror
the source directory structure under a separate thumbnail root.
When `--copy-meta` is enabled, it writes extracted metadata into the generated
thumbnail files by importing the extracted JSON through `exiftool`.
Use `--ext` to match the same suffix filter you used during `xif` extraction.

During import, each destination folder also gets a `README.md` summarizing the project and custodian metadata from the card's `import.yml`.

You can customize that README with either of these config keys:

- `destination_readme_template`: inline Jinja template text
- `destination_readme_template_path`: path to a Jinja template file, relative to the config file directory unless absolute

Example:

```yaml
destination_readme_template: |
	# {{ project_name }}

	Custodian: {{ custodian }}
	{% if email %}Contact: {{ email }}{% endif %}
	{% if field_trip_id %}Field trip: {{ field_trip_id }}{% endif %}
	Card: {{ card_number }}
	Token: {{ import_token }}
```

Configuration is read from `config.yml` in the current directory. Defaults:

- `card_store`: `./card_store`
- `import_path_template`: `{{card_store}}/{import_date}/{import_token}`

When you pass a config file to `sdcard register --config-path ...`, all top-level key/value pairs from that config file are copied into each card's `import.yml`. If a top-level value is a list or mapping of choices, registration prompts you to select one value for that card. Card-specific fields like `card_number`, `import_token`, `destination_path`, `card_size_gb`, and `card_format` still take precedence.

### Card import.yml (on each card)

Each SD card uses a small `import.yml` to track its ID and destination. The tool writes this file during `sdcard register`, but you can also create it manually:

```yaml
card_number: 1
import_token: abcd1234   # generated during registration (unique per registration)
import_date: 2024-01-31  # optional; set on first write
destination_path: /data/card_store/2024-01-31/abcd1234  # optional; fallback is the template above
partition_label: SDXC_128GB
card_size_gb: 119.24
card_format: exfat
card_manufacturer: SanDisk  # optional; prompted with --card-details
rated_uhs: U3               # optional; prompted with --card-details
max_read_speed_mb_s: 200.0  # optional; prompted with --card-details
max_write_speed_mb_s: 90.0  # optional; prompted with --card-details
```

Only `card_number` and a token are required; `destination_path` can be omitted to use the template.

Notes:
- `register_date` is set when the card is registered.
- `import_token` is generated at registration and reused on import.
- `import_date` is set the first time the card is actually copied/moved.
- The stable `import_token` lets you retry large batch imports (e.g., 30+ cards) without changing destinations if a run fails.
 
a full yaml file allows for useful metadata
```yaml
field_trip_id : DR2024-02
start_date : 2024-05-13
end_date : 2024-05-25
custodian : Nikki Minnow
email : nikki.minnow@....
project_name : Outlook
instrument : gopro_bruv
card_number : 44
import_date: <- set by  first import
import_token : <-set by registration
import_template : "{{card_store}}/{instrument}/{import_date}/{card_number}_{import_token}"
```
## Development

```bash
git clone https://github.com/NickMortimer/sdcard.git
cd sdcard
pip install -e .
```

## License

MIT

## SSH Operation and Permissions

If you want to use `sdcard` to mount and eject SD cards over SSH without sudo, you must allow your user to run `udisksctl` actions via polkit. Add the following rule (as root) to `/etc/polkit-1/rules.d/49-udisks2-mount-eject.rules`:

```javascript
polkit.addRule(function(action, subject) {
    if (
        (action.id == "org.freedesktop.udisks2.filesystem-mount" ||
         action.id == "org.freedesktop.udisks2.filesystem-unmount-others" ||
         action.id == "org.freedesktop.udisks2.eject-media" ||
         action.id == "org.freedesktop.udisks2.power-off-drive") &&
        subject.isInGroup("plugdev")
    ) {
        return polkit.Result.YES;
    }
});
```

- Make sure your user is in the `plugdev` group (or change the group as needed).
- Restart polkit or reboot for the rule to take effect.
- This allows mounting, unmounting, ejecting, and powering off drives without sudo for users in the specified group—even over SSH.

To add your user to the `plugdev` group (required for the rule above), run:

```bash
sudo usermod -aG plugdev $USER
```

Then log out and log back in (or restart your SSH session) for the group change to take effect.

If your system does not already have a `plugdev` group, create it first:

```bash
sudo groupadd plugdev
```
