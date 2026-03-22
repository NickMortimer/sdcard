# SD Card Management Tool

Minimal, instrument-agnostic SD card helper for probing, registering, and importing cards to a local store.

## Features

- Probe USB tree and show card/destination throughput estimates
- Register cards by writing a simple `import.yml`
- Import (copy/move) cards into a configurable `card_store`
- Turbo: fan out imports across multiple terminals on Linux

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

# Probe ports, cards, and destination speed
sdcard probe

# Register cards (creates import.yml on each card)
sdcard register --all

# Import cards (auto-discovers if none specified)
sdcard import

# Run concurrent imports on Linux
sdcard turbo
```

Configuration is read from `config.yml` in the current directory. Defaults:

- `card_store`: `./card_store`
- `import_path_template`: `{{card_store}}/{import_date}/{import_token}`

### Card import.yml (on each card)

Each SD card uses a small `import.yml` to track its ID and destination. The tool writes this file during `sdcard register`, but you can also create it manually:

```yaml
card_number: 1
import_token: abcd1234   # generated during registration (unique per registration)
import_date: 2024-01-31  # optional; set on first write
destination_path: /data/card_store/2024-01-31/abcd1234  # optional; fallback is the template above
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
import_template : "{{card_store}}/{instrument}/{import_date}/{card_number}_{import_token}
```
## Development

```bash
git clone https://github.com/NickMortimer/sdcard.git
cd sdcard
pip install -e .
```

## License

MIT
