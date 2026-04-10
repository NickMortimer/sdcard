from pathlib import Path
from typing import Optional
import jinja2
import typer
import yaml

DEFAULT_CARD_STORE_NAME = "card_store"
# Simple default import path pattern; overridable via config.yml
DEFAULT_IMPORT_TEMPLATE = "{{card_store}}/{import_date}/{import_token}"

class Config:
    def __init__(self, config_path: Optional[Path] = None):
        self.explicit_config_path = config_path is not None
        self.config_path = Path.cwd() / 'config.yml' if config_path is None else Path(config_path)
        self.catalog_dir = self.config_path.parent
        self.loaded_from_file = False
        self.data = self._load_config()

    @property
    def uses_implicit_defaults(self) -> bool:
        return not self.explicit_config_path and not self.loaded_from_file

    def _load_config(self):
        if not self.config_path.exists():
            if self.explicit_config_path:
                raise typer.BadParameter(
                    f"Config file not found: {self.config_path}",
                    param_hint="--config-path",
                )
            raw_data = {}
        else:
            if self.config_path.is_dir():
                raise typer.BadParameter(
                    f"Config path must be a file: {self.config_path}",
                    param_hint="--config-path",
                )
            self.loaded_from_file = True
            raw_data = yaml.safe_load(self.config_path.read_text(encoding='utf-8')) or {}

        # Ensure base paths exist even when no config file is present
        raw_data.setdefault('card_store', str(self.catalog_dir / DEFAULT_CARD_STORE_NAME))
        raw_data.setdefault('import_path_template', DEFAULT_IMPORT_TEMPLATE)
        raw_data['CATALOG_DIR'] = str(self.catalog_dir)

        return self._process_templates(raw_data)

    def _process_templates(self, config_data: dict) -> dict:
        """Process templates in multiple passes to handle nested references"""
        environment = jinja2.Environment()
        processed_data = config_data.copy()
        changed = True
        skip_keys = {"import_path_template"}
        while changed:
            changed = False
            for key, value in processed_data.items():
                if key in skip_keys:
                    continue
                if isinstance(value, str):
                    template = environment.from_string(value)
                    new_value = template.render(**processed_data)
                    if new_value != value:
                        processed_data[key] = new_value
                        changed = True
        
        return processed_data

    def get_path(self, key: str) -> Path:
        """Get a path from the config as a Path object"""
        return Path(self.data.get(key, ""))

    @property
    def settings(self):
        return self.data
if __name__ == "__main__":    
    cfg = Config()