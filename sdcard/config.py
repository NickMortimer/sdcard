from pathlib import Path
from typing import Optional
import jinja2
import yaml

DEFAULT_CARD_STORE_NAME = "card_store"
# Simple default import path pattern; overridable via config.yml
DEFAULT_IMPORT_TEMPLATE = "{{card_store}}/{import_date}/{import_token}"

class Config:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path =  Path.cwd() / 'config.yml' if config_path is None else Path(config_path)
        self.catalog_dir = self.config_path.parent
        self.data = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raw_data = {}
        else:
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