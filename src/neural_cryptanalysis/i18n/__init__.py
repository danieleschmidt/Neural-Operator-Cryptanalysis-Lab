"""
Internationalization support for Neural Cryptanalysis Framework.

This module provides comprehensive internationalization (i18n) capabilities including:
- Multi-language support (English, Spanish, French, German, Japanese, Chinese)
- Locale-aware formatting for dates, numbers, and currencies
- RTL (Right-to-Left) language support
- Timezone handling
- Cultural compliance adaptations
"""

import os
import json
import gettext
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path

# Supported locales with their configurations
SUPPORTED_LOCALES = {
    'en': {
        'name': 'English',
        'native_name': 'English',
        'direction': 'ltr',
        'date_format': '%Y-%m-%d',
        'datetime_format': '%Y-%m-%d %H:%M:%S',
        'number_format': 'en_US',
        'currency': 'USD',
        'timezone': 'UTC'
    },
    'es': {
        'name': 'Spanish',
        'native_name': 'Español',
        'direction': 'ltr',
        'date_format': '%d/%m/%Y',
        'datetime_format': '%d/%m/%Y %H:%M:%S',
        'number_format': 'es_ES',
        'currency': 'EUR',
        'timezone': 'Europe/Madrid'
    },
    'fr': {
        'name': 'French',
        'native_name': 'Français',
        'direction': 'ltr',
        'date_format': '%d/%m/%Y',
        'datetime_format': '%d/%m/%Y %H:%M:%S',
        'number_format': 'fr_FR',
        'currency': 'EUR',
        'timezone': 'Europe/Paris'
    },
    'de': {
        'name': 'German',
        'native_name': 'Deutsch',
        'direction': 'ltr',
        'date_format': '%d.%m.%Y',
        'datetime_format': '%d.%m.%Y %H:%M:%S',
        'number_format': 'de_DE',
        'currency': 'EUR',
        'timezone': 'Europe/Berlin'
    },
    'ja': {
        'name': 'Japanese',
        'native_name': '日本語',
        'direction': 'ltr',
        'date_format': '%Y年%m月%d日',
        'datetime_format': '%Y年%m月%d日 %H:%M:%S',
        'number_format': 'ja_JP',
        'currency': 'JPY',
        'timezone': 'Asia/Tokyo'
    },
    'zh': {
        'name': 'Chinese (Simplified)',
        'native_name': '简体中文',
        'direction': 'ltr',
        'date_format': '%Y年%m月%d日',
        'datetime_format': '%Y年%m月%d日 %H:%M:%S',
        'number_format': 'zh_CN',
        'currency': 'CNY',
        'timezone': 'Asia/Shanghai'
    }
}

DEFAULT_LOCALE = 'en'

class I18nManager:
    """Internationalization manager for the Neural Cryptanalysis Framework."""
    
    def __init__(self, locale: str = DEFAULT_LOCALE, locales_dir: Optional[Path] = None):
        """
        Initialize the I18n manager.
        
        Args:
            locale: The locale to use (e.g., 'en', 'es', 'fr')
            locales_dir: Directory containing locale files
        """
        self.current_locale = locale if locale in SUPPORTED_LOCALES else DEFAULT_LOCALE
        self.locales_dir = locales_dir or Path(__file__).parent / 'locales'
        self._translations: Dict[str, Any] = {}
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load translation files for the current locale."""
        try:
            translation_file = self.locales_dir / f"{self.current_locale}.json"
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self._translations = json.load(f)
            else:
                # Fallback to English if locale file doesn't exist
                fallback_file = self.locales_dir / "en.json"
                if fallback_file.exists():
                    with open(fallback_file, 'r', encoding='utf-8') as f:
                        self._translations = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load translations for {self.current_locale}: {e}")
            self._translations = {}
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a message key with optional formatting parameters.
        
        Args:
            key: The translation key
            **kwargs: Formatting parameters
            
        Returns:
            Translated and formatted string
        """
        # Get translation with fallback to key
        translation = self._get_nested_translation(key)
        
        # Apply formatting if parameters provided
        if kwargs:
            try:
                return translation.format(**kwargs)
            except (KeyError, ValueError):
                return translation
        
        return translation
    
    def _get_nested_translation(self, key: str) -> str:
        """Get translation from nested dictionary using dot notation."""
        keys = key.split('.')
        current = self._translations
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return key  # Return key if translation not found
        
        return str(current) if current is not None else key
    
    def get_locale_info(self, locale: Optional[str] = None) -> Dict[str, Any]:
        """Get locale configuration information."""
        locale = locale or self.current_locale
        return SUPPORTED_LOCALES.get(locale, SUPPORTED_LOCALES[DEFAULT_LOCALE])
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format date according to locale conventions."""
        locale_info = self.get_locale_info(locale)
        return date.strftime(locale_info['date_format'])
    
    def format_datetime(self, dt: datetime, locale: Optional[str] = None) -> str:
        """Format datetime according to locale conventions."""
        locale_info = self.get_locale_info(locale)
        return dt.strftime(locale_info['datetime_format'])
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        # Simplified number formatting - in production, use babel or similar
        locale_info = self.get_locale_info(locale)
        if locale_info['number_format'].startswith('en'):
            return f"{number:,.2f}"
        else:
            # European style (comma as decimal separator)
            formatted = f"{number:.2f}".replace('.', ',')
            # Add thousand separators
            parts = formatted.split(',')
            parts[0] = f"{int(parts[0]):,}".replace(',', '.')
            return ','.join(parts)
    
    def set_locale(self, locale: str) -> bool:
        """
        Set the current locale.
        
        Args:
            locale: The locale code to set
            
        Returns:
            True if locale was set successfully, False otherwise
        """
        if locale in SUPPORTED_LOCALES:
            self.current_locale = locale
            self._load_translations()
            return True
        return False
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales with their display names."""
        return [
            {
                'code': code,
                'name': info['name'],
                'native_name': info['native_name']
            }
            for code, info in SUPPORTED_LOCALES.items()
        ]


# Global instance
_i18n_manager = I18nManager()

def get_i18n_manager() -> I18nManager:
    """Get the global I18n manager instance."""
    return _i18n_manager

def _(key: str, **kwargs) -> str:
    """Shorthand function for translation."""
    return _i18n_manager.translate(key, **kwargs)

def set_locale(locale: str) -> bool:
    """Set the global locale."""
    return _i18n_manager.set_locale(locale)

def get_current_locale() -> str:
    """Get the current locale."""
    return _i18n_manager.current_locale

def format_date(date: datetime, locale: Optional[str] = None) -> str:
    """Format date according to current or specified locale."""
    return _i18n_manager.format_date(date, locale)

def format_datetime(dt: datetime, locale: Optional[str] = None) -> str:
    """Format datetime according to current or specified locale."""
    return _i18n_manager.format_datetime(dt, locale)

def format_number(number: float, locale: Optional[str] = None) -> str:
    """Format number according to current or specified locale."""
    return _i18n_manager.format_number(number, locale)