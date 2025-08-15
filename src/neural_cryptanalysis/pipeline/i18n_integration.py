"""Internationalization Integration for Self-Healing Pipeline.

This module provides i18n support for the self-healing pipeline system,
enabling multi-language support for alerts, status messages, and user interfaces.

Supports:
- Dynamic language switching
- Regional compliance messaging
- Localized alert formatting
- Multi-timezone handling
- Currency and number formatting
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import re

# Define supported locales
class SupportedLocale(Enum):
    """Supported locales for the system."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    FR_FR = "fr_FR"  # French (France)
    FR_CA = "fr_CA"  # French (Canada)
    DE_DE = "de_DE"  # German (Germany)
    IT_IT = "it_IT"  # Italian (Italy)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    KO_KR = "ko_KR"  # Korean (South Korea)
    RU_RU = "ru_RU"  # Russian (Russia)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)


class ComplianceRegion(Enum):
    """Compliance regions for data protection."""
    EU = "eu"        # European Union (GDPR)
    US = "us"        # United States (CCPA, etc.)
    CA = "ca"        # Canada (PIPEDA)
    UK = "uk"        # United Kingdom (UK GDPR)
    JP = "jp"        # Japan (APPI)
    SG = "sg"        # Singapore (PDPA)
    AU = "au"        # Australia (Privacy Act)
    BR = "br"        # Brazil (LGPD)
    GLOBAL = "global" # Global/International


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    locale: SupportedLocale
    language: str
    country: str
    timezone: str
    currency: str
    date_format: str
    time_format: str
    number_format: str
    compliance_region: ComplianceRegion
    rtl: bool = False  # Right-to-left text direction


class PipelineI18nManager:
    """Internationalization manager for self-healing pipeline."""
    
    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations: Dict[str, Dict] = {}
        self.locale_configs: Dict[SupportedLocale, LocaleConfig] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize locale configurations
        self._initialize_locale_configs()
        
        # Load translations
        self._load_translations()
    
    def _initialize_locale_configs(self) -> None:
        """Initialize locale configurations."""
        self.locale_configs = {
            SupportedLocale.EN_US: LocaleConfig(
                locale=SupportedLocale.EN_US,
                language="English",
                country="United States",
                timezone="America/New_York",
                currency="USD",
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                number_format="1,234.56",
                compliance_region=ComplianceRegion.US
            ),
            SupportedLocale.EN_GB: LocaleConfig(
                locale=SupportedLocale.EN_GB,
                language="English",
                country="United Kingdom",
                timezone="Europe/London",
                currency="GBP",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1,234.56",
                compliance_region=ComplianceRegion.UK
            ),
            SupportedLocale.ES_ES: LocaleConfig(
                locale=SupportedLocale.ES_ES,
                language="Español",
                country="España",
                timezone="Europe/Madrid",
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                compliance_region=ComplianceRegion.EU
            ),
            SupportedLocale.FR_FR: LocaleConfig(
                locale=SupportedLocale.FR_FR,
                language="Français",
                country="France",
                timezone="Europe/Paris",
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1 234,56",
                compliance_region=ComplianceRegion.EU
            ),
            SupportedLocale.DE_DE: LocaleConfig(
                locale=SupportedLocale.DE_DE,
                language="Deutsch",
                country="Deutschland",
                timezone="Europe/Berlin",
                currency="EUR",
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                compliance_region=ComplianceRegion.EU
            ),
            SupportedLocale.JA_JP: LocaleConfig(
                locale=SupportedLocale.JA_JP,
                language="日本語",
                country="日本",
                timezone="Asia/Tokyo",
                currency="JPY",
                date_format="%Y/%m/%d",
                time_format="%H:%M",
                number_format="1,234",
                compliance_region=ComplianceRegion.JP
            ),
            SupportedLocale.ZH_CN: LocaleConfig(
                locale=SupportedLocale.ZH_CN,
                language="简体中文",
                country="中国",
                timezone="Asia/Shanghai",
                currency="CNY",
                date_format="%Y-%m-%d",
                time_format="%H:%M",
                number_format="1,234.56",
                compliance_region=ComplianceRegion.GLOBAL
            ),
            SupportedLocale.AR_SA: LocaleConfig(
                locale=SupportedLocale.AR_SA,
                language="العربية",
                country="المملكة العربية السعودية",
                timezone="Asia/Riyadh",
                currency="SAR",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1,234.56",
                compliance_region=ComplianceRegion.GLOBAL,
                rtl=True
            )
        }
    
    def _load_translations(self) -> None:
        """Load translation files."""
        # Define default translations for pipeline-specific messages
        default_translations = {
            "en_US": {
                "pipeline": {
                    "status": {
                        "healthy": "System is operating normally",
                        "warning": "System performance degraded",
                        "critical": "System requires immediate attention",
                        "failed": "System has failed",
                        "recovering": "System is recovering"
                    },
                    "alerts": {
                        "high_cpu": "High CPU utilization detected: {usage}%",
                        "high_memory": "High memory usage detected: {usage}%",
                        "high_error_rate": "Elevated error rate detected: {rate}%",
                        "slow_response": "Slow response times detected: {time}ms",
                        "recovery_action": "Recovery action '{action}' executed",
                        "recovery_success": "Recovery action successful",
                        "recovery_failed": "Recovery action failed: {reason}"
                    },
                    "actions": {
                        "restart_service": "Restart Service",
                        "scale_resources": "Scale Resources",
                        "clear_cache": "Clear Cache",
                        "optimize_algorithms": "Optimize Algorithms",
                        "circuit_breaker": "Activate Circuit Breaker",
                        "graceful_degradation": "Enable Graceful Degradation"
                    },
                    "monitoring": {
                        "system_health": "System Health",
                        "performance_metrics": "Performance Metrics",
                        "resource_usage": "Resource Usage",
                        "error_rate": "Error Rate",
                        "response_time": "Response Time",
                        "uptime": "Uptime: {hours} hours",
                        "last_incident": "Last incident: {time}",
                        "alerts_active": "{count} active alerts"
                    },
                    "compliance": {
                        "data_retention": "Data retention: {days} days",
                        "privacy_notice": "This system processes operational data in accordance with privacy regulations",
                        "audit_log": "All actions are logged for security auditing",
                        "secure_processing": "Data is processed using secure, encrypted channels"
                    }
                }
            },
            "es_ES": {
                "pipeline": {
                    "status": {
                        "healthy": "El sistema funciona normalmente",
                        "warning": "Rendimiento del sistema degradado",
                        "critical": "El sistema requiere atención inmediata",
                        "failed": "El sistema ha fallado",
                        "recovering": "El sistema se está recuperando"
                    },
                    "alerts": {
                        "high_cpu": "Alta utilización de CPU detectada: {usage}%",
                        "high_memory": "Alto uso de memoria detectado: {usage}%",
                        "high_error_rate": "Tasa de error elevada detectada: {rate}%",
                        "slow_response": "Tiempos de respuesta lentos detectados: {time}ms",
                        "recovery_action": "Acción de recuperación '{action}' ejecutada",
                        "recovery_success": "Acción de recuperación exitosa",
                        "recovery_failed": "Acción de recuperación falló: {reason}"
                    },
                    "actions": {
                        "restart_service": "Reiniciar Servicio",
                        "scale_resources": "Escalar Recursos",
                        "clear_cache": "Limpiar Caché",
                        "optimize_algorithms": "Optimizar Algoritmos",
                        "circuit_breaker": "Activar Cortacircuitos",
                        "graceful_degradation": "Habilitar Degradación Gradual"
                    },
                    "monitoring": {
                        "system_health": "Salud del Sistema",
                        "performance_metrics": "Métricas de Rendimiento",
                        "resource_usage": "Uso de Recursos",
                        "error_rate": "Tasa de Error",
                        "response_time": "Tiempo de Respuesta",
                        "uptime": "Tiempo activo: {hours} horas",
                        "last_incident": "Último incidente: {time}",
                        "alerts_active": "{count} alertas activas"
                    },
                    "compliance": {
                        "data_retention": "Retención de datos: {days} días",
                        "privacy_notice": "Este sistema procesa datos operacionales de acuerdo con las regulaciones de privacidad",
                        "audit_log": "Todas las acciones se registran para auditoría de seguridad",
                        "secure_processing": "Los datos se procesan usando canales seguros y encriptados"
                    }
                }
            },
            "fr_FR": {
                "pipeline": {
                    "status": {
                        "healthy": "Le système fonctionne normalement",
                        "warning": "Performance du système dégradée",
                        "critical": "Le système nécessite une attention immédiate",
                        "failed": "Le système a échoué",
                        "recovering": "Le système se rétablit"
                    },
                    "alerts": {
                        "high_cpu": "Utilisation CPU élevée détectée: {usage}%",
                        "high_memory": "Utilisation mémoire élevée détectée: {usage}%",
                        "high_error_rate": "Taux d'erreur élevé détecté: {rate}%",
                        "slow_response": "Temps de réponse lents détectés: {time}ms",
                        "recovery_action": "Action de récupération '{action}' exécutée",
                        "recovery_success": "Action de récupération réussie",
                        "recovery_failed": "Action de récupération échouée: {reason}"
                    },
                    "actions": {
                        "restart_service": "Redémarrer le Service",
                        "scale_resources": "Dimensionner les Ressources",
                        "clear_cache": "Vider le Cache",
                        "optimize_algorithms": "Optimiser les Algorithmes",
                        "circuit_breaker": "Activer le Disjoncteur",
                        "graceful_degradation": "Activer la Dégradation Graduelle"
                    },
                    "monitoring": {
                        "system_health": "Santé du Système",
                        "performance_metrics": "Métriques de Performance",
                        "resource_usage": "Utilisation des Ressources",
                        "error_rate": "Taux d'Erreur",
                        "response_time": "Temps de Réponse",
                        "uptime": "Temps de fonctionnement: {hours} heures",
                        "last_incident": "Dernier incident: {time}",
                        "alerts_active": "{count} alertes actives"
                    },
                    "compliance": {
                        "data_retention": "Rétention des données: {days} jours",
                        "privacy_notice": "Ce système traite les données opérationnelles conformément aux réglementations sur la confidentialité",
                        "audit_log": "Toutes les actions sont enregistrées pour l'audit de sécurité",
                        "secure_processing": "Les données sont traitées via des canaux sécurisés et chiffrés"
                    }
                }
            },
            "de_DE": {
                "pipeline": {
                    "status": {
                        "healthy": "System läuft normal",
                        "warning": "Systemleistung beeinträchtigt",
                        "critical": "System erfordert sofortige Aufmerksamkeit",
                        "failed": "System ist ausgefallen",
                        "recovering": "System erholt sich"
                    },
                    "alerts": {
                        "high_cpu": "Hohe CPU-Auslastung erkannt: {usage}%",
                        "high_memory": "Hohe Speichernutzung erkannt: {usage}%",
                        "high_error_rate": "Erhöhte Fehlerrate erkannt: {rate}%",
                        "slow_response": "Langsame Antwortzeiten erkannt: {time}ms",
                        "recovery_action": "Wiederherstellungsaktion '{action}' ausgeführt",
                        "recovery_success": "Wiederherstellungsaktion erfolgreich",
                        "recovery_failed": "Wiederherstellungsaktion fehlgeschlagen: {reason}"
                    },
                    "actions": {
                        "restart_service": "Service Neustarten",
                        "scale_resources": "Ressourcen Skalieren",
                        "clear_cache": "Cache Leeren",
                        "optimize_algorithms": "Algorithmen Optimieren",
                        "circuit_breaker": "Schutzschalter Aktivieren",
                        "graceful_degradation": "Schrittweise Verschlechterung Aktivieren"
                    },
                    "monitoring": {
                        "system_health": "Systemzustand",
                        "performance_metrics": "Leistungsmetriken",
                        "resource_usage": "Ressourcennutzung",
                        "error_rate": "Fehlerrate",
                        "response_time": "Antwortzeit",
                        "uptime": "Betriebszeit: {hours} Stunden",
                        "last_incident": "Letzter Vorfall: {time}",
                        "alerts_active": "{count} aktive Warnmeldungen"
                    },
                    "compliance": {
                        "data_retention": "Datenspeicherung: {days} Tage",
                        "privacy_notice": "Dieses System verarbeitet Betriebsdaten gemäß Datenschutzbestimmungen",
                        "audit_log": "Alle Aktionen werden für Sicherheitsaudits protokolliert",
                        "secure_processing": "Daten werden über sichere, verschlüsselte Kanäle verarbeitet"
                    }
                }
            },
            "ja_JP": {
                "pipeline": {
                    "status": {
                        "healthy": "システムは正常に動作しています",
                        "warning": "システムパフォーマンスが低下しています",
                        "critical": "システムに緊急の対応が必要です",
                        "failed": "システムが故障しました",
                        "recovering": "システムが回復中です"
                    },
                    "alerts": {
                        "high_cpu": "高いCPU使用率を検出: {usage}%",
                        "high_memory": "高いメモリ使用量を検出: {usage}%",
                        "high_error_rate": "高いエラー率を検出: {rate}%",
                        "slow_response": "遅いレスポンス時間を検出: {time}ms",
                        "recovery_action": "復旧アクション'{action}'を実行しました",
                        "recovery_success": "復旧アクションが成功しました",
                        "recovery_failed": "復旧アクションが失敗しました: {reason}"
                    },
                    "actions": {
                        "restart_service": "サービス再起動",
                        "scale_resources": "リソーススケール",
                        "clear_cache": "キャッシュクリア",
                        "optimize_algorithms": "アルゴリズム最適化",
                        "circuit_breaker": "サーキットブレーカー作動",
                        "graceful_degradation": "段階的縮退を有効化"
                    },
                    "monitoring": {
                        "system_health": "システムヘルス",
                        "performance_metrics": "パフォーマンス指標",
                        "resource_usage": "リソース使用量",
                        "error_rate": "エラー率",
                        "response_time": "レスポンス時間",
                        "uptime": "稼働時間: {hours}時間",
                        "last_incident": "最後のインシデント: {time}",
                        "alerts_active": "{count}件のアクティブアラート"
                    },
                    "compliance": {
                        "data_retention": "データ保持: {days}日",
                        "privacy_notice": "このシステムはプライバシー規制に従って運用データを処理します",
                        "audit_log": "すべてのアクションはセキュリティ監査のためにログに記録されます",
                        "secure_processing": "データは安全で暗号化されたチャネルを使用して処理されます"
                    }
                }
            },
            "zh_CN": {
                "pipeline": {
                    "status": {
                        "healthy": "系统运行正常",
                        "warning": "系统性能下降",
                        "critical": "系统需要立即关注",
                        "failed": "系统已故障",
                        "recovering": "系统正在恢复"
                    },
                    "alerts": {
                        "high_cpu": "检测到高CPU使用率: {usage}%",
                        "high_memory": "检测到高内存使用率: {usage}%",
                        "high_error_rate": "检测到高错误率: {rate}%",
                        "slow_response": "检测到响应时间过慢: {time}ms",
                        "recovery_action": "已执行恢复操作'{action}'",
                        "recovery_success": "恢复操作成功",
                        "recovery_failed": "恢复操作失败: {reason}"
                    },
                    "actions": {
                        "restart_service": "重启服务",
                        "scale_resources": "扩展资源",
                        "clear_cache": "清理缓存",
                        "optimize_algorithms": "优化算法",
                        "circuit_breaker": "激活断路器",
                        "graceful_degradation": "启用优雅降级"
                    },
                    "monitoring": {
                        "system_health": "系统健康",
                        "performance_metrics": "性能指标",
                        "resource_usage": "资源使用",
                        "error_rate": "错误率",
                        "response_time": "响应时间",
                        "uptime": "运行时间: {hours}小时",
                        "last_incident": "最后事件: {time}",
                        "alerts_active": "{count}个活跃警报"
                    },
                    "compliance": {
                        "data_retention": "数据保留: {days}天",
                        "privacy_notice": "此系统根据隐私法规处理运营数据",
                        "audit_log": "所有操作都会被记录用于安全审计",
                        "secure_processing": "数据通过安全加密通道处理"
                    }
                }
            }
        }
        
        # Store translations
        for locale, translations in default_translations.items():
            self.translations[locale] = translations
        
        # Try to load additional translations from files
        self._load_external_translations()
    
    def _load_external_translations(self) -> None:
        """Load translations from external files if available."""
        try:
            # Check for translations in the i18n directory
            i18n_dir = Path(__file__).parent.parent / "i18n" / "locales"
            if i18n_dir.exists():
                for locale_file in i18n_dir.glob("*.json"):
                    locale_code = locale_file.stem
                    try:
                        with open(locale_file, 'r', encoding='utf-8') as f:
                            external_translations = json.load(f)
                            
                        # Merge with existing translations
                        if locale_code in self.translations:
                            self._merge_translations(self.translations[locale_code], external_translations)
                        else:
                            self.translations[locale_code] = external_translations
                            
                        self.logger.debug(f"Loaded translations for {locale_code}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load translations from {locale_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Failed to load external translations: {e}")
    
    def _merge_translations(self, base: Dict, overlay: Dict) -> None:
        """Merge overlay translations into base translations."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_translations(base[key], value)
            else:
                base[key] = value
    
    def set_locale(self, locale: Union[SupportedLocale, str]) -> bool:
        """Set the current locale."""
        try:
            if isinstance(locale, str):
                locale = SupportedLocale(locale)
            
            if locale not in self.locale_configs:
                self.logger.warning(f"Unsupported locale: {locale}")
                return False
            
            self.current_locale = locale
            self.logger.info(f"Locale set to {locale.value}")
            return True
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Failed to set locale {locale}: {e}")
            return False
    
    def get_current_locale(self) -> SupportedLocale:
        """Get the current locale."""
        return self.current_locale
    
    def get_locale_config(self, locale: SupportedLocale = None) -> LocaleConfig:
        """Get configuration for a locale."""
        locale = locale or self.current_locale
        return self.locale_configs.get(locale, self.locale_configs[self.default_locale])
    
    def translate(self, key: str, locale: SupportedLocale = None, **kwargs) -> str:
        """Translate a message key."""
        locale = locale or self.current_locale
        locale_code = locale.value.lower()
        
        # Try exact locale match first
        if locale_code in self.translations:
            translations = self.translations[locale_code]
        else:
            # Try language fallback (e.g., en_US -> en)
            language = locale_code.split('_')[0]
            fallback_key = f"{language}_US" if language == "en" else f"{language}_{language.upper()}"
            
            if fallback_key in self.translations:
                translations = self.translations[fallback_key]
            else:
                # Final fallback to default locale
                default_locale_code = self.default_locale.value.lower()
                translations = self.translations.get(default_locale_code, {})
        
        # Navigate through nested keys
        keys = key.split('.')
        value = translations
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Key not found, return the key itself with fallback indicator
                self.logger.debug(f"Translation key not found: {key} for locale {locale_code}")
                return f"[{key}]"
        
        # Format the message with provided parameters
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except KeyError as e:
                self.logger.warning(f"Missing parameter {e} for translation key {key}")
                return value
        
        return str(value)
    
    def format_datetime(self, dt: datetime, locale: SupportedLocale = None, include_time: bool = True) -> str:
        """Format datetime according to locale conventions."""
        config = self.get_locale_config(locale)
        
        try:
            if include_time:
                format_str = f"{config.date_format} {config.time_format}"
            else:
                format_str = config.date_format
            
            return dt.strftime(format_str)
            
        except Exception as e:
            self.logger.warning(f"Failed to format datetime for locale {config.locale}: {e}")
            # Fallback to ISO format
            return dt.isoformat()
    
    def format_number(self, number: Union[int, float], locale: SupportedLocale = None, 
                     decimal_places: int = 2) -> str:
        """Format number according to locale conventions."""
        config = self.get_locale_config(locale)
        
        try:
            # This is a simplified implementation
            # In production, you'd use a proper localization library like babel
            if ',' in config.number_format and '.' in config.number_format:
                # Format like "1,234.56"
                if isinstance(number, float):
                    return f"{number:,.{decimal_places}f}"
                else:
                    return f"{number:,}"
            elif ' ' in config.number_format:
                # Format like "1 234,56" (French style)
                formatted = f"{number:.{decimal_places}f}".replace(',', ' ').replace('.', ',')
                return formatted
            else:
                # Default formatting
                return str(number)
                
        except Exception as e:
            self.logger.warning(f"Failed to format number for locale {config.locale}: {e}")
            return str(number)
    
    def format_currency(self, amount: Union[int, float], locale: SupportedLocale = None) -> str:
        """Format currency according to locale conventions."""
        config = self.get_locale_config(locale)
        
        try:
            formatted_number = self.format_number(amount, locale, decimal_places=2)
            
            # Simple currency formatting
            currency_symbols = {
                'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
                'CNY': '¥', 'SAR': 'ر.س', 'CAD': 'C$', 'AUD': 'A$'
            }
            
            symbol = currency_symbols.get(config.currency, config.currency)
            
            # Different currencies have different placement conventions
            if config.currency in ['USD', 'CAD', 'AUD']:
                return f"{symbol}{formatted_number}"
            elif config.currency == 'EUR':
                return f"{formatted_number} {symbol}"
            else:
                return f"{symbol} {formatted_number}"
                
        except Exception as e:
            self.logger.warning(f"Failed to format currency for locale {config.locale}: {e}")
            return f"{config.currency} {amount}"
    
    def get_compliance_message(self, message_type: str, locale: SupportedLocale = None, **kwargs) -> str:
        """Get compliance message for the locale's region."""
        config = self.get_locale_config(locale)
        
        # Map compliance regions to specific message keys
        region_key_map = {
            ComplianceRegion.EU: "gdpr",
            ComplianceRegion.US: "ccpa",
            ComplianceRegion.UK: "gdpr",  # UK GDPR
            ComplianceRegion.CA: "pipeda",
            ComplianceRegion.SG: "pdpa",
            ComplianceRegion.GLOBAL: "general"
        }
        
        region_key = region_key_map.get(config.compliance_region, "general")
        compliance_key = f"compliance.{region_key}.{message_type}"
        
        return self.translate(compliance_key, locale, **kwargs)
    
    def get_supported_locales(self) -> List[SupportedLocale]:
        """Get list of supported locales."""
        return list(self.locale_configs.keys())
    
    def get_pipeline_alert_message(self, alert_type: str, locale: SupportedLocale = None, **kwargs) -> str:
        """Get localized alert message for pipeline events."""
        key = f"pipeline.alerts.{alert_type}"
        return self.translate(key, locale, **kwargs)
    
    def get_pipeline_status_message(self, status: str, locale: SupportedLocale = None) -> str:
        """Get localized status message."""
        key = f"pipeline.status.{status}"
        return self.translate(key, locale)
    
    def get_pipeline_action_name(self, action: str, locale: SupportedLocale = None) -> str:
        """Get localized action name."""
        key = f"pipeline.actions.{action}"
        return self.translate(key, locale)


# Global instance for easy access
_i18n_manager = None

def get_i18n_manager() -> PipelineI18nManager:
    """Get the global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = PipelineI18nManager()
    return _i18n_manager

def set_global_locale(locale: Union[SupportedLocale, str]) -> bool:
    """Set the global locale."""
    return get_i18n_manager().set_locale(locale)

def translate(key: str, locale: SupportedLocale = None, **kwargs) -> str:
    """Translate a message key using the global manager."""
    return get_i18n_manager().translate(key, locale, **kwargs)

def format_pipeline_datetime(dt: datetime, locale: SupportedLocale = None) -> str:
    """Format datetime for pipeline messages."""
    return get_i18n_manager().format_datetime(dt, locale)

def get_localized_alert(alert_type: str, locale: SupportedLocale = None, **kwargs) -> str:
    """Get localized alert message."""
    return get_i18n_manager().get_pipeline_alert_message(alert_type, locale, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the i18n system
    i18n = PipelineI18nManager()
    
    # Test different locales
    test_locales = [SupportedLocale.EN_US, SupportedLocale.ES_ES, SupportedLocale.FR_FR, 
                   SupportedLocale.DE_DE, SupportedLocale.JA_JP, SupportedLocale.ZH_CN]
    
    for locale in test_locales:
        i18n.set_locale(locale)
        
        print(f"\n=== {locale.value} ===")
        print(f"Status: {i18n.get_pipeline_status_message('healthy')}")
        print(f"Alert: {i18n.get_pipeline_alert_message('high_cpu', usage=85)}")
        print(f"Action: {i18n.get_pipeline_action_name('restart_service')}")
        
        # Test datetime formatting
        now = datetime.now()
        print(f"DateTime: {i18n.format_datetime(now)}")
        
        # Test number formatting
        print(f"Number: {i18n.format_number(1234.56)}")
        
        # Test currency formatting
        config = i18n.get_locale_config()
        print(f"Currency: {i18n.format_currency(99.99)}")
    
    print("\n✅ I18n system testing completed")