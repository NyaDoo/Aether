"""
功能模块注册 -- 自动发现

扫描 src/modules/ 下的子目录，自动查找 ModuleDefinition 实例。
新增模块只需创建 src/modules/<name>/__init__.py 并导出 ModuleDefinition，无需修改此文件。
"""

import importlib
from pathlib import Path

from src.core.logger import logger
from src.core.modules.base import ModuleDefinition


def discover_modules() -> list[ModuleDefinition]:
    """
    自动发现所有模块定义

    扫描 src/modules/ 下的每个子目录，导入其 __init__.py，
    查找所有 ModuleDefinition 实例并返回。
    """
    modules_dir = Path(__file__).parent
    discovered: list[ModuleDefinition] = []
    seen_names: set[str] = set()

    for child in sorted(modules_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if not (child / "__init__.py").exists():
            continue

        module_path = f"src.modules.{child.name}"
        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            logger.error("Failed to import module {}: {}", module_path, e)
            continue

        # 扫描模块顶层属性，查找 ModuleDefinition 实例
        for obj in vars(mod).values():
            if isinstance(obj, ModuleDefinition):
                name = obj.metadata.name
                if name in seen_names:
                    logger.warning("Duplicate module name '{}' in {}, skipping", name, module_path)
                    continue
                seen_names.add(name)
                discovered.append(obj)
                logger.debug("Discovered module: {} from {}", name, module_path)

    return discovered


ALL_MODULES: list[ModuleDefinition] = discover_modules()

__all__ = ["ALL_MODULES", "discover_modules"]
