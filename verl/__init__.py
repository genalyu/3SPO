# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import os


def patch_vllm_device_id_to_physical_device_id():
    """Monkey-patch vLLM's device_id_to_physical_device_id to handle MIG UUIDs.
    See https://github.com/vllm-project/vllm/issues/13815
    """
    try:
        import vllm.platforms.interface as vllm_interface

        def patched_fn(device_id):
            try:
                return int(device_id)
            except (ValueError, TypeError):
                # In MIG environment, device_id can be a string like 'MIG-xxx'
                # Each worker is already restricted to a single visible MIG, so
                # downstream code should treat it as local cuda:0.
                if isinstance(device_id, str) and device_id.startswith("MIG-"):
                    return 0
                return device_id

        if hasattr(vllm_interface, 'device_id_to_physical_device_id'):
            vllm_interface.device_id_to_physical_device_id = patched_fn

        for attr_name in ("Platform", "CudaPlatform", "NvmlCudaPlatform", "NonNvmlCudaPlatform"):
            platform_cls = getattr(vllm_interface, attr_name, None)
            if platform_cls is not None and hasattr(platform_cls, "device_id_to_physical_device_id"):
                platform_cls.device_id_to_physical_device_id = staticmethod(patched_fn)

        # For newer vLLM versions (e.g., 0.7.2+), it might be a method of Platform or subclasses
        try:
            from vllm.platforms.cuda import CudaPlatform
            if hasattr(CudaPlatform, 'device_id_to_physical_device_id'):
                CudaPlatform.device_id_to_physical_device_id = staticmethod(patched_fn)
        except (ImportError, AttributeError):
            pass

        # Newer vLLM releases route some checks through platform registry/current_platform.
        for module_name, attr_name in [
            ("vllm.platforms", "current_platform"),
            ("vllm.platforms.cuda", "CudaPlatform"),
        ]:
            try:
                module = importlib.import_module(module_name)
                platform_obj = getattr(module, attr_name, None)
                if platform_obj is not None and hasattr(platform_obj, "device_id_to_physical_device_id"):
                    setattr(platform_obj, "device_id_to_physical_device_id", staticmethod(patched_fn))
            except (ImportError, AttributeError):
                pass

    except (ImportError, AttributeError):
        pass


patch_vllm_device_id_to_physical_device_id()
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging.version import parse as parse_version
from .protocol import DataProto
from .utils.logging_utils import set_basic_config
from .utils.device import is_npu_available

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]

if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    import importlib

    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()

if is_npu_available:
    package_name = 'transformers'
    required_version_spec = '4.51.0'
    try:
        installed_version = get_version(package_name)
        installed = parse_version(installed_version)
        required = parse_version(required_version_spec)

        if not installed >= required:
            raise ValueError(f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is {installed}.")
    except PackageNotFoundError as e:
        raise ImportError(
            f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}"
        ) from e
