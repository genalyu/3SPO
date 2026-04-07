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

from importlib.metadata import PackageNotFoundError, version

from packaging import version as vs

from verl.utils.import_utils import is_sglang_available


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

        # For newer vLLM versions (e.g., 0.7.2+), it might be a method of Platform or subclasses
        try:
            from vllm.platforms.cuda import CudaPlatform
            if hasattr(CudaPlatform, 'device_id_to_physical_device_id'):
                CudaPlatform.device_id_to_physical_device_id = staticmethod(patched_fn)
        except (ImportError, AttributeError):
            pass

        for module_name, attr_name in [
            ("vllm.platforms", "current_platform"),
            ("vllm.platforms.cuda", "CudaPlatform"),
        ]:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                platform_obj = getattr(module, attr_name, None)
                if platform_obj is not None and hasattr(platform_obj, "device_id_to_physical_device_id"):
                    setattr(platform_obj, "device_id_to_physical_device_id", staticmethod(patched_fn))
            except (ImportError, AttributeError):
                pass

    except (ImportError, AttributeError):
        pass


patch_vllm_device_id_to_physical_device_id()


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)
vllm_version = None


if package_version == "0.5.4":
    vllm_version = "0.5.4"
    from .vllm_v_0_5_4 import parallel_state
    from .vllm_v_0_5_4.llm import LLM, LLMEngine
elif package_version == "0.6.3" or package_version.startswith("0.6.3"):
    # rocm version: "0.6.3+rocmxxx"
    vllm_version = "0.6.3"
    from .vllm_v_0_6_3 import parallel_state
    from .vllm_v_0_6_3.llm import LLM, LLMEngine
elif vs.parse(package_version) >= vs.parse("0.7.0"):
    # From 0.6.6.post2 on, vllm supports SPMD inference
    # See https://github.com/vllm-project/vllm/pull/12071

    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    if not is_sglang_available():
        raise ValueError(f"vllm version {package_version} not supported and SGLang also not Found. Currently supported vllm versions are 0.6.3 and 0.7.0+")

__all__ = ["LLM", "LLMEngine", "parallel_state"]
