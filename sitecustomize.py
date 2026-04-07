import importlib
import os


def _patch_vllm_mig_device_ids():
    def patched_fn(device_id):
        try:
            return int(device_id)
        except (ValueError, TypeError):
            if isinstance(device_id, str) and device_id.startswith("MIG-"):
                # When a worker is restricted to a single MIG UUID, downstream
                # code should address it as local cuda:0.
                return 0
            return device_id

    try:
        import vllm.platforms.interface as vllm_interface

        if hasattr(vllm_interface, "device_id_to_physical_device_id"):
            vllm_interface.device_id_to_physical_device_id = patched_fn

        for attr_name in ("Platform", "CudaPlatform", "NvmlCudaPlatform", "NonNvmlCudaPlatform"):
            platform_cls = getattr(vllm_interface, attr_name, None)
            if platform_cls is not None and hasattr(platform_cls, "device_id_to_physical_device_id"):
                platform_cls.device_id_to_physical_device_id = staticmethod(patched_fn)

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
    except Exception:
        # Keep Python startup resilient even if vLLM is unavailable.
        pass


def _patch_single_mig_visibility():
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if isinstance(original_cvd, str) and original_cvd.startswith("MIG-") and "," not in original_cvd:
        # Some vLLM init-time checks incorrectly int() CUDA_VISIBLE_DEVICES.
        # Single-MIG workers should treat the visible device as local index 0.
        os.environ["ORIG_CUDA_VISIBLE_DEVICES"] = original_cvd
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


_patch_single_mig_visibility()
_patch_vllm_mig_device_ids()
