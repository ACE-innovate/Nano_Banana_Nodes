# nano_banana_parallel.py
# Parallel ComfyUI node for Gemini 2.5 Flash Image (preview)
# - Supports 1-9 parallel calls with shared or individual inputs
import base64
import asyncio
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    t = x.detach().cpu()
    if t.ndim == 4: t = t[0]
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def _placeholder() -> torch.Tensor:
    img = Image.new("RGB", (512, 512), (100, 100, 100))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def _decode_image_bytes(data: bytes | str) -> Image.Image:
    def _looks(b: bytes) -> bool:
        return len(b) >= 12 and (b.startswith(b"\x89PNG\r\n\x1a\n") or b.startswith(b"\xff\xd8\xff") or (b[:4]==b"RIFF" and b[8:12]==b"WEBP"))
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        try: return Image.open(BytesIO(raw)).convert("RGB")
        except: pass
        try:
            raw2 = base64.b64decode(raw, validate=False)
            if _looks(raw2): return Image.open(BytesIO(raw2)).convert("RGB")
        except: pass
    if isinstance(data, str):
        s = data.strip()
        if s.startswith("data:"):
            try:
                b64 = s.split(",", 1)[1]
                raw = base64.b64decode(b64, validate=False)
                return Image.open(BytesIO(raw)).convert("RGB")
            except: pass
        try:
            raw = base64.b64decode(s, validate=False)
            return Image.open(BytesIO(raw)).convert("RGB")
        except: pass
    raise ValueError("Unrecognized image bytes/encoding from API")

def _pil_part(pil: Image.Image, max_side=1024, mime="image/jpeg"):
    from google.genai import types  # lazy import
    pil = pil.convert("RGB")
    w, h = pil.size
    s = max(w, h)
    if s > max_side:
        sc = max_side / float(s)
        pil = pil.resize((int(w*sc), int(h*sc)), Image.LANCZOS)
    buf = BytesIO()
    if mime == "image/jpeg":
        pil.save(buf, "JPEG", quality=92, optimize=True)
    else:
        pil.save(buf, "PNG")
    data = buf.getvalue()
    Image.open(BytesIO(data)).verify()
    return types.Part.from_bytes(data=data, mime_type=mime)

class NanoBananaParallel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True, "tooltip": "Gemini API key (paid image access)"}),
                "prompt": ("STRING", {"default": "Generate a high-quality, photorealistic image", "multiline": True}),
                "in_parallel": ("INT", {"default": 1, "min": 1, "max": 9, "step": 1, "tooltip": "Number of parallel calls (1-9)"}),
                "parallels_share_inputs": ("BOOLEAN", {"default": True, "tooltip": "Use same inputs for all parallel calls"}),
            },
            "optional": {
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {"default": "generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_safety": ("BOOLEAN", {"default": True}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
        }

    def __init__(self):
        self._current_parallel_count = 1
        self._current_share_inputs = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        in_parallel = kwargs.get("in_parallel", 1)
        parallels_share_inputs = kwargs.get("parallels_share_inputs", True)
        
        # Validate that we have the required inputs for each active slot
        if parallels_share_inputs:
            # Shared inputs - need one set
            if not kwargs.get("source_image"):
                return "Source image is required when sharing inputs"
        else:
            # Individual inputs - need source for each active slot
            for i in range(1, in_parallel + 1):
                if not kwargs.get(f"source_image_{i}"):
                    return f"Source image is required for slot {i} when not sharing inputs"
        
        return True

    def _get_dynamic_input_types(self, in_parallel: int, parallels_share_inputs: bool) -> Dict[str, Any]:
        """Generate dynamic input types based on parallel settings"""
        inputs = {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True, "tooltip": "Gemini API key (paid image access)"}),
                "prompt": ("STRING", {"default": "Generate a high-quality, photorealistic image", "multiline": True}),
                "in_parallel": ("INT", {"default": 1, "min": 1, "max": 9, "step": 1, "tooltip": "Number of parallel calls (1-9)"}),
                "parallels_share_inputs": ("BOOLEAN", {"default": True, "tooltip": "Use same inputs for all parallel calls"}),
            },
            "optional": {
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {"default": "generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_safety": ("BOOLEAN", {"default": True}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
        }
        
        if parallels_share_inputs:
            # Shared inputs - one set for all parallel calls
            inputs["required"]["source_image"] = ("IMAGE", {"tooltip": "Source image for all parallel calls"})
            inputs["optional"]["ref_a"] = ("IMAGE", {"forceInput": False, "tooltip": "Reference image A for all parallel calls"})
            inputs["optional"]["ref_b"] = ("IMAGE", {"forceInput": False, "tooltip": "Reference image B for all parallel calls"})
        else:
            # Individual inputs - separate set for each parallel call
            for i in range(1, in_parallel + 1):
                inputs["required"][f"source_image_{i}"] = ("IMAGE", {"tooltip": f"Source image for call {i}"})
                inputs["optional"][f"ref_a_{i}"] = ("IMAGE", {"forceInput": False, "tooltip": f"Reference image A for call {i}"})
                inputs["optional"][f"ref_b_{i}"] = ("IMAGE", {"forceInput": False, "tooltip": f"Reference image B for call {i}"})
        
        return inputs

    def _get_dynamic_return_types(self, in_parallel: int) -> Tuple[List[str], List[str]]:
        """Generate dynamic return types based on parallel count"""
        return_types = []
        return_names = []
        
        for i in range(1, in_parallel + 1):
            return_types.append("IMAGE")
            return_names.append(f"result_{i}")
        
        # Add operation log
        return_types.append("STRING")
        return_names.append("operation_log")
        
        return return_types, return_names

    RETURN_TYPES = ("IMAGE", "STRING")  # Default - will be overridden
    RETURN_NAMES = ("result_1", "operation_log")  # Default - will be overridden
    FUNCTION = "run"
    CATEGORY = "NanoBanana (Parallel)"
    DESCRIPTION = "Gemini 2.5 Flash Image (preview) with parallel execution support"

    def _prep_refs(self, *imgs) -> List[Image.Image]:
        out = []
        for im in imgs:
            if isinstance(im, torch.Tensor): 
                out.append(_tensor_to_pil(im))
        return out

    def _stack(self, pils: List[Image.Image]) -> torch.Tensor:
        if not pils: return _placeholder()
        w = min(p.width for p in pils); h = min(p.height for p in pils)
        batch = []
        for p in pils:
            if p.size != (w, h): p = p.resize((w, h), Image.LANCZOS)
            arr = np.asarray(p, dtype=np.float32) / 255.0
            batch.append(torch.from_numpy(arr)[None, ...])
        return torch.cat(batch, 0)

    def _single_api_call(self, api_key: str, prompt: str, source_img: Optional[torch.Tensor], 
                        ref_a: Optional[torch.Tensor], ref_b: Optional[torch.Tensor],
                        operation: str, temperature: float, batch_count: int, 
                        enable_safety: bool, slot_id: int) -> Tuple[torch.Tensor, str]:
        """Execute a single API call for one slot"""
        try:
            from google import genai
            from google.genai import types, errors
            client = genai.Client(api_key=api_key)

            parts = [types.Part.from_text(text=prompt)]
            
            # Add source image if provided
            if source_img is not None:
                source_pil = _tensor_to_pil(source_img)
                parts.append(_pil_part(source_pil, 1024, "image/jpeg"))
            
            # Add reference images if provided
            refs = self._prep_refs(ref_a, ref_b)
            for ref_pil in refs:
                parts.append(_pil_part(ref_pil, 1024, "image/jpeg"))
            
            contents = [types.Content(role="user", parts=parts)]

            pils: List[Image.Image] = []
            log = []
            
            for i in range(batch_count):
                try:
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash-image-preview",
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=temperature),
                    )
                    found = 0
                    if getattr(resp, "candidates", None):
                        for c in resp.candidates:
                            for part in getattr(c.content, "parts", []):
                                inline = getattr(part, "inline_data", None)
                                if inline and getattr(inline, "data", None):
                                    try:
                                        pils.append(_decode_image_bytes(inline.data))
                                        found += 1
                                    except Exception as e:
                                        log.append(f"Slot {slot_id} batch {i+1} decode fail: {e}")
                    log.append(f"Slot {slot_id} batch {i+1}: Generated {found} images")
                except errors.APIError as e:
                    log.append(f"Slot {slot_id} batch {i+1} APIError {getattr(e,'code','')}: {getattr(e,'message',str(e))}")
                except Exception as e:
                    log.append(f"Slot {slot_id} batch {i+1} error: {e}")

            if not pils:
                log.append(f"Slot {slot_id}: No images were generated.")
                return _placeholder(), "\n".join(log)

            tensor = self._stack(pils)
            log.append(f"Slot {slot_id}: Successfully generated {tensor.shape[0]} image(s).")
            return tensor, "\n".join(log)

        except ImportError:
            return _placeholder(), f"Slot {slot_id}: google.genai not installed. Install with: pip install google-genai pillow numpy"
        except Exception as e:
            return _placeholder(), f"Slot {slot_id}: Fatal error: {e}"

    def run(self, api_key: str, prompt: str, in_parallel: int = 1, 
            parallels_share_inputs: bool = True, operation: str = "generate",
            temperature: float = 0.7, enable_safety: bool = True, batch_count: int = 1,
            **kwargs) -> Tuple[torch.Tensor, ...]:
        """Execute parallel API calls"""
        
        # Update dynamic return types
        return_types, return_names = self._get_dynamic_return_types(in_parallel)
        self.RETURN_TYPES = tuple(return_types)
        self.RETURN_NAMES = tuple(return_names)
        
        key = (api_key or "").strip()
        if not key:
            error_msg = "NANO BANANA ERROR: No API key provided in node input."
            return tuple([_placeholder()] * in_parallel + [error_msg])

        # Prepare input data for each slot
        slot_data = []
        
        if parallels_share_inputs:
            # Shared inputs - use same data for all slots
            source_img = kwargs.get("source_image")
            ref_a = kwargs.get("ref_a")
            ref_b = kwargs.get("ref_b")
            
            for i in range(in_parallel):
                slot_data.append({
                    'source': source_img,
                    'ref_a': ref_a,
                    'ref_b': ref_b,
                    'slot_id': i + 1
                })
        else:
            # Individual inputs - separate data for each slot
            for i in range(in_parallel):
                slot_id = i + 1
                source_img = kwargs.get(f"source_image_{slot_id}")
                ref_a = kwargs.get(f"ref_a_{slot_id}")
                ref_b = kwargs.get(f"ref_b_{slot_id}")
                
                slot_data.append({
                    'source': source_img,
                    'ref_a': ref_a,
                    'ref_b': ref_b,
                    'slot_id': slot_id
                })

        # Validate that all required sources are provided
        missing_sources = []
        for slot in slot_data:
            if slot['source'] is None:
                missing_sources.append(slot['slot_id'])
        
        if missing_sources:
            error_msg = f"NANO BANANA ERROR: Missing source images for slots: {', '.join(map(str, missing_sources))}"
            return tuple([_placeholder()] * in_parallel + [error_msg])

        # Execute parallel calls
        results = []
        logs = []
        
        try:
            with ThreadPoolExecutor(max_workers=min(in_parallel, 9)) as executor:
                # Submit all tasks
                future_to_slot = {}
                for slot in slot_data:
                    future = executor.submit(
                        self._single_api_call,
                        key, prompt, slot['source'], slot['ref_a'], slot['ref_b'],
                        operation, temperature, batch_count, enable_safety, slot['slot_id']
                    )
                    future_to_slot[future] = slot['slot_id']
                
                # Collect results in order
                slot_results = {}
                for future in as_completed(future_to_slot):
                    slot_id = future_to_slot[future]
                    try:
                        result_tensor, log_msg = future.result()
                        slot_results[slot_id] = (result_tensor, log_msg)
                    except Exception as e:
                        slot_results[slot_id] = (_placeholder(), f"Slot {slot_id}: Execution error: {e}")
                
                # Order results by slot ID
                for i in range(1, in_parallel + 1):
                    if i in slot_results:
                        result_tensor, log_msg = slot_results[i]
                        results.append(result_tensor)
                        logs.append(log_msg)
                    else:
                        results.append(_placeholder())
                        logs.append(f"Slot {i}: No result available")

        except Exception as e:
            error_msg = f"Parallel execution error: {e}"
            return tuple([_placeholder()] * in_parallel + [error_msg])

        # Combine all logs
        combined_log = f"Parallel execution completed ({in_parallel} slots):\n" + "\n".join(logs)
        
        return tuple(results + [combined_log])

# Node class mappings
NODE_CLASS_MAPPINGS = {"NanoBananaParallel": NanoBananaParallel}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaParallel": "Nano Banana (Gemini 2.5) — Parallel"}
