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
                "in_parallel": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1, "tooltip": "Number of parallel calls (1-5)"}),
                "parallels_share_inputs": ("BOOLEAN", {"default": False, "tooltip": "Use slot 1 inputs for all parallel calls"}),
                # Slot 1 inputs (used for all when sharing)
                "prompt_1": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 1"}),
                # Slot 2 inputs
                "prompt_2": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 2"}),
                # Slot 3 inputs
                "prompt_3": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 3"}),
                # Slot 4 inputs
                "prompt_4": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 4"}),
                # Slot 5 inputs
                "prompt_5": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt 5"}),
            },
            "optional": {
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {"default": "generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_safety": ("BOOLEAN", {"default": True}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                # Slot 1 image inputs
                "source_image_1": ("IMAGE", {"forceInput": False, "tooltip": "Source image for call 1"}),
                "ref_a_1": ("IMAGE", {"forceInput": False, "tooltip": "Reference image A for call 1"}),
                "ref_b_1": ("IMAGE", {"forceInput": False, "tooltip": "Reference image B for call 1"}),
                # Slot 2 image inputs
                "source_image_2": ("IMAGE", {"forceInput": False, "tooltip": "Source image for call 2"}),
                "ref_a_2": ("IMAGE", {"forceInput": False, "tooltip": "Reference image A for call 2"}),
                "ref_b_2": ("IMAGE", {"forceInput": False, "tooltip": "Reference image B for call 2"}),
                # Slot 3 image inputs
                "source_image_3": ("IMAGE", {"forceInput": False, "tooltip": "Source image for call 3"}),
                "ref_a_3": ("IMAGE", {"forceInput": False, "tooltip": "Reference image A for call 3"}),
                "ref_b_3": ("IMAGE", {"forceInput": False, "tooltip": "Reference image B for call 3"}),
                # Slot 4 image inputs
                "source_image_4": ("IMAGE", {"forceInput": False, "tooltip": "Source image for call 4"}),
                "ref_a_4": ("IMAGE", {"forceInput": False, "tooltip": "Reference image A for call 4"}),
                "ref_b_4": ("IMAGE", {"forceInput": False, "tooltip": "Reference image B for call 4"}),
                # Slot 5 image inputs
                "source_image_5": ("IMAGE", {"forceInput": False, "tooltip": "Source image for call 5"}),
                "ref_a_5": ("IMAGE", {"forceInput": False, "tooltip": "Reference image A for call 5"}),
                "ref_b_5": ("IMAGE", {"forceInput": False, "tooltip": "Reference image B for call 5"}),
            },
        }

    def __init__(self):
        self._current_parallel_count = 1
        self._current_share_inputs = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        in_parallel = kwargs.get("in_parallel", 1)
        parallels_share_inputs = kwargs.get("parallels_share_inputs", False)
        operation = kwargs.get("operation", "generate")
        
        # Only require source images for editing modes, not for generate mode
        if operation in ["edit", "style_transfer", "object_insertion"]:
            if parallels_share_inputs:
                # When sharing, only need slot 1 source for editing modes
                if not kwargs.get("source_image_1"):
                    return "Source image 1 is required for editing modes when sharing inputs"
            else:
                # Individual inputs - need source for each active slot for editing modes
                missing_sources = []
                for i in range(1, in_parallel + 1):
                    if not kwargs.get(f"source_image_{i}"):
                        missing_sources.append(str(i))
                if missing_sources:
                    return f"Source images are required for slots {', '.join(missing_sources)} in editing modes when not sharing inputs"
        
        # References are always optional - no validation needed
        return True


    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("result_1", "result_2", "result_3", "result_4", "result_5", "operation_log")
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
            
            # Add source image if provided (required for editing modes, optional for generate)
            if source_img is not None:
                source_pil = _tensor_to_pil(source_img)
                parts.append(_pil_part(source_pil, 1024, "image/jpeg"))
            
            # Add reference images if provided (always optional)
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

    def run(self, api_key: str, in_parallel: int = 1, parallels_share_inputs: bool = False,
            prompt_1: str = "", prompt_2: str = "", prompt_3: str = "", prompt_4: str = "", prompt_5: str = "",
            operation: str = "generate", temperature: float = 0.7, enable_safety: bool = True, batch_count: int = 1,
            source_image_1=None, ref_a_1=None, ref_b_1=None,
            source_image_2=None, ref_a_2=None, ref_b_2=None,
            source_image_3=None, ref_a_3=None, ref_b_3=None,
            source_image_4=None, ref_a_4=None, ref_b_4=None,
            source_image_5=None, ref_a_5=None, ref_b_5=None,
            **kwargs) -> Tuple[torch.Tensor, ...]:
        """Execute parallel API calls"""
        
        # No need to update return types - they're static now
        
        key = (api_key or "").strip()
        if not key:
            error_msg = "NANO BANANA ERROR: No API key provided in node input."
            return tuple([_placeholder()] * 5 + [error_msg])

        # Prepare input data for each slot
        slot_data = []
        
        if parallels_share_inputs:
            # Shared inputs - use slot 1 data for all slots
            shared_prompt = prompt_1 or "Generate a high-quality, photorealistic image"
            
            for i in range(in_parallel):
                slot_data.append({
                    'prompt': shared_prompt,
                    'source': source_image_1,
                    'ref_a': ref_a_1,
                    'ref_b': ref_b_1,
                    'slot_id': i + 1
                })
        else:
            # Individual inputs - separate data for each slot
            prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5]
            sources = [source_image_1, source_image_2, source_image_3, source_image_4, source_image_5]
            refs_a = [ref_a_1, ref_a_2, ref_a_3, ref_a_4, ref_a_5]
            refs_b = [ref_b_1, ref_b_2, ref_b_3, ref_b_4, ref_b_5]
            
            for i in range(in_parallel):
                slot_id = i + 1
                individual_prompt = prompts[i] or "Generate a high-quality, photorealistic image"
                
                slot_data.append({
                    'prompt': individual_prompt,
                    'source': sources[i],
                    'ref_a': refs_a[i],
                    'ref_b': refs_b[i],
                    'slot_id': slot_id
                })

        # Validate that all required sources are provided (only for editing modes)
        if operation in ["edit", "style_transfer", "object_insertion"]:
            missing_sources = []
            for slot in slot_data:
                if slot['source'] is None:
                    missing_sources.append(slot['slot_id'])
            
            if missing_sources:
                if parallels_share_inputs:
                    error_msg = f"NANO BANANA ERROR: Missing source_image_1 (required for editing modes when sharing inputs)"
                else:
                    error_msg = f"NANO BANANA ERROR: Missing source images for slots: {', '.join(map(str, missing_sources))} (required for editing modes when not sharing inputs)"
                return tuple([_placeholder()] * 5 + [error_msg])

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
                        key, slot['prompt'], slot['source'], slot['ref_a'], slot['ref_b'],
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
            return tuple([_placeholder()] * 5 + [error_msg])

        # Pad results to always return 5 outputs
        while len(results) < 5:
            results.append(_placeholder())
        
        # Combine all logs
        combined_log = f"Parallel execution completed ({in_parallel} slots):\n" + "\n".join(logs)
        
        return tuple(results + [combined_log])

# Node class mappings
NODE_CLASS_MAPPINGS = {"NanoBananaParallel": NanoBananaParallel}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaParallel": "Nano Banana (Gemini 2.5) — Parallel"}
