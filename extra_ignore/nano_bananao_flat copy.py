# nano_banana_standalone.py
# Standalone ComfyUI node for Gemini 2.5 Flash Image (preview)
# - No config files, no env vars. API key is entered in the node.
import base64
from io import BytesIO
from typing import List, Tuple

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

class NanoBananaStandalone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True, "tooltip": "Gemini API key (paid image access)"}),
                "prompt": ("STRING", {"default": "Generate a high-quality, photorealistic image", "multiline": True}),
            },
            "optional": {
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {"default": "generate"}),
                "reference_image_1": ("IMAGE", {"forceInput": False}),
                "reference_image_2": ("IMAGE", {"forceInput": False}),
                "reference_image_3": ("IMAGE", {"forceInput": False}),
                "reference_image_4": ("IMAGE", {"forceInput": False}),
                "reference_image_5": ("IMAGE", {"forceInput": False}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_safety": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "run"
    CATEGORY = "NanoBanana (Standalone)"
    DESCRIPTION = "Gemini 2.5 Flash Image (preview). Standalone; enter API key in the node."

    def _prep_refs(self, *imgs) -> List[Image.Image]:
        out = []
        for im in imgs:
            if isinstance(im, torch.Tensor): out.append(_tensor_to_pil(im))
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

    def run(
        self, api_key: str, prompt: str, operation: str = "generate",
        reference_image_1=None, reference_image_2=None, reference_image_3=None,
        reference_image_4=None, reference_image_5=None,
        batch_count: int = 1, temperature: float = 0.7, enable_safety: bool = True
    ) -> Tuple[torch.Tensor, str]:
        key = (api_key or "").strip()
        if not key:
            return _placeholder(), "NANO BANANA ERROR: No API key provided in node input."

        refs = self._prep_refs(reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5)
        log = []
        try:
            from google import genai
            from google.genai import types, errors
            client = genai.Client(api_key=key)

            parts = [types.Part.from_text(text=prompt)]
            for p in refs: parts.append(_pil_part(p, 1024, "image/jpeg"))
            contents = [types.Content(role="user", parts=parts)]

            pils: List[Image.Image] = []
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
                                        pils.append(_decode_image_bytes(inline.data)); found += 1
                                    except Exception as e:
                                        log.append(f"Decode fail: {e}")
                    log.append(f"Batch {i+1}: Generated {found} images")
                except errors.APIError as e:
                    log.append(f"Batch {i+1} APIError {getattr(e,'code','')}: {getattr(e,'message',str(e))}")
                except Exception as e:
                    log.append(f"Batch {i+1} error: {e}")

            if not pils:
                log.append("No images were generated.")
                return _placeholder(), "\n".join(log)

            tensor = self._stack(pils)
            log.append(f"Successfully generated {tensor.shape[0]} image(s).")
            return tensor, "\n".join(log)

        except ImportError:
            return _placeholder(), "google.genai not installed. Install with: pip install google-genai pillow numpy"
        except Exception as e:
            log.append(f"Fatal error: {e}")
            return _placeholder(), "\n".join(log)

NODE_CLASS_MAPPINGS = {"NanoBananaStandalone": NanoBananaStandalone}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaStandalone": "Nano Banana (Gemini 2.5) — Standalone"}
