# -*- coding: utf-8 -*-
# NEW NODE: ACE_gemini_flash_image_2_5_multi
# - Single or MULTI-image input (batch on `images`)
# - Robust inline_data handling (auto base64 decode even when bytes)
# - Safe streaming (keeps final/largest), non-stream fallback
# - Returns stacked tensor when multiple images come back
# - Separate class & mappings from the single-image node
# - Hardening: request timeouts + 3 retries on timeouts/rate-limits (with backoff)

import io, os, base64, mimetypes, time, random
from typing import List, Tuple, Optional, Iterable

import numpy as np
from PIL import Image
import torch

_GENAI_OK = True
try:
    from google import genai
    from google.genai import types as gtypes
    from google.genai import errors as gerrors
except Exception:
    _GENAI_OK = False


# ---------- Tensor/PIL helpers ----------

def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    mode = "RGBA" if pil.mode in ("RGBA", "LA") else "RGB"
    arr = np.array(pil.convert(mode)).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 2)
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,C]

def _tensor_to_png_bytes(t: torch.Tensor) -> bytes:
    if t.dim() == 4:
        arr = t[0].detach().cpu().numpy()
    elif t.dim() == 3:
        arr = t.detach().cpu().numpy()
    else:
        raise ValueError("IMAGE must be 3D or 4D")
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB")
    buf = io.BytesIO(); pil.save(buf, "PNG"); return buf.getvalue()

def _stack_or_first(tensors: List[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros((1,1,1,3), dtype=torch.float32)
    sizes = {(t.shape[1], t.shape[2], t.shape[3]) for t in tensors}
    return torch.cat(tensors, dim=0) if len(sizes) == 1 else tensors[0]

def _iter_single_images(images: Optional[torch.Tensor]) -> Iterable[torch.Tensor]:
    """
    Yields each image as [H,W,C] float32 (0..1). Accepts:
      - None
      - [H,W,C]
      - [B,H,W,C]
    """
    if images is None:
        return
    if images.dim() == 3:
        yield images
    elif images.dim() == 4:
        for i in range(images.shape[0]):
            yield images[i]
    else:
        raise ValueError("images must be 3D [H,W,C] or 4D [B,H,W,C]")


# ---------- Byte normalization ----------

_BASE64_CHARS = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n")

def _looks_like_b64_bytes(b: bytes) -> bool:
    if not b or len(b) < 32:
        return False
    magic = b[:8]
    if magic.startswith(b"\x89PNG\r\n\x1a\n") or magic.startswith(b"\xff\xd8\xff") or magic.startswith(b"GIF87a") or magic.startswith(b"GIF89a") or magic.startswith(b"RIFF"):
        return False
    if any(c not in _BASE64_CHARS for c in b[: min(2048, len(b))]):
        return False
    try:
        base64.b64decode(b, validate=True)
        return True
    except Exception:
        return False

def _normalize_inline_data(data) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, str):
        # data URL?
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        try:
            return base64.b64decode(data)
        except Exception:
            return data.encode("utf-8", "ignore")
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if isinstance(data, bytes):
        if _looks_like_b64_bytes(data):
            try:
                return base64.b64decode(data, validate=True)
            except Exception:
                pass
        return data
    return None


# ---------- Gemini helpers ----------

def _decode_images(byte_mime_list: List[Tuple[bytes, Optional[str]]]) -> Tuple[List[torch.Tensor], List[str]]:
    tensors: List[torch.Tensor] = []
    warnings: List[str] = []
    for b, mt in byte_mime_list:
        try:
            img = Image.open(io.BytesIO(b))
            img.load()
            tensors.append(_pil_to_tensor(img.convert("RGBA")))
        except Exception as e:
            warnings.append(f"{e} (mime={mt})")
    return tensors, warnings

def _append_inline(parts, out: List[Tuple[bytes, Optional[str]]], texts: List[str]):
    for p in parts or []:
        t = getattr(p, "text", None)
        if t:
            texts.append(t)
        inline = getattr(p, "inline_data", None)
        if inline and getattr(inline, "data", None):
            norm = _normalize_inline_data(inline.data)
            if isinstance(norm, (bytes, bytearray)) and norm:
                out.append((bytes(norm), getattr(inline, "mime_type", None)))

def _best_inline_from_stream(parts, best: Optional[Tuple[bytes, Optional[str]]], best_len: int, texts: List[str]) -> Tuple[Optional[Tuple[bytes, Optional[str]]], int]:
    for p in parts or []:
        t = getattr(p, "text", None)
        if t:
            texts.append(t)
        inline = getattr(p, "inline_data", None)
        if inline and getattr(inline, "data", None):
            norm = _normalize_inline_data(inline.data)
            if isinstance(norm, (bytes, bytearray)) and norm:
                norm = bytes(norm)
                if len(norm) > best_len:
                    best = (norm, getattr(inline, "mime_type", None))
                    best_len = len(norm)
    return best, best_len

def _save_raw_bytes(byte_mime_list: List[Tuple[bytes, Optional[str]]]) -> List[str]:
    paths = []
    base_dir = os.path.expanduser("~/comfyui_temp")
    try: os.makedirs(base_dir, exist_ok=True)
    except Exception: base_dir = "/tmp"
    ts = int(time.time())
    for i, (b, mt) in enumerate(byte_mime_list):
        ext = mimetypes.guess_extension(mt or "") or ".bin"
        path = os.path.join(base_dir, f"gemini_{ts}_{i}{ext}")
        try:
            with open(path, "wb") as f: f.write(b)
            paths.append(path)
        except Exception:
            pass
    return paths


class ACE_gemini_flash_image_2_5_multi_retry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Describe or edit the image(s) ..."}),
            },
            "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 300}),
                "streaming": ("BOOLEAN", {"default": False}),       # safer default
                "force_image_only": ("BOOLEAN", {"default": False}), # show text if any
                "strict_fail": ("BOOLEAN", {"default": True}),
                "max_inline_images": ("INT", {"default": 3, "min": 1, "max": 16}),  # inline cap
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "run"
    CATEGORY = "AI/Gemini"

    # ---- request builders ----

    def _build_contents(self, prompt: str, images: Optional[torch.Tensor], max_inline_images: int) -> Tuple[list, Optional[str]]:
        """
        Returns (contents, note). The note mentions if we truncated a large batch.
        """
        parts = []
        note = None
        if images is not None:
            count = 0
            for img in _iter_single_images(images):
                parts.append(
                    gtypes.Part.from_bytes(
                        data=_tensor_to_png_bytes(img),
                        mime_type="image/png",
                    )
                )
                count += 1
                if count >= max_inline_images:
                    total = images.shape[0] if images.dim() == 4 else 1
                    if total > max_inline_images:
                        note = f"[info] truncated {total}→{max_inline_images} inline images"
                    break
        parts.append(gtypes.Part.from_text(text=prompt))
        return [gtypes.Content(role="user", parts=parts)], note

    # ---- low-level calls (with optional request_options) ----

    def _req_opts(self, timeout_seconds: int):
        try:
            if hasattr(gtypes, "RequestOptions"):
                return gtypes.RequestOptions(timeout=float(timeout_seconds))
        except Exception:
            pass
        return None

    def _stream_once(self, client, model: str, contents, cfg: gtypes.GenerateContentConfig, req_opts=None) -> Tuple[List[Tuple[bytes, Optional[str]]], List[str]]:
        best: Optional[Tuple[bytes, Optional[str]]] = None
        best_len = 0
        texts: List[str] = []
        kwargs = {"model": model, "contents": contents, "config": cfg}
        if req_opts is not None:
            kwargs["request_options"] = req_opts
        for chunk in client.models.generate_content_stream(**kwargs):
            if hasattr(chunk, "text") and chunk.text:
                texts.append(chunk.text)
            cand = (getattr(chunk, "candidates", None) or [None])[0]
            if not cand or not getattr(cand, "content", None):
                continue
            parts = getattr(cand.content, "parts", None) or []
            best, best_len = _best_inline_from_stream(parts, best, best_len, texts)
        return ([best] if best else []), texts

    def _nonstream_once(self, client, model: str, contents, cfg: gtypes.GenerateContentConfig, req_opts=None) -> Tuple[List[Tuple[bytes, Optional[str]]], List[str]]:
        imgs: List[Tuple[bytes, Optional[str]]] = []
        texts: List[str] = []
        kwargs = {"model": model, "contents": contents, "config": cfg}
        if req_opts is not None:
            kwargs["request_options"] = req_opts
        resp = client.models.generate_content(**kwargs)
        for cand in getattr(resp, "candidates", []) or []:
            if not cand or not getattr(cand, "content", None):
                continue
            _append_inline(getattr(cand, "content", None).parts if getattr(cand, "content", None) else None, imgs, texts)
        return imgs, texts

    # ---- retry helpers ----

    def _is_retryable(self, exc: Exception) -> bool:
        # Handle APIError codes and generic timeout/rate limit messages
        msg = str(exc).lower()
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", "").lower() if hasattr(exc, "status") else ""
        retryable_codes = {408, 409, 429, 500, 502, 503, 504}
        if isinstance(exc, getattr(gerrors, "APIError", tuple())) and (code in retryable_codes or "rate" in msg or "quota" in msg or "timeout" in msg or "deadline" in msg or "unavailable" in msg or "temporarily" in msg):
            return True
        # Generic network/timeout hints
        if any(k in msg for k in ["timeout", "timed out", "deadline", "temporarily unavailable", "rate limit", "quota"]):
            return True
        return False

    def _backoff(self, attempt: int):
        # Exponential backoff with jitter: 0.5s, 1s, 2s (+ jitter)
        base = min(0.5 * (2 ** (attempt - 1)), 4.0)
        time.sleep(base + random.uniform(0, 0.25))

    def _generate_with_retries(self, client, mdl: str, contents, cfg: gtypes.GenerateContentConfig,
                               want_stream: bool, timeout_seconds: int, max_attempts: int = 3):
        """
        Tries up to max_attempts. Will switch from stream->non-stream, and reduce inline images to 1 on later attempts.
        Returns (byte_mime_list, texts).
        """
        last_exc = None
        byte_mime_list: List[Tuple[bytes, Optional[str]]] = []
        texts: List[str] = []
        req_opts = self._req_opts(timeout_seconds)

        for attempt in range(1, max_attempts + 1):
            try:
                # First attempt: as requested. Later attempts: force non-stream.
                use_stream = want_stream and attempt == 1
                if use_stream:
                    byte_mime_list, texts = self._stream_once(client, mdl, contents, cfg, req_opts)
                else:
                    byte_mime_list, texts = self._nonstream_once(client, mdl, contents, cfg, req_opts)

                # If we got any inline image bytes, stop retrying.
                if byte_mime_list:
                    return byte_mime_list, texts

                # No bytes but no exception; treat as retryable soft-failure.
                last_exc = RuntimeError("Empty response (no image parts).")
                if attempt < max_attempts:
                    self._backoff(attempt)
                    continue
                break

            except Exception as e:
                last_exc = e
                if attempt < max_attempts and self._is_retryable(e):
                    self._backoff(attempt)
                    # On the final retry, also try to trim the request a bit:
                    # shrink max_output_tokens to reduce payload risk
                    try:
                        cfg.max_output_tokens = max(256, int(getattr(cfg, "max_output_tokens", 2048) / 2))
                    except Exception:
                        pass
                    continue
                else:
                    break

        # After retries, either return what we got (maybe empty) or raise if strict later wants it.
        if byte_mime_list:
            return byte_mime_list, texts
        # Propagate last exception context (caller decides strict_fail)
        raise last_exc if last_exc else RuntimeError("Gemini: unknown failure with no exception.")

    # ---- main ----

    def run(self,
            prompt: str,
            images: torch.Tensor = None,
            api_key: str = "",
            model: str = "gemini-2.5-flash-image-preview",
            temperature: float = 0.6,
            max_output_tokens: int = 2048,
            timeout_seconds: int = 60,
            streaming: bool = False,
            force_image_only: bool = False,
            strict_fail: bool = True,
            max_inline_images: int = 3):

        key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()

        if strict_fail and (not _GENAI_OK or not key):
            why = "google-genai not installed" if not _GENAI_OK else "missing GEMINI_API_KEY"
            raise RuntimeError(f"GeminiNoImage: {why}")

        if not _GENAI_OK or not key:
            out = images if images is not None else torch.zeros((1,1,1,3), dtype=torch.float32)
            msg = ("google-genai not installed. pip install --upgrade google-genai"
                   if not _GENAI_OK else "[Missing GEMINI_API_KEY]")
            return (out if out.dim()==4 else out.unsqueeze(0), msg)

        client = genai.Client(api_key=key)
        contents, trunc_note = self._build_contents(prompt, images, int(max_inline_images))
        mdl = (model or "gemini-2.5-flash-image-preview").strip()

        # Always ask for text+image to surface any safety/text responses
        cfg = gtypes.GenerateContentConfig(
            response_modalities=["TEXT","IMAGE"],
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )

        # 1) generate with retries (handles timeouts / rate limits up to 3 attempts)
        try:
            byte_mime_list, texts = self._generate_with_retries(
                client=client,
                mdl=mdl,
                contents=contents,
                cfg=cfg,
                want_stream=bool(streaming),
                timeout_seconds=int(timeout_seconds),
                max_attempts=3,  # REQUIRED: exactly three attempts on retryable errors
            )
        except Exception as e:
            if strict_fail:
                raise RuntimeError(f"GeminiNoImage: {e}") from e
            out = images if images is not None else torch.zeros((1,1,1,3), dtype=torch.float32)
            return (out if out.dim()==4 else out.unsqueeze(0), f"[Gemini error] {e}")

        # 2) decode
        tensors, decode_warnings = _decode_images(byte_mime_list)

        # 3) streaming fallback already handled in retries; if nothing, return error or passthrough
        if not tensors:
            saved_paths = []
            if byte_mime_list and not strict_fail:
                saved_paths = _save_raw_bytes(byte_mime_list)
            msg = "decode failed for all returned image parts."
            extras = []
            if trunc_note: extras.append(trunc_note)
            if decode_warnings: extras.append(" | ".join(decode_warnings))
            if saved_paths: extras.append("saved raw bytes: " + ", ".join(saved_paths))
            if not force_image_only and texts: extras.append(" ".join(texts))
            extra = " ".join(extras).strip()
            if strict_fail:
                raise RuntimeError(f"GeminiNoImage: {msg}" + (f" ({extra})" if extra else ""))
            out = images if images is not None else torch.zeros((1,1,1,3), dtype=torch.float32)
            return (out if out.dim()==4 else out.unsqueeze(0), (extra or msg))

        # Optional note (e.g., truncation) appears in text output
        if trunc_note:
            texts = [trunc_note] + (texts or [])

        text_out = "" if force_image_only else ("\n".join(texts) if texts else "")
        return (_stack_or_first(tensors), text_out)


NODE_CLASS_MAPPINGS = {"ACE_gemini_flash_image_2_5_multi_retry": ACE_gemini_flash_image_2_5_multi_retry}
NODE_DISPLAY_NAME_MAPPINGS = {"ACE_gemini_flash_image_2_5_multi_retry": "Gemini 2.5 Flash Image with retry"}
