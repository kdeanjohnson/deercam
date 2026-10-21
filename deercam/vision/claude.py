import anthropic

from .base import ImagePart, T, encode_jpeg


class RefusalError(RuntimeError):
    pass


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def structured(self, system: str, images: list[ImagePart], prompt: str, schema: type[T]) -> T:
        content = []
        for part in images:
            content.append({"type": "text", "text": part.caption})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": encode_jpeg(part.path)}})
        content.append({"type": "text", "text": prompt})

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,  # effort is left at Opus 5.5's default (medium)
            # Server-side fallback to another model if a safety classifier declines the request.
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
        if response.stop_reason == "refusal":
            raise RefusalError(str(response.stop_details))
        return response.parsed_output

    def models(self) -> list[str]:
        return [m.id for m in self.client.models.list(limit=100)]
