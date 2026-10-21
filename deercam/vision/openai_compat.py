from openai import OpenAI

from .base import ImagePart, T, encode_jpeg


class OpenAICompatProvider:
    """Any API that speaks OpenAI chat completions with image inputs and json_schema structured
    outputs: xAI (Grok), OpenAI, Google Gemini's compatibility endpoint, local servers."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
        self.model = model

    def structured(self, system: str, images: list[ImagePart], prompt: str, schema: type[T]) -> T:
        content = []
        for part in images:
            content.append({"type": "text", "text": part.caption})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_jpeg(part.path)}", "detail": "high"}})
        content.append({"type": "text", "text": prompt})

        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format=schema,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"{self.name} returned no parsed output: {message.refusal or message.content!r}")
        return message.parsed

    def models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())
