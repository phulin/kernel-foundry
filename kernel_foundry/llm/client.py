from __future__ import annotations

import time

import openai

from kernel_foundry.config import EvolutionConfig


class LLMClient:
    """Thin wrapper around the OpenAI ChatCompletion API with retry logic."""

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(api_key=config.openai_api_key)
        self.tokens_input = 0
        self.tokens_output = 0
        self.tokens_cached = 0

    def generate(
        self,
        prompt: str,
        n: int = 1,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> list[str]:
        """
        Generate n completions for the given prompt.
        Returns list of raw response strings (length == n).
        Retries on rate-limit and transient errors with exponential backoff.
        """
        effective_model = model or self._config.llm_model
        effective_temp = (
            temperature if temperature is not None else self._config.llm_temperature
        )

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=effective_model,
                    messages=[{"role": "user", "content": prompt}],
                    n=n,
                    temperature=effective_temp,
                    max_completion_tokens=self._config.llm_max_tokens,
                    top_p=self._config.llm_top_p,
                )
                if response.usage:
                    self.tokens_input += response.usage.prompt_tokens
                    self.tokens_output += response.usage.completion_tokens
                    details = response.usage.prompt_tokens_details
                    if details and details.cached_tokens:
                        self.tokens_cached += details.cached_tokens
                return [choice.message.content or "" for choice in response.choices]
            except openai.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            except openai.APITimeoutError:
                wait = 2**attempt
                print(f"  API timeout. Waiting {wait}s...")
                time.sleep(wait)
            except openai.APIError as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                print(f"  API error ({e}). Waiting {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts")

    def generate_single(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.generate(prompt, n=1, model=model, temperature=temperature)[0]
