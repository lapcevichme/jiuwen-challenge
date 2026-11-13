import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any

from openjiuwen.core.foundation.llm import (
    AssistantMessage,
    BaseModelClient,
    Model,
    ModelClientConfig,
    ModelRequestConfig,
    UsageMetadata,
)
from openjiuwen.core.single_agent import AgentCard, BaseAgent

from harness.contracts import SolveResult, TaskSpec
from harness.llm_client import ChatResult, LLMClient
from harness.scheduler import FallbackSolver, ScheduleLimits, run_tasks

PROVIDER = "contest_proxy"
_BACKENDS: dict[str, LLMClient] = {}


class BudgetedProxyClient(BaseModelClient):

    __client_name__ = PROVIDER

    async def invoke(self, messages, *, tools=None, temperature=None, top_p=None, model=None, max_tokens=None,
                     stop=None, output_parser=None, timeout=None, **kwargs) -> AssistantMessage:
        backend = _BACKENDS[self.model_client_config.client_id]
        result = await backend.chat(
            model or self.model_config.model_name, self._convert_messages_to_dict(messages),
            max_tokens=max_tokens, timeout=timeout if timeout is not None else self.model_client_config.timeout,
            label=kwargs.get("label", ""), min_tokens=kwargs.get("min_tokens"),
        )
        return AssistantMessage(
            content=result.content, finish_reason=result.finish_reason or "null",
            usage_metadata=UsageMetadata(
                model_name=model or "", input_tokens=result.prompt_tokens, output_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens, total_latency=result.seconds,
            ),
            provider_metadata={"chat_result": dataclasses.asdict(result)},
        )

    async def stream(self, *args, **kwargs):
        raise NotImplementedError("the harness uses whole responses: usage must be known before the next send")

    async def generate_image(self, *args, **kwargs):
        raise NotImplementedError

    async def generate_speech(self, *args, **kwargs):
        raise NotImplementedError

    async def generate_video(self, *args, **kwargs):
        raise NotImplementedError


class JiuwenChat:

    def __init__(self, backend: LLMClient, *, default_model: str = "qwen3.8-27b-noreason"):
        self.backend = backend
        self.ledger = backend.ledger
        config = ModelClientConfig(client_provider=PROVIDER, api_key="set-in-backend", api_base=backend.url,
                                   verify_ssl=True)
        _BACKENDS[config.client_id] = backend
        self.model = Model(config, ModelRequestConfig(model=default_model, temperature=0))

    async def chat(self, model: str, messages: list[dict[str, str]], *, max_tokens: int, timeout: float,
                   label: str = "", min_tokens: int | None = None) -> ChatResult:
        reply = await self.model.invoke(messages, model=model, max_tokens=max_tokens, timeout=timeout,
                                        label=label, min_tokens=min_tokens)
        return ChatResult(**reply.provider_metadata["chat_result"])


class ContestAgent(BaseAgent):

    def __init__(
        self, deterministic: Callable[[TaskSpec], Awaitable[SolveResult]], *,
        fallback: FallbackSolver | None = None,
        last_resort: Callable[[TaskSpec], Awaitable[SolveResult]] | None = None,
        limits: ScheduleLimits = ScheduleLimits(),
    ):
        super().__init__(AgentCard(
            name="contest-harness",
            description="Budget-aware solver for math, search, data_analysis and coding tasks",
        ))
        self.deterministic = deterministic
        self.fallback = fallback
        self.last_resort = last_resort
        self.limits = limits

    def configure(self, config) -> "ContestAgent":
        if isinstance(config, ScheduleLimits):
            self.limits = config
        return self

    async def invoke(self, inputs: Any, session=None) -> dict:
        tasks: Iterable[TaskSpec] = inputs["tasks"]
        await run_tasks(tasks, self.deterministic, deadline=inputs["deadline"], fallback=self.fallback,
                        last_resort=self.last_resort, limits=self.limits)
        return {"status": "finished"}

    async def stream(self, inputs: Any, session=None, stream_modes=None) -> AsyncIterator[dict]:
        yield await self.invoke(inputs, session)
