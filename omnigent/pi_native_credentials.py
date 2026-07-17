"""Translate the omnigent-configured model provider into native Pi config.

A native Pi session launches the ``pi`` CLI, which authenticates from its own
config directory (``~/.pi/agent``). Without help, a user who ran ``omnigent
setup`` would still have to run ``pi`` ``/login`` separately — unlike
claude-native / codex-native, which route through the provider that ``omnigent
setup`` configured.

This module closes that gap. It resolves the provider configured for the Pi
surface (``~/.omnigent/config.yaml``) and writes a per-session ``models.json``
into a *managed* Pi config dir (selected via ``PI_CODING_AGENT_DIR``), so the
runner-owned ``pi`` process authenticates exactly like the configured harness —
mirroring how codex-native routes through the Databricks AI Gateway.

The managed config dir is per-session (like codex-native's managed
``CODEX_HOME``), so this never mutates the user's global ``~/.pi/agent``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from omnigent.model_override import normalize_model_for_provider
from omnigent.onboarding.databricks_config import (
    DATABRICKS_ANTHROPIC_MODELS,
    DATABRICKS_GPT_MODELS,
)
from omnigent.onboarding.provider_config import (
    CHAT_WIRE_API,
    CLI_CONFIG_KIND,
    DATABRICKS_KIND,
    GATEWAY_KIND,
    KEY_KIND,
    LOCAL_KIND,
    PI_SURFACE,
    ProviderEntry,
    default_provider_for_harness,
    load_config,
)

if TYPE_CHECKING:
    # Annotation-only import (the runtime import is lazy inside the function,
    # since ``ambient`` pulls in onboarding-only deps this module avoids on the
    # runner's session-create hot path).
    from omnigent.onboarding.ambient import CodexConfigTransport

_LOGGER = logging.getLogger(__name__)

# Env var the ``pi`` CLI reads to relocate its config dir (default
# ``~/.pi/agent``). Setting it per session gives Pi a managed, isolated
# config dir we own — the analog of codex-native's ``CODEX_HOME``.
PI_CODING_AGENT_DIR_ENV_VAR = "PI_CODING_AGENT_DIR"

# Provider id registered in the generated ``models.json``. Stable so
# ``--provider`` can select it.
_PI_PROVIDER_ID = "omnigent"

# Default model for the Databricks AI Gateway's Anthropic surface — the same
# default the in-process Databricks executor pins. Used when the session
# carries no explicit model override.
_DATABRICKS_PI_DEFAULT_MODEL = "databricks-claude-sonnet-4-6"

# Databricks AI Gateway Anthropic Messages surface. Pi speaks this protocol
# natively (``api: anthropic-messages``); the gateway authenticates with a
# workspace bearer token, so we set ``authHeader`` (Authorization: Bearer).
_DATABRICKS_ANTHROPIC_GATEWAY_PATH = "/ai-gateway/anthropic"

# The Databricks AI Gateway exposes one surface per protocol under the same
# workspace origin: Codex/OpenAI-Responses at ``/codex/v1`` and Anthropic
# Messages at ``/anthropic``. ``isaac configure codex`` writes the Codex
# base_url; pi-native rewrites it to the Anthropic surface Pi speaks natively.
_DATABRICKS_GATEWAY_CODEX_SUFFIX = "/codex/v1"
_DATABRICKS_GATEWAY_ANTHROPIC_SUFFIX = "/anthropic"

# Databricks serving-endpoints surface: the OpenAI-compatible endpoint
# (``/serving-endpoints/v1/chat/completions``) that non-Claude gateway models
# (GLM, Gemini, DeepSeek, …) are served on. Pi speaks it via
# ``api: openai-completions``. The gateway rejects these models on the
# Anthropic Messages surface ("API type 'anthropic/v1/messages' is not
# supported"), so routing them here is the fix for #2575. Mirrors the
# in-process harness (``inner/pi_executor._build_models_json``), which routes
# the same families to this surface.
_DATABRICKS_SERVING_ENDPOINTS_PATH = "/serving-endpoints"

# Provider id for the OpenAI-completions (serving-endpoints) surface in the
# generated ``models.json``. Distinct from ``_PI_PROVIDER_ID`` (the Anthropic
# surface) so both families can be registered side by side and Pi's ``/model``
# picker can offer either — the "secondary provider registration for model
# discovery" the issue calls for.
_PI_OPENAI_PROVIDER_ID = "omnigent-completions"

# Compatibility flags Pi applies on the Databricks OpenAI-completions surface.
# ``supportsDeveloperRole: False`` is the load-bearing one for #2575 —
# Databricks-gatewayed Gemini rejects the OpenAI ``developer`` role Pi sends by
# default; the rest mirror the proven harness-path settings. Stored as a tuple
# of pairs so ``PiProviderConfig`` stays a hashable frozen dataclass.
_DATABRICKS_OPENAI_COMPAT: tuple[tuple[str, bool], ...] = (
    ("supportsDeveloperRole", False),
    ("supportsStore", False),
    ("supportsStrictMode", False),
    ("supportsReasoningEffort", False),
)

# Model-id substrings whose Databricks endpoints stream chain-of-thought on
# ``reasoning_content`` rather than ``content``; Pi needs ``reasoning: true`` on
# the model entry to surface it (per #2575). NOTE: the exact Pi ``models.json``
# key is owned by the Pi CLI — confirm against its schema before relying on it.
_PI_REASONING_MODEL_MARKERS: tuple[str, ...] = ("glm", "deepseek")

# The static Claude / GPT catalogs are NOT duplicated here: they live in their
# shared home, ``omnigent.onboarding.databricks_config`` (imported above; the
# in-process harness ``inner/pi_executor`` renders the same lists — single
# source of truth). The only ids with no existing home are the non-Claude
# models #2575 establishes on the gateway (the harness's catch-all list is
# deliberately empty because these ids churn): registered here, in the same
# entry shape the shared lists use, until a live catalog (cf. #2369) replaces
# static seeding.
_DATABRICKS_PI_EXTRA_COMPLETIONS_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "databricks-glm-5-2", "input": ["text", "image"]},
    {"id": "databricks-gemini-3-5-flash", "input": ["text", "image"]},
)
# Launch id for the OpenAI-completions surface when a session boots on Claude:
# that provider is the (launch-inert) companion then, but still needs a concrete
# ``model`` — a stable id from the shared GPT list.
_DATABRICKS_PI_OPENAI_DEFAULT_MODEL = "databricks-gpt-5-5"

# Trusted parent domain suffixes for a Databricks-owned host. The AI Gateway
# lives under a per-workspace subdomain of one of these (the canonical form is
# ``<workspace>.ai-gateway.cloud.databricks.com``); the Azure / GCP control
# planes serve workspaces under their own parent domains. We anchor on the
# leading "." so a look-alike like ``...cloud.databricks.com.evil.test`` (which
# ends in ``.evil.test``) is rejected.
_DATABRICKS_TRUSTED_HOST_SUFFIXES = (
    ".cloud.databricks.com",  # AWS workspaces + ai-gateway (incl. *.staging.cloud.databricks.com)
    ".azuredatabricks.net",  # Azure Databricks
    ".gcp.databricks.com",  # GCP Databricks
)

# A genuine AI Gateway host carries the ``ai-gateway`` DNS label; we require it
# (alongside a trusted suffix) so a non-gateway Databricks host isn't routed as
# the gateway's Anthropic surface.
_DATABRICKS_AI_GATEWAY_LABEL = "ai-gateway"


def _is_databricks_ai_gateway_url(base_url: str) -> bool:
    """Return ``True`` only for a genuine Databricks AI Gateway base URL.

    Hardens the old substring scan over the whole base_url (scheme+host+path),
    which look-alikes such as ``https://databricks-ai-gateway.evil.test/...``,
    ``https://x.cloud.databricks.com.evil.test/...`` or
    ``https://evil.test/databricks/ai-gateway/v1`` all defeated — leaking the
    workspace bearer token to an attacker-controlled host. We parse the URL and
    validate the *hostname* (not the raw string): require an ``https`` scheme, a
    resolvable hostname carrying the ``ai-gateway`` DNS label, and a hostname
    that ends with a trusted Databricks-owned parent domain suffix.

    :param base_url: The codex provider table's ``base_url``.
    :returns: ``True`` iff the URL is an https Databricks AI Gateway endpoint.
    """
    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    hostname = hostname.lower()
    # ``ai-gateway`` must be a full DNS label, not a substring of one (so
    # ``databricks-ai-gateway.evil.test`` does not qualify on the label alone).
    labels = hostname.split(".")
    if _DATABRICKS_AI_GATEWAY_LABEL not in labels:
        return False
    return any(hostname.endswith(suffix) for suffix in _DATABRICKS_TRUSTED_HOST_SUFFIXES)


@dataclass(frozen=True)
class PiProviderConfig:
    """A resolved native-Pi provider, ready to render into ``models.json``.

    :param provider_id: Provider id used in ``models.json`` and ``--provider``.
    :param base_url: Endpoint base URL the ``pi`` CLI talks to.
    :param api: Pi API type, e.g. ``"anthropic-messages"`` or
        ``"openai-responses"``.
    :param model: Model id to select, e.g. ``"databricks-claude-sonnet-4-6"``.
    :param api_key: Credential value for ``models.json`` ``apiKey`` — a literal
        key, an env-var name, or a ``"!command"`` shell form (resolved by Pi at
        request time, used for short-lived gateway tokens).
    :param auth_header: When ``True``, Pi sends ``Authorization: Bearer
        <apiKey>`` (gateways) instead of a provider-native key header.
    :param compat: Pi ``compat`` flags for the block (tuple of ``(name, bool)``
        pairs); ``()`` for providers that need none.
    :param image_input: When ``True``, declare ``input: ["text", "image"]`` on
        every model entry.
    :param reasoning: When ``True``, flag reasoning-capable ids (GLM/DeepSeek)
        in this provider's catalog with ``reasoning: true``.
    :param catalog: Pi model entries (``{"id": ..., ...}`` mappings, the same
        shape the harness lists use) to advertise under this provider; the
        launch ``model`` is appended when absent. Empty means just ``model``.
    :param companions: Extra provider blocks to register alongside this one
        (for model discovery); they never drive launch selection.
    """

    provider_id: str
    base_url: str
    api: str
    model: str
    api_key: str
    auth_header: bool
    # ``compat`` block for the Databricks OpenAI-completions surface (empty for
    # every other provider). Tuple-of-pairs to keep the frozen dataclass
    # hashable; rendered back to a dict in ``_render_block``.
    compat: tuple[tuple[str, bool], ...] = ()
    # Declare image input on the model entry (mirrors the harness path; avoids
    # Pi silently dropping attached images for a dynamically-registered id, #515).
    image_input: bool = False
    # When ``True``, flag any reasoning-capable id (GLM/DeepSeek) in this
    # provider's catalog with ``reasoning: true`` — Databricks serving-endpoints
    # streams their chain-of-thought on ``reasoning_content`` (#2575). Left off
    # for vendor-direct providers, which make no such assumption.
    reasoning: bool = False
    # Model entries to list under this provider (Pi's /model picker shows exactly
    # the registered ids). Reused verbatim from the harness's static lists, so
    # they carry the curated metadata (contextWindow/maxTokens/name/input).
    # Empty → just this provider's launch ``model``.
    catalog: tuple[dict[str, Any], ...] = ()
    # Additional provider blocks to also register (e.g. the Anthropic surface
    # registered alongside the OpenAI one so Pi's ``/model`` picker offers both
    # families). Companions are launch-inert — only the top-level
    # ``provider_id``/``model`` drive ``--provider``/``--model`` selection.
    companions: tuple[PiProviderConfig, ...] = ()

    def _render_block(self) -> dict[str, Any]:
        """Render this single provider as one Pi ``models.json`` provider block."""
        # Copy each catalog entry so flagging below never mutates the shared
        # module-level lists (the harness's "rebind, don't append" concern).
        models: list[dict[str, Any]] = [dict(entry) for entry in self.catalog]
        if not any(entry.get("id") == self.model for entry in models):
            # Pi only accepts a ``provider/<id>`` selector whose id is
            # registered under that provider, so the launch model must appear
            # even when it's newer than the static catalog. Declare image input
            # for the dynamically-registered id (mirrors the harness, #515).
            entry = {"id": self.model}
            if self.image_input:
                entry["input"] = ["text", "image"]
            models.append(entry)
        if self.reasoning:
            for entry in models:
                if _is_reasoning_model(str(entry.get("id", ""))):
                    entry.setdefault("reasoning", True)
        provider: dict[str, Any] = {
            "baseUrl": self.base_url,
            "api": self.api,
            "apiKey": self.api_key,
            "models": models,
        }
        if self.auth_header:
            provider["authHeader"] = True
        if self.compat:
            provider["compat"] = dict(self.compat)
        return provider

    def to_models_config(self) -> dict[str, Any]:
        """Render this provider (and any companions) as a Pi ``models.json`` mapping."""
        providers: dict[str, Any] = {self.provider_id: self._render_block()}
        for companion in self.companions:
            providers[companion.provider_id] = companion._render_block()
        return {"providers": providers}


def _is_claude_model(model: str) -> bool:
    """Return whether *model* is a Claude-family id (routes to Anthropic Messages).

    Databricks gateway ids keep their mechanical ``databricks-`` prefix on this
    path (unlike the vendor-direct inline path), so ``databricks-claude-*`` and a
    bare ``claude-*`` both match, while ``databricks-glm-5-2`` /
    ``databricks-gemini-*`` / ``databricks-deepseek-*`` do not.
    """
    return "claude" in model.lower()


def _is_reasoning_model(model: str) -> bool:
    """Return whether *model* streams chain-of-thought on ``reasoning_content``.

    GLM and DeepSeek endpoints on the Databricks gateway emit reasoning on
    ``reasoning_content`` rather than ``content``; Pi needs ``reasoning: true``
    on the model entry to surface it (#2575).
    """
    lower = model.lower()
    return any(marker in lower for marker in _PI_REASONING_MODEL_MARKERS)


def _databricks_gateway_pi_provider(
    *,
    anthropic_base_url: str,
    openai_base_url: str | None,
    model: str | None,
    api_key: str,
) -> PiProviderConfig:
    """Build a family-aware Pi provider for a Databricks AI Gateway.

    The gateway serves Claude on the Anthropic Messages surface and non-Claude
    models (GLM, Gemini, DeepSeek, …) on the OpenAI-compatible serving-endpoints
    surface. The old code hardcoded ``anthropic-messages`` for every model, so a
    non-Claude selection hung on the Anthropic surface (#2575). Route by the
    resolved model's family:

    * Both surfaces are registered whenever the gateway exposes them, so Pi's
      ``/model`` picker offers every family regardless of the launch model. The
      *launched* (primary) provider is chosen by the resolved model's family:
      Claude → Anthropic Messages; non-Claude → OpenAI-completions on
      serving-endpoints (with the Databricks ``compat`` flags; ``reasoning`` is
      derived per model id). The other family rides along as a launch-inert
      companion, each provider carrying its family catalog for discovery.

    When *openai_base_url* is ``None`` the gateway shape exposes no reachable
    serving-endpoints surface (today: the cli-config ai-gateway subdomain, whose
    workspace serving-endpoints host isn't safely derivable from that base URL).
    A non-Claude selection then keeps the prior behavior — the resolved id is
    sent to the Anthropic surface, where it errors loudly — rather than silently
    swapping to a Claude model. Tracked as a #2575 follow-up.

    :param anthropic_base_url: The gateway's Anthropic Messages base URL.
    :param openai_base_url: The gateway's OpenAI-completions base URL, or
        ``None`` when this gateway shape has no reachable one.
    :param model: Session model override, or ``None`` to use the default.
    :param api_key: The ``!command`` bearer-token apiKey (shared by both
        surfaces of the same gateway).
    :returns: The resolved family-aware Pi provider config.
    """
    resolved = model or _DATABRICKS_PI_DEFAULT_MODEL
    resolved_is_claude = _is_claude_model(resolved)

    # No reachable OpenAI-completions surface (today: cli-config's ai-gateway
    # subdomain). Keep the prior single-provider Anthropic behavior — a
    # non-Claude id is honored (and errors loud) rather than silently swapped.
    if openai_base_url is None:
        if not resolved_is_claude:
            _LOGGER.warning(
                "pi-native: non-Claude model %r selected on a Databricks gateway with no "
                "reachable OpenAI-completions surface; Pi will use the Anthropic surface "
                "(unsupported for this model). Tracked as a #2575 follow-up.",
                resolved,
            )
        return PiProviderConfig(
            provider_id=_PI_PROVIDER_ID,
            base_url=anthropic_base_url,
            api="anthropic-messages",
            model=resolved,
            api_key=api_key,
            auth_header=True,
        )

    # Both surfaces reachable: register BOTH so Pi's /model picker offers every
    # family no matter which one this session launches with. The launched
    # (primary) provider is chosen by the resolved model's family; the other is
    # a launch-inert companion. Each provider carries its family catalog
    # (``_render_block`` appends the launch model when it's outside the list) —
    # the shared curated lists from ``onboarding.databricks_config``, so the
    # same ids and metadata (contextWindow / maxTokens / display name / image
    # input) the in-process harness renders.
    anthropic_model = resolved if resolved_is_claude else _DATABRICKS_PI_DEFAULT_MODEL
    anthropic = PiProviderConfig(
        provider_id=_PI_PROVIDER_ID,
        base_url=anthropic_base_url,
        api="anthropic-messages",
        model=anthropic_model,
        api_key=api_key,
        auth_header=True,
        image_input=True,
        catalog=tuple(DATABRICKS_ANTHROPIC_MODELS),
    )
    openai_model = resolved if not resolved_is_claude else _DATABRICKS_PI_OPENAI_DEFAULT_MODEL
    openai = PiProviderConfig(
        provider_id=_PI_OPENAI_PROVIDER_ID,
        base_url=openai_base_url,
        api="openai-completions",
        model=openai_model,
        api_key=api_key,
        auth_header=False,
        compat=_DATABRICKS_OPENAI_COMPAT,
        image_input=True,
        reasoning=True,
        catalog=(*DATABRICKS_GPT_MODELS, *_DATABRICKS_PI_EXTRA_COMPLETIONS_MODELS),
    )
    if resolved_is_claude:
        return replace(anthropic, companions=(openai,))
    return replace(openai, companions=(anthropic,))


def _databricks_pi_provider(entry: ProviderEntry, *, model: str | None) -> PiProviderConfig | None:
    """Resolve a Databricks-profile provider into Pi gateway config.

    :param entry: The resolved default provider entry (``kind="databricks"``).
    :param model: Session model override, or ``None`` to use the default.
    :returns: The Pi provider config, or ``None`` when the profile's host
        can't be resolved (caller falls back to Pi's own login).
    """
    # Imported lazily: codex_executor pulls in heavy inner deps, and this
    # module is imported on the runner's session-create path.
    from omnigent.inner.codex_executor import _databricks_codex_auth_command
    from omnigent.inner.databricks_executor import _read_databrickscfg_host

    host = _read_databrickscfg_host(entry.profile)
    if not host:
        return None
    host = host.rstrip("/")
    auth_command = _databricks_codex_auth_command(host, entry.profile)
    # Pi resolves a "!command" apiKey at request time, so the gateway bearer
    # token is refreshed per request (the auth command itself force-refreshes),
    # matching codex-native's refresh semantics. The ``databricks`` kind knows
    # the workspace host, so both the Anthropic (/ai-gateway/anthropic) and the
    # OpenAI-completions (/serving-endpoints) surfaces are reachable.
    return _databricks_gateway_pi_provider(
        anthropic_base_url=f"{host}{_DATABRICKS_ANTHROPIC_GATEWAY_PATH}",
        openai_base_url=f"{host}{_DATABRICKS_SERVING_ENDPOINTS_PATH}",
        model=model,
        api_key=f"!{auth_command}",
    )


def _gateway_anthropic_base_url(codex_base_url: str) -> str:
    """Rewrite a Codex gateway base URL to the Anthropic Messages surface.

    The Databricks AI Gateway serves each protocol under the same workspace
    origin: ``.../codex/v1`` (OpenAI Responses) and ``.../anthropic``
    (Anthropic Messages). ``isaac configure codex`` records the Codex URL;
    Pi speaks Anthropic Messages natively, so we point it at ``/anthropic``.

    :param codex_base_url: The provider table's ``base_url``, e.g.
        ``"https://<workspace>.ai-gateway.cloud.databricks.com/codex/v1"``.
    :returns: The Anthropic-surface base URL, e.g.
        ``"https://<workspace>.ai-gateway.cloud.databricks.com/anthropic"``.
    """
    trimmed = codex_base_url.rstrip("/")
    if trimmed.endswith(_DATABRICKS_GATEWAY_CODEX_SUFFIX):
        trimmed = trimmed[: -len(_DATABRICKS_GATEWAY_CODEX_SUFFIX)]
    if trimmed.endswith(_DATABRICKS_GATEWAY_ANTHROPIC_SUFFIX):
        return trimmed
    return f"{trimmed}{_DATABRICKS_GATEWAY_ANTHROPIC_SUFFIX}"


def _cli_config_databricks_transport(entry: ProviderEntry) -> CodexConfigTransport | None:
    """Return the codex transport for a pi-consumable Databricks cli-config entry.

    Shared core of :func:`_cli_config_pi_provider` and
    :func:`cli_config_pi_provider_capable`: validates that *entry* is a codex
    ``cli-config`` whose pinned ``[model_providers.X]`` table in
    ``~/.codex/config.toml`` is a genuine Databricks AI Gateway carrying a
    bearer-token command. Returns the resolved
    :class:`~omnigent.onboarding.ambient.CodexConfigTransport` when so, else
    ``None`` (logging the reason at INFO).

    :param entry: The provider entry (expected ``kind="cli-config"``).
    :returns: The codex transport when *entry* is a pi-consumable Databricks
        AI Gateway, else ``None``.
    """
    # Only codex cli-config providers are model_provider-shaped today; a
    # claude analog would be a different mechanism entirely.
    if entry.cli != "codex" or not entry.model_provider:
        return None
    # Imported lazily: ambient pulls in onboarding-only deps, and this module
    # is imported on the runner's session-create hot path.
    from omnigent.onboarding.ambient import (
        _codex_config_path,
        codex_config_provider_transport,
    )

    transport = codex_config_provider_transport(_codex_config_path(), entry.model_provider)
    if transport is None:
        _LOGGER.info(
            "pi-native: cli-config provider %r (model_provider %r) has no resolvable "
            "[model_providers.%s] base_url in ~/.codex/config.toml; Pi will use its own login.",
            entry.name,
            entry.model_provider,
            entry.model_provider,
        )
        return None
    # Identify the Databricks AI Gateway robustly (not by workspace id): parse
    # the codex base_url and validate its *hostname* against a trusted
    # Databricks domain suffix allowlist plus the ``ai-gateway`` DNS label — a
    # substring scan over the whole base_url would forward the workspace bearer
    # token to look-alike hosts (e.g. ``databricks-ai-gateway.evil.test``).
    if not _is_databricks_ai_gateway_url(transport.base_url):
        _LOGGER.info(
            "pi-native: cli-config provider %r (model_provider %r, base_url %r) is not a "
            "recognized Databricks AI Gateway; Pi will use its own login.",
            entry.name,
            entry.model_provider,
            transport.base_url,
        )
        return None
    if not transport.auth_command:
        _LOGGER.info(
            "pi-native: Databricks cli-config provider %r carries no [model_providers.%s.auth] "
            "token command; Pi will use its own login.",
            entry.name,
            entry.model_provider,
        )
        return None
    return transport


def cli_config_pi_provider_capable(entry: ProviderEntry) -> bool:
    """Return whether a ``cli-config`` *entry* is pi-consumable.

    A codex ``cli-config`` provider IS reusable by Pi exactly when
    :func:`_cli_config_pi_provider` would resolve — i.e. its pinned
    ``[model_providers.X]`` table is a genuine Databricks AI Gateway with a
    bearer-token command. This is the capability predicate the selection layer
    (:mod:`omnigent.onboarding.provider_config`) consults to decide whether a
    cli-config provider may serve / default the ``pi`` surface, keeping the
    single source of truth here (and avoiding an import cycle —
    ``provider_config`` lazy-imports this rather than the reverse).

    :param entry: The provider entry to classify (expected
        ``kind="cli-config"``; any other kind returns ``False``).
    :returns: ``True`` iff Pi can route through this cli-config provider.
    """
    return _cli_config_databricks_transport(entry) is not None


def _cli_config_pi_provider(entry: ProviderEntry, *, model: str | None) -> PiProviderConfig | None:
    """Resolve a Codex ``cli-config`` Databricks-gateway provider into Pi config.

    The common enterprise setup: ``isaac configure codex`` writes a custom
    ``[model_providers.X]`` table (base_url + token-printing ``auth`` command)
    into ``~/.codex/config.toml`` and ``omnigent setup`` adopts it as a
    ``cli-config`` provider. Codex-native routes through that table; pi-native
    used to return ``None`` here — silently falling back to Pi's own
    ``/login`` (often stale creds) — which is the bug this fixes.

    We read the *transport* (base URL + bearer-token command) from the codex
    config table the entry pins, rewrite the base URL to the gateway's
    Anthropic Messages surface (Pi speaks it natively), and emit a ``!command``
    apiKey so Pi refreshes the gateway token per request — exactly like the
    ``databricks`` kind path. The workspace-specific base URL and token path
    are read from config, never hardcoded.

    :param entry: The resolved default provider (``kind="cli-config"``,
        ``cli="codex"``), carrying the ``model_provider`` id and display name.
    :param model: Session model override, or ``None`` to use the default.
    :returns: The Pi provider config, or ``None`` when the entry is not a
        Databricks gateway, its codex provider table can't be resolved, or it
        carries no token command (caller falls back to Pi's own login).
    """
    transport = _cli_config_databricks_transport(entry)
    if transport is None:
        return None
    # Pi resolves a "!command" apiKey at request time, so the gateway bearer
    # token (the codex auth command prints it) is refreshed per request —
    # matching codex-native's refresh semantics. ``openai_base_url=None``: the
    # cli-config gateway is the ai-gateway *subdomain*, whose OpenAI
    # serving-endpoints host isn't safely derivable from that base URL, so
    # non-Claude routing is deferred (a #2575 follow-up). Claude is unaffected —
    # it uses the Anthropic surface either way.
    return _databricks_gateway_pi_provider(
        anthropic_base_url=_gateway_anthropic_base_url(transport.base_url),
        openai_base_url=None,
        model=model,
        api_key=f"!{transport.auth_command}",
    )


def _inline_family_pi_provider(
    entry: ProviderEntry, *, model: str | None
) -> PiProviderConfig | None:
    """Resolve a key/gateway/local provider into Pi config from its family.

    Prefers the Anthropic family (Pi speaks ``anthropic-messages`` natively),
    falling back to the OpenAI family via the Responses API.

    :param entry: The resolved default provider entry.
    :param model: Session model override, or ``None`` to use the family default.
    :returns: The Pi provider config, or ``None`` when no usable family with a
        base URL and credential is configured.
    """
    for family_name in ("anthropic", "openai"):
        family = entry.family(family_name)
        if family is None or not family.base_url:
            continue
        # Determine the API type based on family and wire_api setting.
        if family_name == "anthropic":
            api = "anthropic-messages"
        elif family.wire_api == CHAT_WIRE_API:
            api = "openai-completions"
        else:
            api = "openai-responses"
        # A static key (or $VAR) — Pi reads a literal/env apiKey directly; an
        # auth_command becomes a "!command" Pi resolves at request time.
        if family.api_key:
            api_key = family.api_key
            auth_header = False
        elif family.auth_command:
            api_key = f"!{family.auth_command}"
            auth_header = True
        else:
            continue
        resolved_model = model or entry.family_default_model(family_name)
        if not resolved_model:
            continue
        # A session model override can arrive as a Databricks-gateway id
        # (``databricks-claude-opus-4-7``) — that prefix only routes through the
        # Databricks AI Gateway (``_databricks_pi_provider``). This family is
        # vendor-direct (key / inline gateway / local Anthropic|OpenAI endpoint),
        # so strip the mechanical ``databricks-`` prefix to the bare vendor id
        # the endpoint can actually route. ``normalize_model_for_provider`` is
        # prefix-mechanical: it only strips ``databricks-claude-*``/
        # ``databricks-gpt-*`` and passes non-mechanical ids (e.g.
        # ``zai-org/GLM-4.7``) and already-bare ids through unchanged. Family
        # defaults are bare, so the no-override path is unaffected.
        resolved_model = normalize_model_for_provider(resolved_model, KEY_KIND)
        return PiProviderConfig(
            provider_id=_PI_PROVIDER_ID,
            base_url=family.base_url,
            api=api,
            model=resolved_model,
            api_key=api_key,
            auth_header=auth_header,
        )
    return None


def resolve_pi_native_provider(
    *,
    model: str | None = None,
    config_loader: Callable[[], dict[str, Any]] = load_config,
) -> PiProviderConfig | None:
    """Resolve the omnigent-configured provider for a native Pi session.

    Reads the default provider for the Pi surface from
    ``~/.omnigent/config.yaml`` and translates it into Pi ``models.json``
    config. Returns ``None`` — leaving Pi to use its own ``/login`` — when no
    usable provider is configured, or the default is a subscription / CLI-login
    provider (a CLI's own login can't be reused outside that CLI).

    :param model: Session model override (``model_override``), or ``None`` to
        use the provider's default model.
    :param config_loader: Injection seam for tests; defaults to
        :func:`load_config`.
    :returns: The resolved provider config, or ``None`` to fall back to Pi's
        own credentials.
    """
    try:
        config = config_loader()
        # Pi is multi-family; ``omnigent setup`` marks defaults per family, not
        # for ``pi``. Use the shared house-pattern selection so pi resolves its
        # default exactly like the rest of the codebase — an explicit pi default
        # wins, else the anthropic (Pi's native surface) then openai family
        # default, skipping kinds that can't drive pi. Crucially this now lets a
        # cli-config Databricks AI Gateway through (it is pi-consumable via
        # ``_cli_config_pi_provider``), so an unrelated anthropic-family default
        # no longer shadows it.
        entry = default_provider_for_harness(config, PI_SURFACE)
        if entry is None:
            _LOGGER.info(
                "pi-native: no omnigent-configured provider for the pi/anthropic/openai "
                "surface; Pi will use its own login."
            )
            return None
        if entry.kind == DATABRICKS_KIND:
            resolved = _databricks_pi_provider(entry, model=model)
        elif entry.kind == CLI_CONFIG_KIND:
            # A Codex cli-config provider whose [model_providers.X] table is the
            # Databricks AI Gateway IS reusable by Pi (the gateway exposes an
            # Anthropic surface Pi speaks). Translate it rather than dropping to
            # Pi's own login — the bug this module fixes.
            resolved = _cli_config_pi_provider(entry, model=model)
        elif entry.kind in (KEY_KIND, GATEWAY_KIND, LOCAL_KIND):
            resolved = _inline_family_pi_provider(entry, model=model)
        else:
            # subscription (a CLI's own login can't be reused outside that CLI):
            # let Pi use its own login.
            _LOGGER.info(
                "pi-native: configured provider %r (kind %r) cannot drive Pi; "
                "Pi will use its own login.",
                entry.name,
                entry.kind,
            )
            return None
        if resolved is None:
            # The provider matched a translatable kind but its details could not
            # be resolved (e.g. a Databricks gateway whose codex config table is
            # missing). Don't swallow it silently — a future user mystified by an
            # "OpenRouter auth error despite configuring Databricks" needs this.
            _LOGGER.warning(
                "pi-native: configured provider %r (kind %r) could not be translated "
                "into native Pi config; Pi will use its own login (which may hold "
                "unrelated/stale credentials).",
                entry.name,
                entry.kind,
            )
        return resolved
    except Exception:  # noqa: BLE001 — any resolution failure must not break launch
        # Any failure (malformed config, duplicate per-family default, or an
        # unresolved ``api_key: $VAR``) falls back to Pi's own login rather than
        # failing the terminal launch.
        _LOGGER.warning(
            "pi-native: failed to resolve the omnigent-configured provider; Pi will "
            "use its own login.",
            exc_info=True,
        )
        return None


def write_pi_models_config(agent_dir: Path, provider: PiProviderConfig) -> Path:
    """Write *provider* as ``models.json`` into a managed Pi config dir.

    :param agent_dir: The managed Pi config dir (``PI_CODING_AGENT_DIR``).
    :param provider: The resolved provider config to render.
    :returns: Path to the written ``models.json``.
    """
    agent_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(agent_dir, 0o700)
    models_path = agent_dir / "models.json"
    # 0o600: the apiKey may be a literal token (key-kind providers).
    fd = os.open(models_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(provider.to_models_config(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return models_path


def pi_native_provider_launch(
    agent_dir: Path, provider: PiProviderConfig
) -> tuple[dict[str, str], list[str]]:
    """Write the managed config and return the launch env + CLI args for Pi.

    :param agent_dir: The managed Pi config dir for this session.
    :param provider: The resolved provider config.
    :returns: ``(env, args)`` — the env vars to merge into the terminal spec
        (relocating Pi's config dir) and the ``--provider``/``--model`` args to
        append to the Pi command.
    """
    write_pi_models_config(agent_dir, provider)
    env = {PI_CODING_AGENT_DIR_ENV_VAR: str(agent_dir)}
    args = ["--provider", provider.provider_id, "--model", provider.model]
    return env, args
