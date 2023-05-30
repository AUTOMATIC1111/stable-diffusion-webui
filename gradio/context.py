# Defines the Context class, which is used to store the state of all Blocks that are being rendered.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from gradio.blocks import BlockContext, Blocks


class Context:
    root_block: Blocks | None = None  # The current root block that holds all blocks.
    block: BlockContext | None = None  # The current block that children are added to.
    id: int = 0  # Running id to uniquely refer to any block that gets defined
    ip_address: str | None = None  # The IP address of the user.
    hf_token: str | None = None  # The token provided when loading private HF repos
