# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Myenv Environment.

The myenv environment simulates a simple customer support chatbot that
responds to user queries by echoing back support-style messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MyenvAction(Action):
    """Action for the Myenv environment - a customer query message."""

    message: str = Field(..., description="Customer query or support request")


class MyenvObservation(Observation):
    """Observation from the Myenv environment - the chatbot's echoed response."""

    echoed_message: str = Field(default="", description="Chatbot response to the customer query")
    message_length: int = Field(default=0, description="Length of the chatbot response")
