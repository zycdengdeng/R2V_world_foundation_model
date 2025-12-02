# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HuggingFace mirror configuration utility.

Import this module EARLY (before huggingface_hub) to configure HF mirror.

Usage:
    # At the very top of your script, before any other imports:
    import cosmos_transfer2._src.imaginaire.utils.hf_mirror  # noqa: F401

    # Or manually configure:
    from cosmos_transfer2._src.imaginaire.utils.hf_mirror import configure_hf_mirror
    configure_hf_mirror("https://hf-mirror.com")
"""

import os


def configure_hf_mirror(endpoint: str = "https://hf-mirror.com") -> None:
    """Configure HuggingFace to use a mirror endpoint.

    This function MUST be called BEFORE importing huggingface_hub to work correctly.

    Args:
        endpoint: The mirror endpoint URL. Defaults to hf-mirror.com for China.
    """
    # Set all possible environment variables for HF endpoint
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_URL"] = endpoint
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint

    # Try to patch huggingface_hub if already imported
    try:
        import huggingface_hub.constants as hf_constants

        # Patch the constants directly
        if hasattr(hf_constants, "ENDPOINT"):
            hf_constants.ENDPOINT = endpoint
        if hasattr(hf_constants, "HF_HUB_URL"):
            hf_constants.HF_HUB_URL = endpoint
        if hasattr(hf_constants, "_CACHED_NO_TOKEN"):
            # Force re-evaluation of endpoint
            hf_constants._CACHED_NO_TOKEN = None

        print(f"[hf_mirror] Configured HuggingFace endpoint: {endpoint}")
        print(f"[hf_mirror] Note: huggingface_hub was already imported, patched constants directly")

    except ImportError:
        # huggingface_hub not yet imported - environment variables will be used
        print(f"[hf_mirror] Set HF_ENDPOINT={endpoint} (huggingface_hub not yet imported)")


# Auto-configure if HF_ENDPOINT is set
_default_endpoint = os.environ.get("HF_ENDPOINT")
if _default_endpoint:
    configure_hf_mirror(_default_endpoint)
