# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure ordinary emission separately from explicit resource audit requests.

Run with python and tests/ut/ir on PYTHONPATH. The slices sweep varies a counted
inner extent, not static source stages. No original nest is unfolded to compile.
"""

import argparse
import json
from pathlib import Path

from pypto import backend
from pypto.backend import BackendType

from transforms.test_scoped_palette_realization import palette_scaling_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    rows = []
    sweeps = {
        "K": [(k, 4, 4) for k in (32, 320, 3200, 32000)],
        "outer_visits": [(32, visits, 4) for visits in (4, 40, 400, 4000)],
        "inner_slice_extent": [(32, 4, slices) for slices in (2, 4, 8)],
    }
    for axis, configurations in sweeps.items():
        for configuration in configurations:
            row = palette_scaling_sample(*configuration)
            row["sweep"] = axis
            rows.append(row)
            args.output.write_text(
                json.dumps(
                    {
                        "timing_scope": "single descriptive runs; stage timings and audit requests separate",
                        "static_stage_qualification": "not claimed; slice count is a loop extent",
                        "device_validation": False,
                        "rows": rows,
                    },
                    indent=2,
                    default=lambda value: getattr(value, "name", str(value)),
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
