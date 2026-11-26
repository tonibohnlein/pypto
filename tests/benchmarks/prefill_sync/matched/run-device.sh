#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Run in a pinned 7178c3bc2 PyPTO 910B2 environment using the Run J toolchain.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
output=${1:?Supply a new output directory}
card=${2:?Supply a device id}
mkdir "$output"
for rows in 48 80; do
    for arm in B R; do
        python "$here/matrix_device.py" --arm "$arm" --rows "$rows" --device "$card" --launches 100 --out "$output/$rows-$arm"
        python - "$here/expected-matrix.json" "$output/$rows-$arm/result.json" <<'PY'
import json
import sys
from pathlib import Path

expected, result = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
ref, = [r for r in expected if (r['rows'], r['arm']) == (result['rows'], result['arm'])]
if result['artefacts']['sources']['ptoas/kernel.cpp']['sha256'] != ref['cpp_sha256']:
    raise SystemExit('Compiled source differs from the pinned host build')
if result['exact_matches'] != 100 or result['failures']:
    raise SystemExit('Matrix exact comparison failed')
PY
    done
done
