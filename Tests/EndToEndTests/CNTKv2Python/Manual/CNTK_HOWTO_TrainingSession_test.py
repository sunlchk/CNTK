# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path, "..", "Tutorials")

from nb_helper import get_output_stream_from_cell

notebook = os.path.join(abs_path, "..", "..", "..", "..", "Manual", "CNTK_HOWTO_Training_Session.ipynb")

def test_cntk_HOWTO_training_session_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedOutput = 'Minibatch\[ 191- 200\]: loss = '
def test_cntk_HOWTO_training_session_evalCorrect(nb):
    testCells = [cell for cell in nb.cells
                if cell.cell_type == 'code']
    assert len(testCells) == 4
    for c in testCells:
        text = get_output_stream_from_cell(c)
        print(text)
        assert re.search(expectedOutput, text)
