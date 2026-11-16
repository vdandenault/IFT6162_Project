import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_ontario_stage_costs(stage_costs, reference_stage_costs):
    assert stage_costs.shape == (2880, 4)
    assert reference_stage_costs.shape == (2880, 4)
    np.testing.assert_array_equal(stage_costs, reference_stage_costs, strict=True)
