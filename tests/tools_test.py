import numpy as np
import torch
import pytest

from bll.tools import Scaler
from bll_pytorch.tools import Scaler as ScalerPT

@pytest.fixture
def random_data():
    rng = np.random.default_rng(0)

    x = rng.normal(loc=5, scale=2, size=(50, 10))
    y = rng.normal(loc=-3, scale=5, size=(50, 3))
    return x, y

class TestScaler:
    def test_scaler_numpy(self, random_data):
        """Test the numpy scaler."""
        x, y = random_data

        scaler = Scaler(x, y)
        x_scaled, y_scaled = scaler.scale(x, y)
        print(np.mean(x_scaled, axis=0))

        assert np.isclose(np.mean(x_scaled, axis=0), 0).all()
        assert np.isclose(np.std(x_scaled, axis=0), 1).all()

        x_inv = scaler.unscale(x_scaled)
        assert np.isclose(x, x_inv).all()

    def test_scaler_pytorch(self, random_data):
        """Test the pytorch scaler."""
        # note: not close with float32 due to precision issues
        x, y = random_data

        x_tensor = torch.tensor(x, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.float64)

        scaler = ScalerPT(x_tensor, y_tensor)
        x_scaled, y_scaled = scaler.scale(x_tensor, y_tensor)

        assert torch.isclose(torch.mean(x_scaled, dim=0), torch.zeros(x.shape[1], dtype=torch.float64)).all()
        # note: correction=0 for population std, like numpy's default
        assert torch.isclose(torch.std(x_scaled, dim=0, correction=0), torch.ones(x.shape[1], dtype=torch.float64)).all()

        x_inv, y_inv = scaler.unscale(x_scaled, y_scaled)
        assert torch.isclose(x_tensor, x_inv).all()

    def test_scalers_against_each_other(self, random_data):
        """Test that the numpy and pytorch scalers give the same results."""
        x, y = random_data

        scaler_np = Scaler(x, y)
        x_scaled_np, y_scaled_np = scaler_np.scale(x, y)
        x_inverse_np, y_inverse_np = scaler_np.unscale(x_scaled_np, y_scaled_np)

        x_tensor = torch.tensor(x, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.float64)

        scaler_pt = ScalerPT(x_tensor, y_tensor)
        x_scaled_pt, y_scaled_pt = scaler_pt.scale(x_tensor, y_tensor)
        x_inverse_pt, y_inverse_pt = scaler_pt.unscale(x_scaled_pt, y_scaled_pt)

        assert np.isclose(scaler_np.scaler_x.mean_, scaler_pt.scaler_x.mean.numpy()).all()
        assert np.isclose(scaler_np.scaler_x.scale_, scaler_pt.scaler_x.std.numpy()).all()

        assert np.isclose(x_scaled_np, x_scaled_pt.numpy()).all()
        assert np.isclose(y_scaled_np, y_scaled_pt.numpy()).all()

        assert np.isclose(x_inverse_np, x_inverse_pt.numpy()).all()
        assert np.isclose(y_inverse_np, y_inverse_pt.numpy()).all()
