import os
import sys
import pytest
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import tensorflow as tf
import keras

from bll.bayesianlastlayer import BayesianLastLayer as BayesianLastLayerTF
import bll.tools as tools


from bll_pytorch.bayesianlastlayer import BayesianLastLayer as BayesianLastLayerPT
from bll_pytorch.torch_model import JointModel, sin_activation
import bll_pytorch.tools as tools_pt

seed = 42
n_in = 1
n_out = 2


# @pytest.fixture
# def random_data():
#     rng = np.random.default_rng(0)
# 
#     x = rng.normal(loc=5, scale=2, size=(50, n_in))
#     y = rng.normal(loc=-3, scale=5, size=(50, n_out))
# 
#     return x, y

@pytest.fixture
def random_data():
    function_types = [2, 0]
    sigma_noise = [2e-2, 5e-1]
    n_channels = len(function_types)

    x, y = tools.get_data(50,[0,1], function_type=function_types, sigma=sigma_noise, dtype='float64', random_seed=seed)
    return x, y

@pytest.fixture
def Scaler_numpy(random_data):
    x, y = random_data
    scaler = tools.Scaler(x, y)
    return scaler

@pytest.fixture
def Scaler_pytorch(random_data):
    x, y = random_data
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    scaler = tools_pt.Scaler(x_tensor, y_tensor)
    return scaler


@pytest.fixture
def BLL_tensorflow(Scaler_numpy):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_input = keras.Input(shape=(n_in,))

    initializer = tf.keras.initializers.Constant(value=1)
    # Hidden units
    architecture = [
    (keras.layers.Dense, {'units': 20, 'activation': tf.math.sin, 'name': '01_dense', 'kernel_initializer': initializer, "bias_initializer": initializer}),
    (keras.layers.Dense, {'units': 20, 'activation': tf.nn.tanh, 'name': '02_dense', 'kernel_initializer': initializer, "bias_initializer": initializer}),
    (keras.layers.Dense, {'name': 'output', 'units': n_out, 'kernel_initializer': initializer, "bias_initializer": initializer})
    ]

    # Get layers and outputs:
    model_layers, model_outputs = tools.DNN_from_architecture(model_input, architecture)
    output_model = keras.Model(inputs=model_input, outputs=model_outputs[-1])
    joint_model = keras.Model(model_input, [model_outputs[-2], model_outputs[-1]])

    bll = BayesianLastLayerTF(joint_model, Scaler_numpy)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
    bll.setup_training(optimizer)
    return bll

@pytest.fixture
def BLL_pytorch(Scaler_pytorch):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hidden units
    architecture = [
        (torch.nn.Linear, {'in_features': n_in, 'out_features': 20}),
        (sin_activation, {}),
        (torch.nn.Linear, {'in_features': 20, 'out_features': 20}),
        (torch.nn.Tanh, {}),
        (torch.nn.Linear, {'in_features': 20, 'out_features': n_out})
    ]

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 1.0)


    model = JointModel(architecture)
    model.apply(init_weights)
    bll = BayesianLastLayerPT(model, Scaler_pytorch)

    optimizer = torch.optim.Adam
    bll.setup_training(optimizer, optimizer_kwargs={'lr':5e-3})
    return bll


def test_log_sigma_w(BLL_tensorflow, BLL_pytorch):
    log_sigma_w_tf = BLL_tensorflow.log_sigma_w.numpy()
    log_sigma_w_pt = BLL_pytorch.log_sigma_w.detach().numpy()

    assert np.allclose(log_sigma_w_tf, log_sigma_w_pt, atol=1e-5)

def test_log_sigma_w_set(BLL_tensorflow, BLL_pytorch):
    new_value = np.array([-np.pi])

    BLL_tensorflow.log_sigma_w = new_value
    BLL_pytorch.log_sigma_w = torch.tensor(new_value, dtype=torch.float64)

    log_sigma_w_tf = BLL_tensorflow.log_sigma_w.numpy()
    log_sigma_w_pt = BLL_pytorch.log_sigma_w.detach().numpy()

    assert np.allclose(log_sigma_w_tf, log_sigma_w_pt, atol=1e-5)

def test_log_alpha(BLL_tensorflow, BLL_pytorch):
    log_alpha_tf = BLL_tensorflow.log_alpha.numpy()
    log_alpha_pt = BLL_pytorch.log_alpha.detach().numpy()

    assert np.allclose(log_alpha_tf, log_alpha_pt, atol=1e-5)

def test_Lambda_p_bar(BLL_tensorflow, BLL_pytorch):
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, BLL_tensorflow.n_phi)).astype(np.float32)
    phi_torch = torch.tensor(phi, requires_grad=False)

    Lambda_p_bar_tf = BLL_tensorflow.get_Lambda_p_bar(phi).numpy()
    Lambda_p_bar_pt = BLL_pytorch.get_Lambda_p_bar(phi_torch).detach().numpy()

    assert np.allclose(Lambda_p_bar_tf, Lambda_p_bar_pt, atol=1e-5)


def test_log_marginal_likelihood(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())
    y_torch = torch.tensor(y, requires_grad=False, dtype=torch.get_default_dtype())

    lml_tf = BLL_tensorflow.lml(x, y).numpy()
    lml_pt = BLL_pytorch.lml(x_torch, y_torch).detach().numpy()

    assert np.allclose(lml_tf, lml_pt, atol=1e-5)

def test_exact_log_marginal_likelihood(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())
    y_torch = torch.tensor(y, requires_grad=False, dtype=torch.get_default_dtype())

    lml_tf = BLL_tensorflow.exact_lml(x, y).numpy()
    lml_pt = BLL_pytorch.exact_lml(x_torch, y_torch).detach().numpy()

    assert np.allclose(lml_tf, lml_pt, atol=1e-5)

def test_prepare_prediction(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    #x = x.astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())

    BLL_tensorflow.prepare_prediction(x)
    BLL_pytorch.prepare_prediction(x_torch)

    phi_tf = BLL_tensorflow.Phi
    phi_pt = BLL_pytorch.Phi.detach().numpy()

    lambda_p_bar_tf = BLL_tensorflow.Lambda_p_bar
    lambda_p_bar_pt = BLL_pytorch.Lambda_p_bar.detach().numpy()

    sigma_p_bar_tf = BLL_tensorflow.Sigma_p_bar
    sigma_p_bar_pt = BLL_pytorch.Sigma_p_bar.detach().numpy()

    sigma_e_tf = BLL_tensorflow.Sigma_e
    sigma_e_pt = BLL_pytorch.Sigma_e.detach().numpy()

    assert np.allclose(phi_tf, phi_pt, atol=1e-5)
    assert np.allclose(lambda_p_bar_tf, lambda_p_bar_pt)
    assert np.allclose(sigma_p_bar_tf, sigma_p_bar_pt, atol=1e-5)
    assert np.allclose(sigma_e_tf, sigma_e_pt, atol=1e-5)

def test_predict(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x = x.astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())

    BLL_tensorflow.prepare_prediction(x)
    BLL_pytorch.prepare_prediction(x_torch)

    # note: use return_scaled=True to avoid scaling differences
    # due to different scaler implementations
    mu_tf, var_tf = BLL_tensorflow.predict(x, return_scaled=True)
    mu_pt, var_pt = BLL_pytorch.predict(x_torch, return_scaled=True)

    mu_tf = mu_tf
    var_tf = var_tf
    mu_pt = mu_pt.detach().numpy()
    var_pt = var_pt.detach().numpy()


    assert np.allclose(mu_tf, mu_pt, atol=1e-5) 
    assert np.allclose(var_tf, var_pt, atol=1e-4) # tolerance increased due to small numerical differences in variance


def test_mean(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x = x.astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())

    BLL_tensorflow.prepare_prediction(x)
    BLL_pytorch.prepare_prediction(x_torch)

    # note: use return_scaled=True to avoid scaling differences
    # due to different scaler implementations
    mu_tf, phi_tf = BLL_tensorflow.mean(x, return_scaled=True)
    mu_pt, phi_pt = BLL_pytorch.mean(x_torch, return_scaled=True)

    mu_pt = mu_pt.detach().numpy()
    phi_pt = phi_pt.detach().numpy()

    assert np.allclose(mu_tf, mu_pt, atol=1e-5)
    assert np.allclose(phi_tf, phi_pt, atol=1e-5)

def test_mse(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())
    y_torch = torch.tensor(y, requires_grad=False, dtype=torch.get_default_dtype())

    BLL_tensorflow.prepare_prediction(x)
    BLL_pytorch.prepare_prediction(x_torch)

    mse_tf = BLL_tensorflow.mse(x, y)
    mse_pt = BLL_pytorch.mse(x_torch, y_torch).detach().numpy()

    assert np.allclose(mse_tf, mse_pt, atol=1e-5)

def test_std(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())
    y_torch = torch.tensor(y, requires_grad=False, dtype=torch.get_default_dtype())

    BLL_tensorflow.prepare_prediction(x)
    BLL_pytorch.prepare_prediction(x_torch)

    cov = BLL_tensorflow.predict(x)[1]

    std_tf = BLL_tensorflow.std(cov)
    std_pt = BLL_pytorch.std(torch.tensor(cov)).detach().numpy()

    assert np.allclose(std_tf, std_pt, atol=1e-5)

def test_lpd(BLL_tensorflow, BLL_pytorch, random_data):
    x, y = random_data

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=False, dtype=torch.get_default_dtype())
    y_torch = torch.tensor(y, requires_grad=False, dtype=torch.get_default_dtype())

    BLL_tensorflow.prepare_prediction(x)
    BLL_pytorch.prepare_prediction(x_torch)

    for aggregation in ['none', 'mean', 'median']:
        lpd_tf = BLL_tensorflow.lpd(x, y, aggregate=aggregation)
        lpd_pt = BLL_pytorch.lpd(x_torch, y_torch, aggregate=aggregation).detach().numpy()
        assert np.allclose(lpd_tf, lpd_pt, atol=1e-5)