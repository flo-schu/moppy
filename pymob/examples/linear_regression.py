import numpy as np

rng = np.random.default_rng(seed=1)

def linear_model(n=50):
    def model(x, a, b):
        return a + x * b

    parameters = dict(
        a=0,
        b=1,
        sigma_y=1,
    )

    x = np.linspace(-5, 5, n)
    y = model(x=x, a=parameters["a"], b=parameters["b"])
    y_noise = rng.normal(loc=y, scale=parameters["sigma_y"])

    return model, x, y, y_noise, parameters