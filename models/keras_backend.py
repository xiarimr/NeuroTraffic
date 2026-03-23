import os


def setup_keras_backend():
    """Force Keras 3 to use the torch backend unless user overrides it."""
    backend = os.environ.get("KERAS_BACKEND")
    if backend is None:
        os.environ["KERAS_BACKEND"] = "torch"
