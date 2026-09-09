from app.ibm_client import get_model


def test_ibm_model_connection():

    model = get_model()

    assert model is not None