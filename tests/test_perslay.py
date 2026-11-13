
import numpy as np
import tensorflow as tf
import gudhi
import gudhi.tensorflow.perslay as prsl


def test_perslay_forward_pass():
    diagram = np.array([
        [0., 4.],
        [1., 2.],
        [3., 8.],
        [6., 8.]
    ], dtype=np.float32)

    diagrams = tf.ragged.constant(
        [diagram.tolist()],
        dtype=tf.float32,
        ragged_rank=1
    )

    phi = prsl.GaussianPerslayPhi(
        (5, 5),
        ((-0.5, 1.5), (-0.5, 1.5)),
        0.1
    )

    weight = prsl.PowerPerslayWeight(1.0, 0.0)

    perslay = prsl.Perslay(
        weight=weight,
        phi=phi,
        perm_op=tf.math.reduce_sum,
        rho=tf.identity
    )

    output = perslay(diagrams)

    assert output.shape == (1, 5, 5, 1)
    assert np.all(np.isfinite(output.numpy()))
