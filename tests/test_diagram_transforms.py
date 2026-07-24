import numpy as np

from persiscope.diagram_transforms import DiagramTransform, H0Rotate, Identity, RotateScale


def test_identity_returns_same_values():
    diagram = np.array([[0.0, 1.0], [0.5, 2.0]])
    out = Identity()(diagram)
    assert np.allclose(out, diagram)


def test_rotate_scale_known_rotation():
    # Rotate (1, 0) by -90 degrees, alpha=1 -> (0, -1).
    diagram = np.array([[1.0, 0.0]])
    out = RotateScale(theta=-np.pi / 2, alpha=1.0)(diagram)
    assert np.allclose(out, [[0.0, -1.0]], atol=1e-12)


def test_rotate_scale_applies_alpha():
    diagram = np.array([[1.0, 1.0]])
    out = RotateScale(theta=0.0, alpha=0.5)(diagram)
    assert np.allclose(out, [[0.5, 0.5]])


def test_rotate_scale_empty_diagram():
    out = RotateScale()(np.zeros((0, 2)))
    assert out.shape == (0, 2)


def test_h0_rotate_identity_angle():
    # angle = pi/2 -> rotation by 0 -> unchanged.
    diagram = np.array([[0.0, 1.0], [0.3, 0.9]])
    out = H0Rotate(angle=np.pi / 2)(diagram)
    assert np.allclose(out, diagram)


def test_transforms_satisfy_protocol():
    for t in (Identity(), RotateScale(), H0Rotate()):
        assert isinstance(t, DiagramTransform)


def test_toplevel_theta_matches_explicit_transform():
    import persiscope as ps

    rng = np.random.default_rng(4)
    emb = rng.normal(size=(20, 4))
    via_theta = ps.TopologicalTransformer(
        theta=-np.pi / 4, n_bootstrap=8, random_state=0
    ).fit_transform(emb)
    via_transform = ps.TopologicalTransformer(
        diagram_transform=RotateScale(theta=-np.pi / 4), n_bootstrap=8, random_state=0
    ).fit_transform(emb)
    assert np.allclose(via_theta.mean_landscape, via_transform.mean_landscape)


def test_theta_and_transform_conflict():
    import pytest

    import persiscope as ps

    with pytest.raises(ValueError):
        ps.TopologicalTransformer(theta=-np.pi / 4, diagram_transform=RotateScale())


def test_compare_accepts_theta():
    import persiscope as ps

    rng = np.random.default_rng(5)
    embs = [rng.normal(size=(16, 4)) for _ in range(2)]
    res = ps.compare(embs, theta=-np.pi / 4, n_bootstrap=6, random_state=0)
    assert res.matrix.shape == (2, 2)
    assert "RotateScale(theta=-0.785" in res.representations[0].params["diagram_transform"]
