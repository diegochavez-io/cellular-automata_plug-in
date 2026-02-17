#!/usr/bin/env python3
"""
Test script for smoothed parameter infrastructure.

Verifies:
1. SmoothedParameter drift behavior
2. LeniaParameterCoupler mu/sigma coupling
3. SurvivalGuardian rescue mechanism
4. Viewer integration of smoothing system
"""

import sys
import numpy as np
from cellular_automata.smoothing import SmoothedParameter, LeniaParameterCoupler, SurvivalGuardian
from cellular_automata.lenia import Lenia

def test_smoothed_parameter():
    """Test EMA drift over time."""
    print("Testing SmoothedParameter...")
    sp = SmoothedParameter(0.15, time_constant=2.0)
    sp.set_target(0.30)

    # After 1 frame at 60fps
    sp.update(1/60)
    val_1frame = sp.get_value()
    assert abs(val_1frame - 0.15) < 0.01, f"Should barely move after 1 frame: {val_1frame}"

    # After ~5 seconds (300 frames)
    for _ in range(299):
        sp.update(1/60)
    val_5s = sp.get_value()
    assert abs(val_5s - 0.30) < 0.02, f"Should be near target after 5s: {val_5s}"

    # Test snap
    sp.snap(0.5)
    assert sp.get_value() == 0.5, "Snap should set value immediately"
    assert sp.target == 0.5, "Snap should set target too"

    print("  ✓ SmoothedParameter working correctly")

def test_parameter_coupler():
    """Test mu/sigma coupling."""
    print("Testing LeniaParameterCoupler...")
    preset = {"mu": 0.12, "sigma": 0.010}
    coupler = LeniaParameterCoupler(preset)

    # Test coupling effect
    mu_target = 0.35
    sigma_target = 0.010
    coupled_mu, coupled_sigma = coupler.couple(mu_target, sigma_target)

    # Coupling should adjust values toward baseline ratio
    assert coupled_mu != mu_target or coupled_sigma != sigma_target, "Coupling should adjust values"

    # Test bounds enforcement
    mu_extreme = 0.50  # Beyond safe max (0.35)
    sigma_extreme = 0.001  # Below safe min (0.005)
    bounded_mu, bounded_sigma = coupler.couple(mu_extreme, sigma_extreme)
    assert bounded_mu <= 0.35, f"mu should be clamped to max: {bounded_mu}"
    assert bounded_sigma >= 0.005, f"sigma should be clamped to min: {bounded_sigma}"

    print("  ✓ LeniaParameterCoupler working correctly")

def test_survival_guardian():
    """Test invisible density injection."""
    print("Testing SurvivalGuardian...")
    engine = Lenia(size=256)
    engine.world[:] = 0.0001  # Very low mass (below critical threshold)
    guardian = SurvivalGuardian(engine, critical_mass=0.002, rescue_cooldown=5.0)

    # Should rescue on first check
    initial_mass = engine.get_mass()
    rescued = guardian.check_and_rescue(0.1)
    assert rescued, "Should rescue when mass below threshold"

    new_mass = engine.get_mass()
    assert new_mass > initial_mass, f"Mass should increase after rescue: {initial_mass} -> {new_mass}"

    # Should NOT rescue again immediately (cooldown)
    engine.world[:] = 0.0001  # Reset mass
    rescued_again = guardian.check_and_rescue(0.1)
    assert not rescued_again, "Should not rescue again during cooldown"

    # Should rescue after cooldown expires
    # Manually advance cooldown timer
    guardian.cooldown_timer = 0.0
    engine.world[:] = 0.0001  # Reset mass
    rescued_after_cooldown = guardian.check_and_rescue(0.1)
    assert rescued_after_cooldown, "Should rescue again after cooldown"

    print("  ✓ SurvivalGuardian working correctly")

def test_viewer_integration():
    """Test that viewer correctly integrates smoothing."""
    print("Testing Viewer integration...")
    from cellular_automata.viewer import Viewer

    # Create viewer with coral preset
    viewer = Viewer(width=512, height=512, sim_size=256, start_preset="coral")

    # Verify smoothed params were created
    assert len(viewer.smoothed_params) > 0, "Smoothed params should be created"
    assert "mu" in viewer.smoothed_params, "mu should have smoothed param"
    assert "sigma" in viewer.smoothed_params, "sigma should have smoothed param"

    # Verify coupler and guardian created for Lenia
    assert viewer.param_coupler is not None, "Parameter coupler should exist for Lenia"
    assert viewer.survival_guardian is not None, "Survival guardian should exist for Lenia"

    # Test parameter callback (should set target, not engine directly)
    callback = viewer._make_param_callback("mu")
    initial_mu = viewer.engine.get_params()["mu"]
    callback(0.20)
    # Engine value shouldn't change immediately (smoothing lag)
    immediate_mu = viewer.engine.get_params()["mu"]
    # Note: The target is set, but the value won't change until update() is called
    assert viewer.smoothed_params["mu"].target == 0.20, "Target should be set immediately"

    print("  ✓ Viewer integration working correctly")

if __name__ == "__main__":
    print("\n=== Testing Smoothed Parameter Infrastructure ===\n")

    test_smoothed_parameter()
    test_parameter_coupler()
    test_survival_guardian()
    test_viewer_integration()

    print("\n✓ All tests passed!\n")
