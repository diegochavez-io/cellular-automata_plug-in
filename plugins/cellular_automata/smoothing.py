"""
EMA-Smoothed Parameter Infrastructure

Provides three core safety mechanisms for parameter control:

1. SmoothedParameter - EMA wrapper for any numeric parameter with time-constant drift
2. LeniaParameterCoupler - Ratio-based mu/sigma coupling to keep organism viable
3. SurvivalGuardian - Invisible density injection on near-death (no visible restart)

All smoothing is frame-rate independent via delta-time integration.
"""

import math
import numpy as np


class SmoothedParameter:
    """EMA wrapper for a single numeric parameter.

    Provides frame-rate-independent exponential moving average smoothing
    with configurable time constant. Slider changes drift organically
    over 2-3 seconds instead of snapping instantly.

    Time constant controls the "feel":
    - tau=2.0s: dreamy drift
    - tau=0.5s: responsive but smooth
    - tau=5.0s: very slow drift
    """

    def __init__(self, initial_value, time_constant=2.0):
        """Initialize smoothed parameter.

        Args:
            initial_value: Starting value (both current and target)
            time_constant: Time in seconds to reach ~63% of target (tau)
        """
        self.target = initial_value
        self.current = initial_value
        self.tau = time_constant

    def set_target(self, new_target):
        """Set new target value (called by slider callbacks).

        Args:
            new_target: New target value to drift toward
        """
        self.target = new_target

    def update(self, dt):
        """Advance EMA by delta-time (called each frame).

        Uses frame-rate-independent exponential smoothing:
        alpha = 1 - exp(-dt / tau)
        current += alpha * (target - current)

        Args:
            dt: Time elapsed in seconds since last update
        """
        if dt <= 0:
            return
        alpha = 1.0 - math.exp(-dt / self.tau)
        self.current += alpha * (self.target - self.current)

    def get_value(self):
        """Get current smoothed value.

        Returns:
            Current value (after EMA smoothing)
        """
        return self.current

    def snap(self, value):
        """Immediately set both target and current (for reset).

        Args:
            value: Value to snap to (no smoothing)
        """
        self.target = value
        self.current = value


class LeniaParameterCoupler:
    """Ratio-based mu/sigma coupling for Lenia organism viability.

    Keeps mu and sigma in a viable relationship by blending their ratio
    toward a baseline. At coupling_strength=0.5, user retains control
    but the organism stays alive during aggressive slider dragging.

    Safe bounds prevent user from putting parameters outside viable ranges:
    - mu: [0.05, 0.35]
    - sigma: [0.005, 0.06]
    """

    def __init__(self, preset):
        """Initialize coupler from preset baseline.

        Args:
            preset: Preset dict with 'mu' and 'sigma' baseline values
        """
        self.baseline_ratio = preset["mu"] / preset["sigma"]
        self.coupling_strength = 0.5  # 50% guidance, 50% user control

        # Safe bounds (organism remains viable within these ranges)
        self.mu_min = 0.05
        self.mu_max = 0.35
        self.sigma_min = 0.005
        self.sigma_max = 0.06

    def couple(self, mu_target, sigma_target):
        """Apply coupling to mu/sigma targets.

        Blends current ratio toward baseline ratio using coupling_strength,
        then clamps to safe bounds.

        Args:
            mu_target: User's requested mu value
            sigma_target: User's requested sigma value

        Returns:
            Tuple of (adjusted_mu, adjusted_sigma)
        """
        # Calculate current ratio
        if sigma_target <= 0:
            sigma_target = self.sigma_min
        current_ratio = mu_target / sigma_target

        # Blend toward baseline ratio
        blended_ratio = (
            current_ratio * (1.0 - self.coupling_strength) +
            self.baseline_ratio * self.coupling_strength
        )

        # Apply blended ratio while preserving user's mu (user moves mu, sigma follows)
        adjusted_mu = mu_target
        adjusted_sigma = adjusted_mu / blended_ratio

        # Clamp to safe bounds
        adjusted_mu = max(self.mu_min, min(self.mu_max, adjusted_mu))
        adjusted_sigma = max(self.sigma_min, min(self.sigma_max, adjusted_sigma))

        # Re-check ratio after clamping (ensure we didn't violate bounds)
        if adjusted_sigma > 0:
            final_ratio = adjusted_mu / adjusted_sigma
            if abs(final_ratio - self.baseline_ratio) > abs(current_ratio - self.baseline_ratio):
                # Clamping made it worse - use original values with gentle clamping
                adjusted_mu = max(self.mu_min, min(self.mu_max, mu_target))
                adjusted_sigma = max(self.sigma_min, min(self.sigma_max, sigma_target))

        return (adjusted_mu, adjusted_sigma)

    def update_baseline(self, preset):
        """Update baseline ratio from new preset (for preset morphing).

        Args:
            preset: New preset dict with 'mu' and 'sigma' values
        """
        self.baseline_ratio = preset["mu"] / preset["sigma"]


class SurvivalGuardian:
    """Invisible density injection on near-death.

    Monitors organism mass and injects a gentle Gaussian blob when
    mass drops critically low. This prevents "death" without visible
    restart (no seed(), no iridescent reset, no LFO reset).

    The injection is subtle enough that the simulation can naturally
    grow from it, creating organic recovery instead of jarring restart.
    """

    def __init__(self, engine, critical_mass=0.002, rescue_cooldown=5.0):
        """Initialize survival guardian.

        Args:
            engine: CAEngine instance to monitor and rescue
            critical_mass: Mass threshold below which to inject density
            rescue_cooldown: Seconds to wait between rescue attempts
        """
        self.engine = engine
        self.critical_mass = critical_mass
        self.rescue_cooldown = rescue_cooldown
        self.cooldown_timer = 0.0

    def check_and_rescue(self, dt):
        """Check mass and inject density if needed (called each frame).

        Decrements cooldown timer by dt. If mass is below critical threshold
        AND cooldown has expired, injects a gentle Gaussian blob at center
        and resets cooldown.

        Args:
            dt: Time elapsed in seconds since last check

        Returns:
            True if rescue was performed, False otherwise
        """
        # Decrement cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= dt

        # Check if rescue is needed
        mass = self.engine.get_mass()
        if mass < self.critical_mass and self.cooldown_timer <= 0:
            # Inject gentle Gaussian blob at center
            size = self.engine.size
            cx, cy = size // 2, size // 2
            radius = size // 8
            amplitude = 0.05  # Subtle injection

            # Create Gaussian blob using ogrid (matches engine_base.py pattern)
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            influence = np.exp(-(dist ** 2) / (2 * (radius / 2.5) ** 2)) * amplitude

            # Add to world (clip to valid range)
            self.engine.world = np.clip(self.engine.world + influence, 0.0, 1.0)

            # Reset cooldown
            self.cooldown_timer = self.rescue_cooldown

            return True

        return False
