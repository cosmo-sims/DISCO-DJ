from .dkd_leapfrog import DriftKickDrift

__all__ = ["DKDPiIntegrator"]


class DKDPiIntegrator(DriftKickDrift):
    """Drift-kick-drift Pi-integrator (D-time integrator), see https://arxiv.org/abs/2301.09655.
    """
    integrator_name: str = "bullfrog"

    def get_integrator_args(self) -> dict:
        integrator_dict = {"bullfrog": self.bullfrog,
                           "fastpm": self.fastpm}
        return integrator_dict[self.integrator_name]()

    # BullFrog integrator, see http://arxiv.org/abs/2409.19049
    def bullfrog(self) -> dict:
        # Unnormalized growth functions
        D1 = self.cosmo._timetables["Dplus_unnormed_at_1"]
        Dplus_u = lambda a_: self.cosmo.Dplus(a_) * D1
        Dplusda_u = lambda a_: self.cosmo.Dplusda(a_) * D1
        D2plus_u = lambda a_: self.cosmo.get_interpolated_property(a_, key_from="a", key_to="D2plus") * D1 ** 2
        D2plusda_u = lambda a_: self.cosmo.get_interpolated_property(a_, key_from="a", key_to="D2plusda") * D1 ** 2

        # Get the time steps in terms of internal time and scale factor a
        all_a, all_internal = self.all_a_and_internal
        internal_mid = (all_internal[:-1] + all_internal[1:]) / 2.0  # midpoint w.r.t. internal time is used
        a_mid = self.internal_to_a(internal_mid)

        # Compute growth functions and their derivatives
        all_D = Dplus_u(all_a)
        all_Dda = Dplusda_u(all_a)
        all_D2 = D2plus_u(all_a)
        all_D2da = D2plusda_u(all_a)

        # Setup the variables for the kick coefficients alpha and beta
        D_begin = all_D[:-1]
        D_end = all_D[1:]
        dD = D_end - D_begin
        xi = D_begin / dD

        Dda_begin = all_Dda[:-1]
        Dda_end = all_Dda[1:]
        D2_begin = all_D2[:-1]
        D2da_begin = all_D2da[:-1]
        D2da_end = all_D2da[1:]

        # Compute coefficient alpha
        bracket = (D2_begin + D2da_begin / Dda_begin * dD / 2.0) / ((xi + 0.5) * dD) - (xi + 0.5) * dD
        alphas = (D2da_end / Dda_end - bracket) / (D2da_begin / Dda_begin - bracket)

        # Compute coefficient beta from Zel'dovich consistency condition
        D_mid_norm = self.cosmo.Dplus(a_mid)  # NOTE: this is NORMALIZED!
        betas = (1.0 - alphas) / D_mid_norm

        # Drift length: in terms of (normalized!) D-time
        D_begin_norm = D_begin / D1
        D_end_norm = D_end / D1
        dD1_norm = D_mid_norm - D_begin_norm
        dD2_norm = D_end_norm - D_mid_norm

        return {"alpha": alphas, "beta": betas, "ddrift1": dD1_norm, "ddrift2": dD2_norm}

    # FastPM integrator, see http://arxiv.org/abs/1603.00476
    def fastpm(self):
        # Get the time steps in terms of internal time and scale factor a
        all_a, all_internal = self.all_a_and_internal
        internal_mid = (all_internal[:-1] + all_internal[1:]) / 2.0  # midpoint w.r.t. internal time is used
        a_mid = self.internal_to_a(internal_mid)

        # Compute the kick coefficients alpha and beta
        a_begin = all_a[:-1]
        a_end = all_a[1:]
        alphas = self.cosmo.Fplus(a_begin) / self.cosmo.Fplus(a_end)
        betas = (1.0 - alphas) / self.cosmo.Dplus(a_mid)  # Zel'dovich consistency

        # Drift length: in terms of (normalized!) D-time
        all_D = self.cosmo.Dplus(all_a)
        D_begin = all_D[:-1]
        D_end = all_D[1:]
        D_mid = self.cosmo.Dplus(a_mid)
        dD1_norm = D_mid - D_begin
        dD2_norm = D_end - D_mid

        return {"alpha": alphas, "beta": betas, "ddrift1": dD1_norm, "ddrift2": dD2_norm}

    def Pi_to_velocity(self, pi, a):
        return pi

    def velocity_to_Pi(self, vel, a):
        return vel
