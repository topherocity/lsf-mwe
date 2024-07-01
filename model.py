import numpy as np
import meep as mp

# define the speed of light
c0 = 299792458 # m/s

class MeepLSFModel:
    ''' defines a simple LSF model using Meep
    
    Parameters
    ----------
    lsm : lsf.LevelSetMethod
        a class defining the basis functions for
        the level set method
    coefficients: list-like
        a list of coefficients which map to level set basis
        functions
    period: float or list-like
        dimension of the unit cell in arbitrary units, in the 
        form of a single float for a square unit cell, or 
        a tuple of two floats (Ny, Nz) in the case of a 
        rectangular unit cell.
    dx: float or list-like
        Yee cell size in arbitrary units.
    substrate_medium : meep.geom.Medium
        a Meep medium describing the substrate material
    substrate_thickness : float
        the thickness of the substrate
    fmax : float, optional
        largest frequency desired from the simulation. The
        default is 40 GHz.
    units_to_meters: float, optional
        conversion of the arbitrary units to meters. The
        default is millimeters (1e-3).
    num_freqs: int
        Number of output frequencies. The default is 100.
    '''
    def __init__(self,
                 lsm,
                 coefficients,
                 period,
                 dx,
                 substrate_medium,
                 substrate_thickness,
                 fmax=40e9,
                 units_to_meters=1e-3,
                 num_freqs=100):
        self.lsm = lsm
        self.coefficients = coefficients
        try:
            self.period = list(period)
        except TypeError:
            self.period = [period, period]

        self.dx = dx
        self.substrate_medium = substrate_medium
        self.substrate_thickness = substrate_thickness
        
        self.fmax = fmax
        self.units_to_meters = units_to_meters
        self.num_freqs = num_freqs
        
    @property
    def sim(self):
        ''' Meep simulation object '''
        try:
            return self._sim
        except AttributeError:
            self.create_sim()
            return self._sim

    @property
    def mask(self):
        ''' mask for the LSF metal pattern '''
        Nx, Ny = np.array(self.period) // self.dx
    
        mask = self.lsm.create_template(self.coefficients,
                                        shape=(Nx, Ny))
        return mask

    def create_sim(self):
        ''' creates the simulation object '''
        # define the dimensions of the geometry
        dx = self.dx
        Sy, Sz = self.period

        # define the PML to be 40 cells thick
        dpml = 40*dx
        pml_layers = [mp.PML(thickness=dpml,
                             direction=mp.X)]
        
        # put an air box 60 cells thick
        dair = 60*dx
        
        # define the total grid thickness
        dsub = self.substrate_thickness

        Sx = 2*(dpml + dair) + 2*dsub
        
        # define the center frequency and bandwidth 
        # in arbitrary units
        fmax = 1.15 * self.fmax / c0 * self.units_to_meters
        fmin = fmax / 2.7**2
        f0 = (fmin + fmax) / 2
        bw = fmax - fmin
        
        # define a z-oriented Gaussian source, placed at 
        # the edge of the PML
        sources = [
            mp.Source(
                mp.GaussianSource(f0, 
                                  fwidth=bw, 
                                  is_integrated=True),
                center=mp.Vector3(-0.5*Sx+dpml+dx, 0, 0),
                size=mp.Vector3(0, Sy, Sz),
                component=mp.Ez)]
        
        # now we add the LSF to the simulation
        grid = mp.MaterialGrid(
            grid_size=mp.Vector3(0, *self.mask.shape),
            medium1=mp.air,
            medium2=mp.metal,
            weights=self.mask.astype('bool'),
            do_averaging=False
        )
        
        # define the geometry
        geometry = [
            # substrate
            mp.Block(
                center=mp.Vector3(dsub/2, 0, 0),
                size=mp.Vector3(dsub, Sy, Sz),
                material=self.substrate_medium),
            # LSF metal pattern
            mp.Block(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(0, Sy, Sz),
                material=grid),   
        ]
        
        # set up the simulation
        self._sim = mp.Simulation(
            resolution=1/dx,
            cell_size=mp.Vector3(Sx, Sy, Sz),
            boundary_layers=pml_layers,
            sources=sources,
            geometry=geometry,
            eps_averaging=False,
            k_point=mp.Vector3(),
            Courant=0.48
        )
        
        # add mode monitors for capturing fields

        nfrq = self.num_freqs
        self.box_x1 = self._sim.add_mode_monitor(
            f0, bw, nfrq,
            mp.ModeRegion(center=mp.Vector3(x=-Sx/2+dpml+2*dx),
                          size=mp.Vector3(0, Sy, Sz)))
        
        self.box_x2 = self._sim.add_mode_monitor(
            f0, bw, nfrq,
            mp.ModeRegion(center=mp.Vector3(x=Sx/2-dpml-2*dx),
                          size=mp.Vector3(0, Sy, Sz)))
            
        # define where to sample the fields for simulation
        # completion
        self.sample_fields = mp.Vector3(0.5*Sx-dpml-dair/2, 0, 0)
        
    def run_simulation(self,
                       sim_verbosity=1,
                       mode_verbosity=0):
        mp.verbosity(sim_verbosity)
        self.sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                500, mp.Ez, self.sample_fields, 1e-4
            )
        )

        mp.verbosity(mode_verbosity)
        p1_coeff = self.sim.get_eigenmode_coefficients(
            self.box_x1, [1],
            eig_parity=mp.NO_PARITY
        )
        
        p2_coeff = self.sim.get_eigenmode_coefficients(
            self.box_x2, [1],
            eig_parity=mp.NO_PARITY
        )
        
        mp.verbosity(1)

        a1, b1 = p1_coeff.alpha.squeeze().T
        a2, b2 = p2_coeff.alpha.squeeze().T
        
        self.f = np.array(
            mp.get_flux_freqs(self.box_x1)
        )*c0/self.units_to_meters

        self.s11 = b1/a1
        self.s21 = a2/a1

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('./mwe.mplstyle')

    coefficients = np.loadtxt('coefficients.txt')

    lsm = LevelSetMethod.uniform_grid(
        (32, 32),
        wrap=True,
        spacing_factor=1)

    model = MeepLSFModel(
        lsm=lsm,
        coefficients=coefficients,
        period=240,
        dx=2,
        substrate_medium=mp.Medium(epsilon=3.0,
                                   D_conductivity=0.001),
        substrate_thickness=6,
        units_to_meters=0.0254/1000,
        fmax=40e9,
    )
    
    model.run_simulation()
    
    def db(x):
        return 20*np.log10(abs(x))
    
    plt.plot(model.f/1e9, db(model.s11),
             'o', markersize=2, color=ln.get_color(),
             label='$S_{11}$, MEEP')
    plt.plot(model.f/1e9, db(model.s21),
        'o', markersize=2, color=ln.get_color(),
        label='$S_{21}$, MEEP')
    
    plt.ylim(-40, 1)
    plt.legend()
    plt.grid()

    plt.xlim(5, 40)
    plt.tight_layout()
    plt.savefig('mwe.png')
