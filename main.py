'''
Created on 13 sep. 2015

@author: danhe
'''

import PyOCLMD as MD

if __name__ == '__main__':
    
    max_iter = 1e4
    eps = 0.0104
    sig = 3.40
    mass = 39.948
    cutoff_radius = 12
    coord, box = MD.create_simple_cubic(1e4, 3.0)
    
    sim = MD.Simulation(coord, mass, box, cutoff_radius, eps, sig, 0.25)
    sim.create_worksizes(512)
    sim.ion_type = ['Ar'] * sim.ions
    
    sim.write_xyz('test.xyz')
    for i in range(int(max_iter)):
        sim.run_nve()
        sim.write_xyz('test.xyz', 10, 'a')
        sim.print_status(10)
    
    pass
