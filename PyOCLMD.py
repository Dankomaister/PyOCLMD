'''
Created on 13 sep. 2015

@author: danhe
'''

import pyopencl as cl
import numpy as np
import time

def create_simple_cubic(ions, sep):
    
    ions = round(ions ** (1 / 3)) ** 3
    d = np.arange(0, ions ** (1 / 3)) * sep
    
    y, z, x = np.meshgrid(d, d, d)
    coordinates = np.zeros((ions, 3))
    coordinates[:, 0] = x.reshape(ions)
    coordinates[:, 1] = y.reshape(ions)
    coordinates[:, 2] = z.reshape(ions)
    
    box_size = np.max(coordinates, axis=0)
    
    return coordinates, box_size

class Simulation():
    
    def __init__(self, coordinates, mass, box_size, cutoff_radius, eps, sig, dt=1, platform=0, device=0):
        
        # Scalars
        self.dfloat = 'float32'
        self.dint = 'int32'
        self.iter = np.int32(0)
        self.kb = np.float32(8.6173324e-5)
        
        self.ions = np.int32(coordinates.shape[0])
        self.dt = np.float32(dt)
        self.eps = np.float32(eps)
        self.sig = np.float32(sig)
        
        self.bin_ratio = np.float32(1)
        self.bin_size = np.int32(self.ions)
        self.cutoff_radius = np.float32(cutoff_radius)
        
        # Timers
        self.timers = {'reset_uint' : 0,
                       'fill_bins' : 0,
                       'prefix_sum' : 0,
                       'block_sum' : 0,
                       'counting_sort' : 0,
                       'reindex' : 0,
                       'get_force' : 0,
                       'verlet_part1' : 0,
                       'verlet_part2' : 0
                       }
        
        # 1D arrays
#         self.mass = mass.astype(self.dfloat)
        self.mass = np.ones(self.ions, dtype=self.dfloat) * mass
        self.mass_sorted = np.zeros(self.ions, dtype=self.dfloat)
        self.potential = np.zeros(self.ions, dtype=self.dfloat)
        self.kinetic = np.zeros(self.ions, dtype=self.dfloat)
        self.temperature = np.zeros(self.ions, dtype=self.dfloat)
        self.bin_index = np.zeros(self.ions, dtype=self.dint)
        self.s_potential = np.array([])
        self.s_kinetic = np.array([])
        self.s_temperature = np.array([])
        
        # 4D arrays
        self.coordinates = np.hstack((coordinates, np.zeros((self.ions, 1)))).astype(self.dfloat)
        self.coordinates_sorted = np.zeros(self.coordinates.shape, dtype=self.dfloat)
        self.velocity = np.zeros(self.coordinates.shape, dtype=self.dfloat)
        self.velocity_sorted = np.zeros(self.coordinates.shape, dtype=self.dfloat)
        self.force = np.zeros(self.coordinates.shape, dtype=self.dfloat)
        self.box_size = np.hstack((box_size, [1])).astype(self.dfloat)
        
        # Setup opencl
        self.cl_platform = cl.get_platforms()[platform]
        self.cl_device = self.cl_platform.get_devices()[device]
        self.cl_max_lws = self.cl_device.max_work_group_size
        
        self.cl_mf = cl.mem_flags
        self.cl_ctx = cl.create_some_context()
        self.cl_queue = cl.CommandQueue(self.cl_ctx)
        self.read_kernel('MD_kernel.cl')
        
        # Create buffers
        #  1D arrays
        self.cl_mass = self.create_buffer(self.mass)
        self.cl_mass_sorted = self.create_buffer(self.mass_sorted)
        self.cl_potential = self.create_buffer(self.potential)
        self.cl_kinetic = self.create_buffer(self.kinetic)
        self.cl_temperature = self.create_buffer(self.temperature)
        self.cl_bin_index = self.create_buffer(self.bin_index)
        
        #  4D arrays
        self.cl_coordinates = self.create_buffer(self.coordinates)
        self.cl_coordinates_sorted = self.create_buffer(self.coordinates_sorted)
        self.cl_velocity = self.create_buffer(self.velocity)
        self.cl_velocity_sorted = self.create_buffer(self.velocity_sorted)
        self.cl_force = self.create_buffer(self.force)
        
        self.create_worksizes(self.cl_max_lws)
        self.create_bins()
        
        return
    
    def read_kernel(self, file_name):
        
        file = open(file_name, 'r')
        kernel = ''.join(file.readlines())
        self.cl_kernel = cl.Program(self.cl_ctx, kernel).build()
        
        return
    
    def create_buffer(self, data):
        
        return cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE | self.cl_mf.COPY_HOST_PTR, hostbuf=data)
    
    def read_buffer(self, read, write):
        
        cl.enqueue_read_buffer(self.cl_queue, read, write).wait()
        
        return
    
    def write_buffer(self, read, write):
        
        cl.enqueue_write_buffer(self.cl_queue, read, write).wait()
        
        return
    
    def create_worksizes(self, lws=1):
        
        self.cl_lws = lws
        self.cl_gws = int(np.ceil(self.ions / self.cl_lws).item() * self.cl_lws)
        self.cl_blockws = np.int32(2 * self.cl_lws)
        
        self.cl_block = cl.LocalMemory(self.cl_blockws.item() * np.dtype(self.dint).itemsize)
        
        return
    
    def create_bins(self):
        
        self.bin_length = np.float32(self.bin_ratio * self.cutoff_radius)
        self.bin_N = np.int32(np.ceil(np.max(self.box_size[0:3]) / self.bin_length))
        self.bins = np.int32(self.bin_N ** 3)
        
        self.ion_count = np.zeros(self.bins, dtype=self.dint)
        self.prefix_sum = np.zeros(self.bins, dtype=self.dint)
        self.block_sum = np.zeros(np.ceil(self.bins / self.cl_blockws), dtype=self.dint)
        
        self.cl_ion_count = self.create_buffer(self.ion_count)
        self.cl_prefix_sum = self.create_buffer(self.prefix_sum)
        self.cl_block_sum = self.create_buffer(self.block_sum)
        
        return
    
    def write_xyz(self, file_name, write_delay=1, io='w'):
        
        if self.iter % write_delay == 0:
            
            self.read_buffer(self.cl_coordinates, self.coordinates)
            
            file = open(file_name, io)
            
            file.write('%i\n' % self.ions)
            file.write('Current iteration %i, simulation time %f fs.\n' % (self.iter, self.iter * self.dt))
            
            for i in zip(self.ion_type, self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]):
                file.write('%3s %8.3f %8.3f %8.3f\n' % i)
            
            file.close()
            
            return
        else:
            return
    
    def get_force(self):
        
        self.cl_bws = int(np.ceil(self.bins / self.cl_lws).item() * self.cl_lws)
        self.cl_bws_half = int(np.ceil(0.5 * self.bins / self.cl_lws).item() * self.cl_lws)
        
        # reset_uint
        timer = time.time()
        reset_uint = self.cl_kernel.reset_uint(self.cl_queue, (self.cl_bws, 1), (self.cl_lws, 1),
                                               self.cl_ion_count, self.bins)
        reset_uint.wait()
        self.timers['reset_uint'] += time.time() - timer
        
        # fill_bins
        timer = time.time()
        fill_bins = self.cl_kernel.fill_bins(self.cl_queue, (self.cl_gws, 1), (self.cl_lws, 1),
                                             self.cl_coordinates, self.cl_bin_index, self.cl_ion_count,
                                             self.bin_length, self.bin_N, self.ions)
        fill_bins.wait()
        self.timers['fill_bins'] += time.time() - timer
        
        # prefix_sum
        timer = time.time()
        prefix_sum = self.cl_kernel.prefix_sum(self.cl_queue, (self.cl_bws_half, 1), (self.cl_lws, 1),
                                               self.cl_ion_count, self.cl_prefix_sum, self.cl_block_sum,
                                               self.cl_block, self.cl_blockws, self.bins)
        prefix_sum.wait()
        self.timers['prefix_sum'] += time.time() - timer
        
        # block_sum
        timer = time.time()
        block_sum = self.cl_kernel.block_sum(self.cl_queue, (self.cl_bws_half, 1), (self.cl_lws, 1),
                                             self.cl_prefix_sum, self.cl_block_sum, self.bins)
        block_sum.wait()
        self.timers['block_sum'] += time.time() - timer
        
        # counting_sort
        timer = time.time()
        counting_sort = self.cl_kernel.counting_sort(self.cl_queue, (self.cl_gws, 1), (self.cl_lws, 1),
                                                     self.cl_bin_index, self.cl_prefix_sum,
                                                     self.cl_coordinates, self.cl_velocity, self.cl_mass,
                                                     self.cl_coordinates_sorted, self.cl_velocity_sorted, self.cl_mass_sorted,
                                                     self.ions)
        counting_sort.wait()
        self.timers['counting_sort'] += time.time() - timer
        
        # reindex
        timer = time.time()
        reindex = self.cl_kernel.reindex(self.cl_queue, (self.cl_gws, 1), (self.cl_lws, 1),
                                         self.cl_coordinates, self.cl_velocity, self.cl_mass,
                                         self.cl_coordinates_sorted, self.cl_velocity_sorted, self.cl_mass_sorted,
                                         self.ions)
        reindex.wait()
        self.timers['reindex'] += time.time() - timer
        
        # get_force
        timer = time.time()
        get_force = self.cl_kernel.get_force(self.cl_queue, (self.cl_gws, 1), (self.cl_lws, 1),
                                             self.cl_coordinates, self.cl_force, self.cl_potential, self.cl_prefix_sum, self.cl_ion_count,
                                             self.bin_length, self.bin_N, self.cutoff_radius, self.eps, self.sig, self.ions)
        get_force.wait()
        self.timers['get_force'] += time.time() - timer
        
        return
    
    def verlet_part1(self):
        
        timer = time.time()
        verlet_part1 = self.cl_kernel.verlet_part1(self.cl_queue, (self.cl_gws, 1), (self.cl_lws, 1),
                                                   self.cl_coordinates, self.cl_velocity, self.cl_force,
                                                   self.cl_mass, self.box_size, self.dt, self.ions)
        verlet_part1.wait()
        self.timers['verlet_part1'] += time.time() - timer
        
        return
    
    def verlet_part2(self):
        
        timer = time.time()
        verlet_part2 = self.cl_kernel.verlet_part2(self.cl_queue, (self.cl_gws, 1), (self.cl_lws, 1),
                                                   self.cl_velocity, self.cl_force,
                                                   self.cl_mass, self.cl_kinetic, self.cl_temperature, self.dt, self.ions)
        verlet_part2.wait()
        self.timers['verlet_part2'] += time.time() - timer
        
        return
    
    def run_nve(self):
        
        self.verlet_part1()
        self.get_force()
        self.verlet_part2()
        
        self.iter += 1
        
        return
    
    def get_system_info(self):
        
        self.read_buffer(self.cl_potential, self.potential)
        self.read_buffer(self.cl_kinetic, self.kinetic)
        self.read_buffer(self.cl_temperature, self.temperature)
        
        self.s_potential = np.append(self.s_potential, np.sum(self.potential))
        self.s_kinetic = np.append(self.s_kinetic, np.sum(self.kinetic))
        self.s_temperature = np.append(self.s_temperature, np.sum(self.temperature))
        
        return
    
    def print_status(self, write_delay=1):
        
        if self.iter < 2:
            print('%9s %13s %13s %13s %13s' % ('Iter', 'Kin. E. (eV)', 'Pot. E. (eV)', 'Tot. E. (eV)', 'Perf. (Mp/s)'))
        
        elif self.iter % write_delay == 0:
            
            self.get_system_info()
            
            print('%9i %13.5f %13.5f %13.5f %13.5f' % (self.iter, self.s_kinetic[-1], self.s_potential[-1],
                                                       self.s_potential[-1] + self.s_kinetic[-1],
                                                       1e-6 * self.iter * self.ions / self.timers['get_force']))
            
            return
        else:
            return
    
    
