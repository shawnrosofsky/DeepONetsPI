from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import uniform_filter
import warnings
# import pandas as pd
import sys
import os
import scipy as sci
import glob
import math
from IPython.display import display
import h5py
import h5py as h5
import multiprocessing as mp
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap, random
import jax.scipy as jsci

# Define file names
def get_samrai_dirs(data_dir):
    samrai_dirs = []
    dumps_file = os.path.join(data_dir, 'dumps.visit')
    with open(dumps_file, "r") as f:
        for line in f.readlines():
            sub_dir = line.split(sep='/')[0]
            samrai_dir = os.path.join(data_dir,sub_dir)
            samrai_dirs.append(samrai_dir)
    return samrai_dirs

def get_output_files(samrai_dirs,output_dir,basename='output-'):
    output_files = []
    for samrai_dir in samrai_dirs:
        it = samrai_dir.split(sep='.')[-1]
        filename = basename + it + '.h5'
        output_file = os.path.join(output_dir,filename)
        output_files.append(output_file)
    return output_files


class SamraiReader:
    def __init__(self, data_dir, vars=None):
        samrai_dirs = self.get_samrai_dirs(data_dir)
        Nt = len(samrai_dirs)
        samrai_dir0 = samrai_dirs[0]
        summary_file0 = os.path.join(samrai_dir0, 'summary.samrai')
        with h5py.File(summary_file0, 'r') as f:
            BASIC_INFO = f['BASIC_INFO']
            # t = BASIC_INFO['time'][()]
            # iteration = BASIC_INFO['time_step_number'][()]
            dxyz = BASIC_INFO['dx'][()].flatten()
            dim = BASIC_INFO['number_dimensions_of_problem'][()]
            dx, dy, dz = dxyz
            var_names = BASIC_INFO['var_names'][()]
            if vars is None:
                vars = [var.decode('utf-8') for var in var_names]
            n_patch_files = BASIC_INFO['number_file_clusters'][()]
            n_global_patches = BASIC_INFO['number_global_patches'][()]
            n_levels = BASIC_INFO['number_levels'][()]
            n_local_patches = BASIC_INFO['number_patches_at_level'][()]
            n_procs = BASIC_INFO['number_processors'][()]
            n_vars_visit = BASIC_INFO['number_visit_variables'][()]
            extents = f['extents']
            patch_map = extents['patch_map']
            patchmap = patch_map[()]
            # display(patch_map.items())
            patches = patch_map['patch_number'][()]
            processors = patch_map['processor_number'][()]
            levels = patch_map['level_number'][()]
            file_clusters = patch_map['file_cluster_number'][()]
            patch_extents = extents['patch_extents']
            ind_lower = patch_extents['lower'][()]
            ind_upper = patch_extents['upper'][()]
            xlo = patch_extents['xlo'][()]
            xup = patch_extents['xup'][()]
            N = np.amax(ind_upper, axis=0) + 1
            xyz_max = np.amax(xup, axis=0)
            xyz_min = np.amin(xlo, axis=0)
            Nx, Ny, Nz = N
            # dshape = tuple(N)
            dshape = (Nx, Ny)
            Ntot = Nx * Ny
            x_max, y_max, z_max = xyz_max
            x_min, y_min, z_min = xyz_min
            x = np.arange(x_min, x_max, dx)
            y = np.arange(y_min, y_max, dy)
            # z = np.arange(z_min, z_max, dz)
            X, Y = np.meshgrid(x, y, indexing='ij')
            chunks = tuple(ind_upper[0] - ind_lower[0] + 1)
            chunks = (chunks[0], chunks[1])
            chunks_full = (Nx, Ny, 1)
        
        self.data_dir = data_dir
        self.vars = vars
        self.samrai_dirs = samrai_dirs
        self.samrai_dir0 = samrai_dir0
        self.simulation_name = os.path.split(self.data_dir)
        
        self.X = X
        self.Y = Y
        self.x, self.y = x, y
        self.dxyz = dxyz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max
        self.N = N
        self.Nt = Nt
        self.Ntot = Ntot
        self.dshape = dshape
        self.dshape_full = (Nx, Ny, Nt)
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dim = dim
        
        self.n_patch_files = n_patch_files
        self.n_global_patches = n_global_patches
        self.n_levels = n_levels
        self.n_local_patches = n_local_patches
        self.n_procs = n_procs
        self.n_vars_visit = n_vars_visit
        
        self.extents = extents
        # self.patch_map = patch_map
        self.patchmap = patchmap
        self.patches = patches
        self.processors = processors
        self.levels = levels
        self.file_clusters = file_clusters
        self.patch_extents = patch_extents
        self.ind_lower = ind_lower
        self.ind_upper = ind_upper
        self.xlo = xlo
        self.xup = xup
        self.chunks = chunks
        self.chunks_full = chunks_full
        self.times, self.iterations = self.get_times()
        self.vars0 = self.get_data(samrai_dir0)
        self.it_max = self.iterations[-1]
        self.padding = int(math.log10(self.it_max)) + 1

        
    def write_params(self, output_file):
        # Fill out common parmeters
        with h5py.File(output_file, 'a') as f:
            params = f.create_group('params')
            # params.create_dataset('X', data=self.X.reshape(-1, 1))
            # params.create_dataset('Y', data=self.Y.reshape(-1, 1))
            
            params.create_dataset('X', data=self.X)
            params.create_dataset('Y', data=self.Y)
            params.create_dataset('x', data=self.x, shape=(self.Nx, 1), dtype=np.float32)
            params.create_dataset('y', data=self.y, shape=(self.Ny, 1), dtype=np.float32)
            params.attrs['dxyz'] = self.dxyz
            params.attrs['xyz_min'] = self.xyz_min
            params.attrs['xyz_max'] = self.xyz_max
            params.attrs['simulation_name'] = self.simulation_name
            # times, iterations = get_times(samrai_dirs, self.Nt)
            params.create_dataset('times', data=self.times, shape=(self.Nt, 1), dtype=np.float32)
            params.create_dataset('iterations', data=self.iterations, shape=(self.Nt, 1), dtype=np.int32)
            for var in self.vars:
                # params.create_dataset(f'{var}0', data=self.vars0[var].reshape(-1, 1), dtype=np.float32)
                params.create_dataset(f'{var}0', data=self.vars0[var], shape=(self.Ntot, 1), dtype=np.float32)


    
    def get_data(self, samrai_dir):
        # patches = range(len(self.patchmap))
        summary_file = os.path.join(samrai_dir, 'summary.samrai')
        # display(self.patchmap)
        data = {var: np.zeros(self.dshape, dtype=np.float32) for var in self.vars}
        for i, p_map in enumerate(self.patchmap):
            # p_map = patchmap[i]
            ind_min = self.ind_lower[i]
            ind_max = self.ind_upper[i]
            file_num = p_map['file_cluster_number']
            proc_num = p_map['processor_number']
            level_num = p_map['level_number']
            patch_num = p_map['patch_number']
            filename = 'processor_cluster.{:05d}.samrai'.format(file_num)
            file = os.path.join(samrai_dir, filename)
            proc = 'processor.{:05d}'.format(proc_num)
            level = 'level.{:05d}'.format(level_num)
            patch = 'patch.{:05d}'.format(patch_num)
            n = ind_max - ind_min + 1
            nx, ny, nz = n
            # display(n)
            patch_shape = tuple(n + 1) # original patch (1 extra grid point)
            patch_shape = (patch_shape[0], patch_shape[1])
            # display(patch_shape)
            ind_x = slice(ind_min[0], ind_max[0]+1)
            ind_y = slice(ind_min[1], ind_max[1]+1)
            # ind_z = slice(ind_min[2], ind_max[2]+1)
            
            with h5.File(file, 'r') as f:
                for var in vars:
                    dat = f[proc][level][patch][var][:].reshape(patch_shape, order='F')
                    # data[var][ind_x, ind_y, ind_z] = dat[:nx, :ny, :nz]
                    data[var][ind_x, ind_y] = dat[:nx, :ny]
        return data
    
    def write_data(self, samrai_dir, output_file, index, print_dir=True):
        if print_dir:
            display(samrai_dir)
        t, it = self.get_time(samrai_dir)
        data = self.get_data(samrai_dir)
        with h5py.File(output_file, 'a') as f:
            for var in self.vars:
                dset = f.require_dataset(var, shape=self.dshape_full, dtype=np.float32, chunks=self.chunks_full)
                dset[:, :, index] = data[var]
                # group = f.require_group(var)
                # dat = data[var].reshape((-1, 1))
                # dset = group.create_dataset(f'{var} it={it[0]:0{self.padding}d}', data=dat, dtype=np.float32)
                # dset = group.create_dataset(f'{var} it={it[0]}', data=dat, dtype=np.float32)
                # dset.attrs['time'] = np.float32(t)
                # dset.attrs['iteration'] = it
    
                
    def generate_output(self, output_file):
        # overwrite current output file
        f = h5py.File(output_file, 'w')
        f.close()
        # write the parameters
        self.write_params(output_file)
        # write the data
        for i, samrai_dir in enumerate(self.samrai_dirs):
            self.write_data(samrai_dir, output_file, i)
    
    def generate_output_parallel(self, output_file, num_procs=None):
        # make output files a list for starmap
        output_files = [output_file] * len(self.samrai_dirs)
        
        # overwrite current output file
        f = h5py.File(output_file, 'w')
        f.close()
        # write the parameters
        self.write_params(output_file)
        # write the data using pmap
        with mp.Pool(num_procs) as p:
            args = zip(samrai_dirs, output_files)
            p.starmap(self.write_data, args)
            
    def get_time(self, samrai_dir):
        summary_file = os.path.join(samrai_dir, 'summary.samrai')
        with h5py.File(summary_file, 'r') as f:
            BASIC_INFO = f['BASIC_INFO']
            t = BASIC_INFO['time'][:]
            it = BASIC_INFO['time_step_number'][:]
        return t, it
        
    def get_times(self):
        times = np.zeros((self.Nt, 1), dtype=np.float32)
        iterations = np.zeros((self.Nt, 1), dtype=np.int32)
        for i, samrai_dir in enumerate(self.samrai_dirs):
            t, it = self.get_time(samrai_dir)
            times[i] = t
            iterations[i] = it
        return times, iterations
    
    def get_samrai_dirs(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        samrai_dirs = []
        dumps_file = os.path.join(data_dir, 'dumps.visit')
        with open(dumps_file, "r") as f:
            for line in f.readlines():
                sub_dir = line.split(sep='/')[0]
                samrai_dir = os.path.join(data_dir,sub_dir)
                samrai_dirs.append(samrai_dir)
        return samrai_dirs
    



if __name__ == '__main__':
    start = time.time()
    try:
        # data_dir, output_dir = sys.argv[1:]
        data_dir, output_file = sys.argv[1:]
    except:
        display("Invalid inputs... Using example file")
        data_dir = os.path.join('bergers2D_periodic', 'outputDir_mesh_novisc-0000')
        # output_dir = 'burgers_test_dump'
        output_file = 'test.h5'
    # os.makedirs(output_dir,exist_ok=True)
    samrai_dirs = get_samrai_dirs(data_dir)
    # output_files = get_output_files(samrai_dirs, output_dir)
    ncpu = mp.cpu_count()
    # with mp.Pool() as p:
    #     args = zip(samrai_dirs, output_files)
    #     p.starmap(merge_samrai, args)
    vars = ['u']
    # for samrai_dir, output_file in zip(samrai_dirs, output_files):
    #     merge_samrai(samrai_dir, output_file, vars=vars)
    
    samrai_reader = SamraiReader(data_dir, vars=vars)
    samrai_reader.generate_output(output_file)
    # samrai_reader.generate_output_parallel(output_file, 4)
    
    # display(samrai_dir, summary_file, output_file)
    # merge_samrai(samrai_dir, output_file, vars=vars, print_file=True)
    end = time.time()
    Time = end - start
    print("Time Elapsed:\t{}".format(Time))
    plt.close('all')
    X = samrai_reader.X
    Y = samrai_reader.Y
    Nx = samrai_reader.Nx
    Ny = samrai_reader.Ny
    # U0 = samrai_reader.vars0['u']
    print(X.shape)
    with h5py.File(output_file, 'r') as f:
        U0 = f['params']['u0'][:]
        U = f['u'][:]
    plt.figure()
    plt.pcolormesh(X, Y, U0.reshape((Nx, Ny)), cmap='jet', vmin=-1, vmax=1, shading='gouraud')
    plt.title("U0")
    plt.axis('square')
    
    plt.figure()
    plt.pcolormesh(X, Y, U[..., 0], cmap='jet', vmin=-1, vmax=1, shading='gouraud')
    plt.title('U')
    plt.axis('square')
    
    plt.show()
    
    
    
    