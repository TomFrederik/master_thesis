import h5py

f = h5py.File('pong_data.hdf5', 'w')
f.create_dataset('done', dtype=int)
f.create_dataset('obs', dtype=float)
f.create_dataset('action', dtype=int)
f.create_dataset('reward', dtype=float)
print(list(f.keys()))
# dset = h5py.Group.create_dataset("trajs")