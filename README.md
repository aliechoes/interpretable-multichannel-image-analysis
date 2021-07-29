# interpretable-cnn

Here you can find the codes and documentations regarding interpretable multichannel image analysis

## Data structure

The images should be saved in h5 format. Each file should contain these keys:
- `image` (16-bit np.array (h,w,c))
- `mask` (np.array (h,w,c), optional)
- `label` (str, optional)
- `donor` (str, optional) 
- `experiment`(str, optional)
- `channels` (list, optional)
Also, the name of the file should be the object number in the `.cif` file
