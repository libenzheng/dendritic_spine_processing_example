import numpy as np
import scipy.ndimage
import skimage.morphology 
import matplotlib.pyplot as plt
import pandas as pd

class ImagePreprocessing:
    
    def __init__(self,data):
        print('** Loading data')
        self.im = data.im_crop
        self.data = data
        print('** Intropolating image')
        self.interpd_im()
        print('** Normalizing image')
        self.normalize_im()
        print('** Thresholding')
        self.multi_thresholding()
        
    def interpd_im(self):
        # interpolation
        im_process  = self.im.copy()
        thickness_microns = self.data.thickness_microns
        pixel_microns = self.data.pixel_microns
        spacing_resolution = self.data.spacing_resolution
        im_resample =  self.resample3d(im_process,
                                  [thickness_microns,pixel_microns,pixel_microns],
                                  [spacing_resolution,spacing_resolution,spacing_resolution])
        im_process = im_resample.copy()
        self.im = im_process.copy()
        
    def normalize_im(self):
        # normalization
        im_process = self.im.copy()
        im_process -= im_process.min()
        im_process = np.divide(im_process,im_process.max())
        self.im = im_process.copy()
        
    def resample3d(self, image, old_spacing , new_spacing):
    # Determine current pixel spacing
        spacing = map(float, old_spacing)
        spacing = np.array(list(spacing))

        resize_x    = spacing[0] / new_spacing[0]
        new_shape_x = np.round(image.shape[0] * resize_x)
        resize_x    = float(new_shape_x) / float(image.shape[0])
        sx = spacing[0] / resize_x

        resize_y    = spacing[1] / new_spacing[1]
        new_shape_y = np.round(image.shape[1] * resize_y)
        resize_y    = new_shape_y / image.shape[1]
        sy = spacing[1] / resize_y

        resize_z    = spacing[2] / new_spacing[2]
        new_shape_z = np.round(image.shape[2] * resize_z)
        resize_z    = float(new_shape_z) / float(image.shape[2])
        sz = spacing[2] / resize_z

        image = scipy.ndimage.interpolation.zoom(image, (resize_x, resize_y, resize_z), order=1)

        return image
    
    
    def multi_thresholding(self):
        
        im_process = self.im.copy()
        
        # morphology parameters
        _closing_radius = im_process.shape[1]/17
        _selem = skimage.morphology.disk(_closing_radius)
        
        # get compute overall fixed pixel intensity threshold
        fixed_thres_coef = 2 # increased coef for better results 
        _all_pixels = im_process.reshape(-1)
        _df_pixel = pd.DataFrame(_all_pixels)
        _fixed_thres = _all_pixels.mean() + _all_pixels.std()*fixed_thres_coef
        _mean_thres = _all_pixels.mean() 
        _minimal_thres = _all_pixels.mean() - _all_pixels.std()
        
        
        # z_stack correction 
        #   -> emphasize center part of z stack and reduce effects from very top and bottom parts
        z_stack_correction_scale = 0.2

        # z-stack correction 
        n_zstack = im_process.shape[0]
        z_stack_mean_set = []
        for _i_im in range(n_zstack):
            _im = im_process[_i_im,:,:]
            _im_pixel_list = _im.reshape(-1)
            _im_pixel_list = sorted(_im_pixel_list,reverse=True)
            _top_pixel_intensity = np.mean(_im_pixel_list[:round(len(_im_pixel_list)/10)])
            z_stack_mean_set.append(_top_pixel_intensity)


        _y = np.array(z_stack_mean_set)
        _x = range(len(_y))
        _z = np.polyfit(_x, _y, 7)
        _poly = np.poly1d(_z)
        _x_fine = np.linspace(_x[0], _x[-1], 100)

        _peak_location = _x_fine[_poly(_x_fine)==_poly(_x_fine).max()][0]
        _z_stack_center_index = int(np.round(_peak_location))
        _dist_from_center = np.min([n_zstack,n_zstack - _z_stack_center_index])
        _z_stack_correction_step = z_stack_correction_scale/_dist_from_center
        
        
        # compute local threshold
        n_averaged_thres = 5
        n_zstack = im_process.shape[0]

        # compute global threshold based on trimmed average of multiotsu thresholds for each slice 
        thres_list = []
        for _i_im in range(n_zstack):
            _im = im_process[_i_im,:,:]
            _thres = skimage.filters.threshold_multiotsu(_im,classes=5,nbins=100) # compute otsu multi-thresholds # 3 classes for two thresholds
            thres_list.append(_thres)
        thres_list = np.array(thres_list)
        thres_small = thres_list[:,0] # get smaller threshold for all images 
        thres_small = np.sort(thres_small)[::-1] # descending sort
        thres_avg = thres_small[:n_averaged_thres].mean()


        
        
        # compute active contours

        _contour_size_thres = 50
        # get masked contour
        im_thres = np.zeros(im_process.shape)
        im_thres_closing = np.zeros(im_process.shape)
        for _i_im in range(n_zstack):
            _im = im_process[_i_im,:,:]

            # ------ z-stack correction --------
            _z_correction_factor = np.abs(_i_im-_z_stack_center_index)*_z_stack_correction_step
            _im = _im*(1-_z_correction_factor)

            # ------- get contour ----------
            _contour_thres = thres_avg
            _region = _im>_contour_thres
            # closing edge 
            _region[0,:]= 0 
            _region[_region.shape[0]-1,:]= 0 
            _region[:,0]= 0 
            _region[:,_region.shape[1]-1]= 0 
            _contours = skimage.measure.find_contours(_region, 0.8,fully_connected = 'high')
            _qualified_index = np.array(range(len(_contours)))[np.array([len(x) for x in _contours]) > _contour_size_thres]
            _contour_set = [_x for _i,_x in enumerate(_contours) if _i in _qualified_index]

            # ------ get masked image ----------
            # sort contour fullfill the fixed-threshold
            _qualified_contour_set = []
            for _contour in _contour_set:
                _mask = np.zeros_like(_im, dtype='bool')
                _mask[np.round(_contour[:, 0]).astype('int'), np.round(_contour[:, 1]).astype('int')] = 1
                _mask = scipy.ndimage.binary_fill_holes(_mask)
                _im_masked = _im.copy()
                _im_masked[~_mask]=0
                # compute pixel intensity for top 5% pixels 
                _im_pixel_list = _im_masked.reshape(-1)
                _im_pixel_list = sorted(_im_pixel_list,reverse=True)
                _top_pixel_intensity = np.mean(_im_pixel_list[:round(len(_im_pixel_list)/20)])
                if (_im_masked.max()>= _fixed_thres)&(_top_pixel_intensity>_minimal_thres):
                    _qualified_contour_set.append(_contour)


            _im_thres_mask = np.zeros_like(_im, dtype='bool')
            for _contour in _qualified_contour_set:
                _im_thres_mask[np.round(_contour[:, 0]).astype('int'), np.round(_contour[:, 1]).astype('int')] = 1
                _im_thres_mask = scipy.ndimage.binary_fill_holes(_im_thres_mask)

             # morphological process
            _im_closing = skimage.morphology.closing(_im_thres_mask, _selem)

            im_thres[_i_im,:,:] = _im_thres_mask
            im_thres_closing[_i_im,:,:] = _im_closing
            
            
            self.im_thres = im_thres.copy()
            self.im_thres_closing = im_thres_closing.copy()
            
            
    def plot_thresholding(self):
        
        im_process = self.im
        im_thres = self.im_thres
        im_thres_closing = self.im_thres_closing
        n_zstack = im_process.shape[0]
        
        fig,axs = plt.subplots(n_zstack,4,figsize=[4,n_zstack*1],dpi=150)
        for _i_im in range(n_zstack):

            _ax = axs[_i_im,0]

            _ax.imshow(im_process[_i_im,:,:],
                       cmap='Greys_r',
                       vmin = im_process.min(),
                       vmax=im_process.max())

        #     _depth = z_depth[_i_im]
        #     _ax.set_ylabel(str(_depth)[0:5] +' um')
            _ax.set_xticks([])
            _ax.set_yticks([])

            _ax = axs[_i_im,1]
            _ax.axis('off')
            _ax.imshow(im_thres[_i_im,:,:],
                       cmap='Greys_r')


            _ax = axs[_i_im,2]
            _ax.axis('off')
            _im_closing = im_thres_closing[_i_im,:,:]
            _ax.imshow(_im_closing,
                       cmap='Greys_r')

            _ax = axs[_i_im,3]
            _ax.axis('off')
            _ax.imshow(im_process[_i_im,:,:],
                       cmap='Greys_r',
                       vmin = im_process.min(),
                       vmax=im_process.max())

            _contour_set = skimage.measure.find_contours(im_thres_closing[_i_im,:,:], 0.8,fully_connected = 'high')
            for _contour in _contour_set:
                _ax.plot(_contour[:, 1], _contour[:, 0], linewidth=1,c='r',ls = ':')



        axs[0,0].set_title('raw')
        axs[0,1].set_title('thres.')
        axs[0,2].set_title('closing')
        axs[0,3].set_title('contour')
            
