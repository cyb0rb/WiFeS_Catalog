import matplotlib.pyplot as plt
from astropy.io import fits as fits

import pkg_resources
pkg_resources.require("numpy==1.26.3")
import numpy as np

from dl import queryClient as qc
import pandas as pd
from scipy.spatial import distance_matrix

# timer function
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


class SkyCatalogue():
    
    @timer
    def __init__(self, query_dist=1.0, map_dist=1.0, mask_radius=20, fov=45):
        
        self.query_dist = query_dist
        self.map_dist = map_dist
        self.dim = int((3600*4) * self.map_dist)
        self.mask_radius = mask_radius
        self.fov = fov
        
        # load all masked stars
        print("Loading masked star data....")
        self.load_mask_data()
        
        # define grid stuff
        print("Defining grid lines...")
        self.define_grid()
        
        pass

    @timer
    def query_tractor(self, ra, dec, dist=1.0, all_bands=True):
        """Queries the Astro Data Lab for the ra, dec and mag of the objects within a square of side length (dist).     
        dist is in degrees
        """
        # Bounds of the square we are querying objects for
        ra_min=ra
        ra_max = ra + dist
        dec_min=dec
        dec_max = dec + dist

        if all_bands:
            query = f"""
            SELECT ra, dec, mag_g,mag_r,mag_i,mag_z
            FROM ls_dr10.tractor_s
            WHERE ra >= ({ra_min}) AND ra < ({ra_max})
            AND dec >= ({dec_min}) AND dec < ({dec_max})
            AND (mag_g<=21 AND mag_g>=16
                OR mag_r<=21 AND mag_r>=16
                OR mag_i<=21 AND mag_i>=16
                OR mag_z<=21 AND mag_z>=16)       
            """
        else:
            query = f"""
            SELECT ra, dec, mag_g,mag_r,mag_i,mag_z
            FROM ls_dr10.tractor_s
            WHERE ra >= ({ra_min}) AND ra < ({ra_max})
            AND dec >= ({dec_min}) AND dec < ({dec_max})
            AND mag_g<=21 AND mag_g>=16     
            """
        
        # check if this completes successfuly
        brick_info = qc.query(sql=query, fmt="pandas")

        mag = []
        passband = []
        brick_info = brick_info.replace([np.inf], np.nan)

        for n in range(len(brick_info['mag_g'])):
            if pd.notna(brick_info['mag_g'][n]):
                mag.append(brick_info['mag_g'][n])
                passband.append('g')
            elif pd.notna(brick_info['mag_r'][n]):
                mag.append(brick_info['mag_r'][n])
                passband.append('r')
            elif pd.notna(brick_info['mag_i'][n]):
                mag.append(brick_info['mag_i'][n])
                passband.append('i')
            elif pd.notna(brick_info['mag_z'][n]):
                mag.append(brick_info['mag_z'][n])
                passband.append('z')
            else:
                brick_info = brick_info.drop(brick_info.iloc['mag'][n])

        brick_info = brick_info.drop(['mag_g','mag_r','mag_i','mag_z'], axis=1)
        brick_info['mag'] = mag
        brick_info['passband'] = passband

        return brick_info
    
    @timer
    def load_mask_data(self):
        """Load all of the mask data files. 
        Returns a pandas Dataframe with columns 'ra', 'dec', 'radius'
        """

        all_masks = []
        for i in range(5):
            with np.load(f"mask_data_files/mask_data_{i}.npz", mmap_mode='r') as mask_data:
                mask_array = mask_data['arr_0']
                mask_array_byteswap = mask_array.byteswap().newbyteorder()
                masked_stars = pd.DataFrame(mask_array_byteswap)
                all_masks.append(masked_stars)
                
        self.mask_df = pd.concat(all_masks, ignore_index=True)
    
    @timer
    def calculate_mask_radius(self, mag):
        return (self.mask_radius/3600) + 1630./3600. * 1.396**(-mag)
    
    @timer
    def combine_data(self, catalog_stars:pd.DataFrame, coords):
        """Combines the data from masked and catalog stars based on some coordinate range"""
        # coords = [ra, ra+map_dist, dec, dec+map_dist]
        
        # cut masked stars to only use the same area as catalog_stars
        masked_box = self.mask_df.query('(@coords[0] < ra < @coords[1]) and (@coords[2] < dec < @coords[3])')
        catalog_box = catalog_stars.query('(@coords[0] < ra < @coords[1]) and (@coords[2] < dec < @coords[3])').copy()
        
        # apply buffer radius to mask and star data
        masked_box.loc[:, 'radius'] = masked_box['radius'] + (self.mask_radius / 3600.)
        # TODO check for nan's / inf
        catalog_box.loc[:, 'radius'] = self.calculate_mask_radius(catalog_box.loc[:,'mag'])
        # print(catalog_box['radius'].isna().sum())
        
        # remove g mag
        catalog_box = catalog_box.drop(['mag','passband'], axis=1)
        
        # combine catalog + mask
        all_stars = pd.concat([masked_box, catalog_box]).reset_index(drop=True)
        return all_stars
    
    @timer
    def create_pixel_columns(self, all_stars:pd.DataFrame, coords):
        """Creates columns for min and max ra and dec for all stars in the dataframe"""
        # coords: [ra, ra+map_dist, dec, dec+map_dist]
        
        # find max and min ra/dec corresponding to the mask of star
        all_stars['max_ra'] = all_stars['ra'] + all_stars['radius']
        all_stars['min_ra'] = all_stars['ra'] - all_stars['radius']
        all_stars['max_dec'] = all_stars['dec'] + all_stars['radius']
        all_stars['min_dec'] = all_stars['dec'] - all_stars['radius']
        
        # boolean for radii that go above 1-degree integer RA/DEC bounds
        expression = '(max_ra > ceil(ra)) | (min_ra < floor(ra)) | (max_dec > ceil(dec)) | (min_dec < floor(dec))'
        all_stars['overlap'] = all_stars.eval(expression)
        
        # ra, dec, and radius in pixels
        # TODO check if off by one is needed?
        all_stars['ra_pix'] = np.round((all_stars['ra'] - coords[0]) * self.dim).astype(int) - 1
        all_stars['dec_pix'] = np.round((all_stars['dec'] - coords[2]) * self.dim).astype(int) - 1
        all_stars['rad_pix'] = np.ceil(all_stars['radius'] * self.dim).astype(int)
        
        all_stars['min_ra_pix'] = all_stars['ra_pix'] - all_stars['rad_pix']
        all_stars['max_ra_pix'] = all_stars['ra_pix'] + all_stars['rad_pix']
        all_stars['min_dec_pix'] = all_stars['dec_pix'] - all_stars['rad_pix']
        all_stars['max_dec_pix'] = all_stars['dec_pix'] + all_stars['rad_pix']
        
        # set stars outside of map range to that value
        all_stars.loc[all_stars['min_ra_pix'] < 0, 'min_ra_pix'] = 0
        all_stars.loc[all_stars['max_ra_pix'] > self.dim, 'max_ra_pix'] = self.dim
        all_stars.loc[all_stars['min_dec_pix'] < 0, 'min_dec_pix'] = 0
        all_stars.loc[all_stars['max_dec_pix'] > self.dim, 'max_dec_pix'] = self.dim
        
        return all_stars
    
    @timer
    def seg_map(self, star_data:pd.DataFrame):
        """Creates segementation map of shape (`dim`, `dim`) based on the mask locations and pixel data of `star_data`"""

        array = np.zeros((self.dim, self.dim), dtype=int)
        array.flatten()
        
        for star in star_data.to_dict('records'):
            
            # center pixel to determine distance from
            center = [[star['dec_pix'], star['ra_pix']]]
            
            # make array of indexes
            # TODO add check of dimension sign to make sure it's not negative
            chunk = np.indices((star['max_dec_pix'] - star['min_dec_pix'], star['max_ra_pix'] - star['min_ra_pix']))
            
            # adjust indices to correspond to the larger grid
            # coord grid is shaped like [ [x1, y1], [x1, y2], ... [x1, yn], [x2, y1], ... [xn, yn] ]
            coord_grid = np.dstack((chunk[0]+star['min_dec_pix'], chunk[1]+star['min_ra_pix']))
            coord_grid = np.concatenate(coord_grid, axis=0)
            
            # calculate distances of each pixel coordinate to the center pixel
            distances = distance_matrix(x=coord_grid, y=center)

            # change all values of the segmap array to 0 where distances are < mask radius
            np.place(array[star['min_dec_pix']:star['max_dec_pix'], star['min_ra_pix']:star['max_ra_pix']], distances < star['rad_pix'], 1)

        array.reshape((self.dim, self.dim))
        return array
    
    @timer
    def define_grid(self):
        """Creates gridlines and centers on pixels for the initialized dimension and field of view"""
        self.gridlines = np.arange(0, self.dim+1, (self.fov/3600 * self.dim))
        centers = []

        for i in range(len(self.gridlines[:-1])):
            centers.append(int((self.gridlines[i] + self.gridlines[i+1])/2 + 0.5))

        self.x_cen, self.y_cen = np.meshgrid(centers, centers)
        return
    
    @timer
    def find_dark_regions(self, segmap):

        dark_regions = []

        for i in range(len(self.gridlines) - 1):
            for j in range(len(self.gridlines) - 1):
                x_start, x_end = (self.gridlines[i]).astype(int), (self.gridlines[i + 1]).astype(int)
                y_start, y_end = (self.gridlines[j]).astype(int), (self.gridlines[j + 1]).astype(int)
                
                if np.all(segmap[y_start:y_end, x_start:x_end] == 0):
                    dark_regions.append([self.x_cen[j, i], self.y_cen[j, i]])

        dr_trans = np.array(dark_regions).transpose()

        return dr_trans, dark_regions

    @timer
    def create_plot(self, array, coords, pix_coords, dr_trans):

        # Creating exclusion map with grid
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

        plt.imshow(array, origin = 'lower', cmap='gray', vmin=0, vmax=1)

        positions = np.linspace(0,self.dim,5)
        x_labels = np.linspace(coords[0],coords[1],5)
        y_labels = np.linspace(coords[2],coords[3],5)

        ax.set_xticks(positions, x_labels)
        ax.set_yticks(positions, y_labels)    

        plt.vlines(self.gridlines, min(pix_coords[1]), max(pix_coords[1]), color='red', linewidth=1)
        plt.hlines(self.gridlines, min(pix_coords[0]), max(pix_coords[0]), color= 'red', linewidth=1)

        plt.plot(dr_trans[0], dr_trans[1], 'rx', markersize=10)

        plt.tight_layout()
        plt.margins(0)
        plt.show()

        return
    
    @timer
    def create_data_frame(self, dark_regions, coords):
        dark_ra = []
        dark_dec = []

        for i in dark_regions:
            ra = i[0] / (self.dim) + coords[0]
            dec = i[1] / (self.dim) + coords[2]
            dark_ra.append(ra)
            dark_dec.append(dec)

        dark_catalogue = pd.DataFrame({'ra':dark_ra, 'dec':dark_dec})
        return dark_catalogue
    
    @timer
    def find_overlapping_extent(self, all_stars):
        # grab everything in a 1 degree square
        # print(f"Finding everything within the square RA=({ra}, {ra+1}) and DEC=({dec}, {dec+1})")
        
        # degree_masks = all_stars.query(f'({ra} < ra < {ra+1}) & ({dec} < dec < {dec+1})')

        min_ra = all_stars['min_ra'].min()
        min_dec = all_stars['min_dec'].min()
        max_ra = all_stars['max_ra'].max()
        max_dec = all_stars['max_dec'].max()
        
        return [min_ra, min_dec, max_ra, max_dec]

    @timer
    def create_degree_square(self, ra, dec, catalog_df, plot_image=False):
        """Generates dark sky positions for a 1 x 1 degree region of the sky with lower "corner" given by (ra,dec)
        """
        
        coords = [ra, ra+1, dec, dec+1]
        print("Combining mask and queried stars...")
        all_stars = self.combine_data(catalog_df, coords)
        print("Calculating pixel values for stars....")
        all_stars = self.create_pixel_columns(all_stars, coords)

        print("Creating segmentation map...")
        segmentation_map = self.seg_map(all_stars)
        
        print("Finding dark regions...")
        dr_trans, dark_regions = self.find_dark_regions(segmentation_map)

        if plot_image:
            print("Plotting dark regions...")
            pix_coords = [all_stars['ra_pix'], all_stars['dec_pix'], all_stars['rad_pix']]
            self.create_plot(segmentation_map, coords, pix_coords, dr_trans)

        print("Converting dark regions to coordinates...")
        dark_catalogue = self.create_data_frame(dark_regions, coords)
        
        print("Finding overlaps...")
        overlap = self.find_overlapping_extent(all_stars)
        
        print("Done!")
        return dark_catalogue, overlap
    
    @timer
    def remove_overlap_positions(self, ra_coords, dec_coords, overlap_store, larger_catalogue, bounds=1):
        # overlap_store = [ [minra, mindec, maxra, maxdec] ]
        catalogue = larger_catalogue.copy()

        # for each ra / dec square and associated overlap
        for ra, dec, overlap in zip(ra_coords, dec_coords, overlap_store):
            
            # make sure this works if ra/dec extents are less than the actual bounds!
            min_ra = overlap[0] if overlap[0] < ra else ra
            min_dec = overlap[1] if overlap[1] < dec else dec
            max_ra = overlap[2] if overlap[2] > ra+bounds else ra+bounds
            max_dec = overlap[3] if overlap[3] > dec+bounds else dec+bounds
            
            # print(f"overlap_store: {overlap}")
            # print(f"min/max ra/dec: {min_ra}, {min_dec}, {max_ra}, {max_dec}")
            # everything within square bounded by ra / dec and bounds that you're checking
            smaller_box = catalogue.query(f'({ra} < ra < {ra + bounds}) & ({dec} < dec < {dec + bounds})') 
            # print(f'Small box: ({ra} < ra < {ra + bounds}) & ({dec} < dec < {dec + bounds})')
            # print(smaller_box.shape)
            
            # select everything within square bounded by the min/max ra/dec of the overlap store
            bigger_box = catalogue.query(f'({min_ra} <= ra <= {max_ra}) & ({min_dec} <= dec <= {max_dec})')
            # print(f'Large box: ({min_ra} <= ra <= {max_ra}) & ({min_dec} <= dec <= {max_dec})')
            # print(bigger_box.shape)
            
            # only perform removal if there are actually sky positions in the "overlap region"
            if smaller_box.shape < bigger_box.shape:
                # everything that is in the bigger box but not the smaller one (within overlapping region)
                overlap_region = pd.concat((bigger_box, smaller_box)).drop_duplicates(keep=False)
                # concatenate the two and drop everything within the overlapping regions
                catalogue = pd.concat((catalogue, overlap_region)).drop_duplicates(keep=False)
            
            

        # for each square w/ corner at ra, dec and associated overlap_store index
        # for ra,dec,i in zip(ra_coords,dec_coords,range(len(overlap_store))):
            
        #     # for each coordinate pair in the catalog
        #     for x,y in zip(catalogue['ra'],catalogue['dec']):
                
        #         # if catalog ra falls within 1 degree ra bounds
        #         if x>ra and x<(ra+bounds):
        #             # if mindec < catalog dec < lower dec bound OR upper dec bound < catalog dec < maxdec
        #             if (y<=dec and y>=overlap_store[i][1]) or (y>=(dec+bounds) and y<=overlap_store[i][3]):
        #                 # select index of anything with that exact catalog dec
        #                 j = catalogue[(catalogue.dec == y)].index
        #                 # drop that index from catalog
        #                 catalogue.drop(np.array(j),inplace=True)
                
        #         # if catalog dec falls within 1 degree dec bounds
        #         if y>dec and y<(dec+bounds):
        #             # if minra < catalog ra < lower ra bound OR upper ra bound < catalog ra < maxra
        #             if (x<=ra and x>overlap_store[i][0]) or (x>=(ra+bounds) and x<=overlap_store[i][2]):
        #                 # select index of anything with that exact catalog ra
        #                 j = catalogue[(catalogue.ra == x)].index
        #                 catalogue.drop(np.array(j),inplace=True)
                        
        return catalogue
        
    @timer
    def create_catalogue(self, ra, dec, query_dist=1.0, plot_image=False, allsky=False):
        
        
        # query sky for some amount
        print(f"Querying the tractor catalog for stars from RA/DEC({ra}, {dec}) to ({ra+query_dist}, {dec+query_dist})...")
        query_df = self.query_tractor(ra, dec, query_dist, all_bands=False)
        
        # make array of ra / dec starting points for degree cubes
        dec_range = np.arange(dec, dec+query_dist)
        ra_range = np.arange(ra, ra+query_dist)
        
        coord_grid = np.meshgrid(ra_range, dec_range)
        ra_coords = coord_grid[0].flatten()
        dec_coords = coord_grid[1].flatten()
        overlap_store = []
        larger_catalogue = pd.DataFrame(columns=['ra','dec'])
        
        # catalogue = self.create_degree_square(ra, dec, query_df)
        
        print("Generating sky catalog...")
        for ra_c, dec_c in zip(ra_coords,dec_coords):
            print(f"> Generating sky catalog for square RA,DEC ({ra_c}, {dec_c}) to ({ra_c+1}, {dec_c+1})..")
            cat, overlap = self.create_degree_square(ra_c, dec_c, query_df, plot_image)
            larger_catalogue = pd.concat([larger_catalogue,cat],axis=0).reset_index(drop=True)
            overlap_store.append(overlap)
            # print('Added (' + str(ra) + ', ' + str(dec) + ') to catalogue')
        
        print("Removing positions from overlapping regions...")
        catalogue = self.remove_overlap_positions(ra_coords, dec_coords, overlap_store, larger_catalogue)
        
        if allsky:
            print(f"Finding largest overlap for whole {query_dist}-degree square...")
            overlap_store = np.asarray(overlap_store)
            min_ra = np.min(overlap_store[:, 0])
            min_dec = np.min(overlap_store[:, 1])
            max_ra = np.max(overlap_store[:, 2])
            max_dec = np.max(overlap_store[:, 3])
            print(f"min RA/DEC = ({min_ra}, {min_dec})    max RA/DEC = ({max_ra}, {max_dec})")
            return catalogue, [min_ra, min_dec, max_ra, max_dec]
        
        return catalogue
    
    def all_sky(self, query_dist=5.0, max_dec=30, max_ra=360):
        """Loop through the entire sky."""
        
        print("================= WHOLE SKY =================")
        # use 5 degree squares
        
        dec_range = np.arange(-90, max_dec, query_dist)
        ra_range = np.arange(0, max_ra, query_dist)
        # dec_range = np.arange(-90, 30, query_dist)
        # ra_range = np.arange(0, 360, query_dist)
        
        coord_grid = np.meshgrid(ra_range, dec_range)
        ra_coords = coord_grid[0].flatten()
        dec_coords = coord_grid[1].flatten()
        overlap_store = []
        larger_catalogue = pd.DataFrame(columns=['ra','dec'])
        
        for ra_c, dec_c in zip(ra_coords, dec_coords):
            print(f"{query_dist}-degree square starting from RA,DEC = {ra_c}, {dec_c} ========================")
            cat, overlap = self.create_catalogue(ra_c, dec_c, query_dist=query_dist, allsky=True)
            larger_catalogue = pd.concat([larger_catalogue,cat],axis=0).reset_index(drop=True)
            overlap_store.append(overlap)
            
        print("WHOLE SKY: Removing positions from overlapping regions...")
        catalogue = larger_catalogue
        catalogue = self.remove_overlap_positions(ra_coords, dec_coords, overlap_store, larger_catalogue, bounds=query_dist)
        print("================= Done! =================")
        return catalogue, ra_coords, dec_coords, overlap_store, larger_catalogue
        
        
        
if __name__=="__main__":
    
    catalog = SkyCatalogue()
    positions = catalog.all_sky(query_dist=30.0)
    # positions = catalog.create_catalogue()