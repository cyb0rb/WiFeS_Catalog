# imports
import matplotlib.pyplot as plt
from astropy.io import fits as fits
from astropy.coordinates import SkyCoord

import pkg_resources
pkg_resources.require("numpy==1.26.3")
import numpy as np

from dl import queryClient as qc
import pandas as pd
from sklearn.neighbors import KDTree

class SkyCatalogue():
    
    # @timer
    def __init__(self, mode="corner", bands=('g','r','i','z'), map_dist=1.0, mask_radius=20, fov=45):
        
        self.bands = bands
        self.map_dist = map_dist
        self.dim = int((3600*4) * self.map_dist)
        self.mask_radius = mask_radius
        self.fov = fov
        self.mode= mode
        
        # load all masked stars
        print("Loading masked star data....")
        self.load_mask_data()
        
        # define grid stuff
        print("Defining grid lines...")
        self.define_grid()
        
        # create distance grid for this dimension
        print("Creating KDTree for distance calculations...")
        self.dist_array = np.indices((self.dim, self.dim), dtype=int)
        self.dist_array = np.dstack((self.dist_array[0], self.dist_array[1]))
        self.dist_array = np.concatenate(self.dist_array, axis=0)
        self.distance_tree = KDTree(self.dist_array)
        print("KDTree created!")
        
        pass

    def galactic_check(self, ra, dec, dist):
        """Check if any of a square with side length `dist` and a centre coordinate (ra,dec) has
        any intersection with the galactic plane (|b| <= 19) or the LMC/SMC

        Parameters
        ----------
        ra : `float`
            Right ascension of the centre of the square (degrees)
        dec : `float`
            Declination of the centre of the square (degrees)
        dist : `float'
            Side length of square region to query (degrees)

        Returns
        -------
        `bool`
            True if region is safe, False otherwise
        """
        if self.mode=="corner":
            ra_min=ra
            ra_max = ra + dist
            dec_min=dec
            dec_max = dec + dist

        if self.mode=="centre":
            ra_min=ra-dist/2
            ra_max = ra + dist/2
            dec_min=dec-dist/2
            dec_max = dec + dist/2

        # check if in LMC
        if (ra_min >=76) and (ra_max <= 86) and (dec_min >= -76) and (dec_max <= -64):
            return False

        # check if in SMC
        if (ra_min >=11) and (ra_max <= 16) and (dec_min >= -76) and (dec_max <= -70):
            return False    
        
        # check if in the gap near the galactic plane
        if (ra_min >=43) and (ra_max <= 75) and (dec_min >= 10) and (dec_max <= 30):
            return False

        # check if on the galactic plane
        c_icrs_min = SkyCoord(ra=ra_min, dec=dec_min, frame='icrs', unit='degree')
        c_icrs_max = SkyCoord(ra=ra_max, dec=dec_max, frame='icrs', unit='degree')

        c_gal_min = c_icrs_min.galactic
        c_gal_max = c_icrs_max.galactic

        if abs(c_gal_min.b.value) <= 19 or abs(c_gal_max.b.value) <= 19:
            return False

        return True

    
    def query_tractor(self, ra, dec, dist, bands):
        """Queries the Astro Data Lab for the ra, dec and mag of the objects within a square of side length (dist).     
        The queried square will range from (ra, dec) to (ra+dist/2, dec+dist/2)
        
        Parameters
        ----------
        ra: `float`
            Right ascension of the centre of the square (degrees)
        dec: `float`
            Declination of bottom left corner of square (degrees)
        dist: `float`
            Side length of square region to query (degrees)
        bands: `tuple` `str`
            Bands to query for objects from the selection of ('g', 'r', 'i', 'z')
            Where objects are detected in multiple bands, the one with the brightest magnitude will be selected
        
        Returns
        -------
        brick_info: `pd.DataFrame`
            Pandas DataFrame containing columns: `ra`, `dec`, `mag`, `passband`
        """
        # Bounds of the square we are querying objects for based on the mode
        if self.mode=="corner":
            ra_min=ra
            ra_max = ra + dist
            dec_min=dec
            dec_max = dec + dist

        if self.mode=="centre":
            ra_min=ra-dist/2
            ra_max = ra + dist/2
            dec_min=dec-dist/2
            dec_max = dec + dist/2
      

        mags = [f"mag_{b}" for b in self.bands]
        conditions = [f"({mag}<=21 AND {mag}>=16)" for mag in mags]
        
        query = f"""
            SELECT ra, dec, {", ".join(mags)}
            FROM ls_dr10.tractor_s
            WHERE ra >= ({ra_min}) AND ra < ({ra_max})
            AND dec >= ({dec_min}) AND dec < ({dec_max})
            AND ({" OR ".join(conditions)})
            """
        
        # check if this completes successfuly
        brick_info = qc.query(sql=query, fmt="pandas")

        # set "mag" column to the minimum magnitude from the given bands
        brick_info.loc[:,"mag"] = np.nanmin(brick_info.iloc[:,2:].values, axis=1)
        brick_info = brick_info.drop(mags, axis=1)

        return brick_info
    
    def load_mask_data(self):
        """Load all files containing mask data.
        Circular masks are set for detected PSF object (stars), galaxies, and globular clusters/planetary nebulae.
        """

        all_masks = []

        # load PSF (stars) masks
        for i in range(5):
            with np.load(f"mask_data_files/mask_data_{i}.npz", mmap_mode='r') as mask_data:
                mask_array = mask_data['arr_0']
                mask_array_byteswap = mask_array.byteswap().newbyteorder()
                masked_stars = pd.DataFrame(mask_array_byteswap)
                all_masks.append(masked_stars)

        # load globular clusters and planetary nebulae masks
        with np.load(f"mask_data_files/mask_data_clusters.npz", mmap_mode='r') as mask_data:
            mask_array = mask_data['arr_0']
            mask_array_byteswap = mask_array.byteswap().newbyteorder()
            masked_stars = pd.DataFrame(mask_array_byteswap)
            all_masks.append(masked_stars)
        
        # load galaxy masks
        with np.load(f"mask_data_files/mask_data_galaxies.npz", mmap_mode='r') as mask_data:
            mask_array = mask_data['arr_0']
            mask_array_byteswap = mask_array.byteswap().newbyteorder()
            masked_stars = pd.DataFrame(mask_array_byteswap)
            all_masks.append(masked_stars)
            
        self.mask_df = pd.concat(all_masks, ignore_index=True)

    
    def calculate_mask_radius(self, mag):
        """Calculate masking radius (in degrees) for an object given some magnitude `mag` with
        a minimum size based on initialized mask radius (default=20 arcsec).
        
        This function is modified from the [legacypipe `mask_radius_for_mag()` function](https://github.com/legacysurvey/legacypipe/blob/DR10.0.12/py/legacypipe/reference.py#L352-L357)

        Parameters
        ----------
        mag : `float`
            Magnitude (in g, r, i or z band) of object

        Returns
        -------
        `float`
            Radius of mask associated with object (degrees)
        """
        return (self.mask_radius/3600) + 1630./3600. * 1.396**(-mag)
    
    def combine_data(self, catalog_stars:pd.DataFrame, coords):
        """Combines the downloaded mask data from load_mask_data with calculated masks from
        calculate_mask_radius over the specified coordinates into a single dataframe.

        Parameters
        ----------
        catalog_stars : `pd.DataFrame`
            Stars queried directly from the tractor catalogue with columns ra, dec, radius
        coords : `list`
            Minimum and maximum right ascensions and declinations of regions queried from the tractor catalogue

        Returns
        -------
        all_stars : `pd.DataFrame'
            Combined dataframe of all stars with associated mask data with columns ra, dec, radius
        """
        
        # cut masked stars to only use the same area as catalog_stars
        masked_box = self.mask_df.query('(@coords[0] < ra < @coords[1]) and (@coords[2] < dec < @coords[3])')
        catalog_box = catalog_stars.query('(@coords[0] < ra < @coords[1]) and (@coords[2] < dec < @coords[3])').copy()
        
        # apply buffer radius to mask and star data
        masked_box.loc[:, 'radius'] = masked_box['radius'] + (self.mask_radius / 3600.)
        # TODO check for nan's / inf        
        catalog_box.loc[:, 'radius'] = self.calculate_mask_radius(catalog_box.loc[:,'mag'])
        # print(catalog_box['radius'].isna().sum())
        
        # remove g mag
        catalog_box = catalog_box.drop(['mag'], axis=1)
        
        # combine catalog + mask
        all_stars = pd.concat([masked_box, catalog_box]).reset_index(drop=True)
        return all_stars
    
    def create_pixel_columns(self, all_stars:pd.DataFrame, coords):
        """Converts coorindate data into a pixel grid and calculates pixel values of all objects.

        Parameters
        ----------
        all_stars : `pd.DataFrame`
            Combined dataframe of all stars with associated mask data with columns ra, dec, radius

        Returns
        -------
        all_stars : `pd.DataFrame`
            Updated dataframe with coordinate conversions to pixels and minimum and maximum coordinate values in pixels
            Contains columns ra, dec, radius, max_ra, min_ra, max_dec, min_dec, ra_pix, dec_pix, rad_pix
        """
        
        # find max and min ra/dec corresponding to the mask of star
        all_stars['max_ra'] = all_stars['ra'] + all_stars['radius']
        all_stars['min_ra'] = all_stars['ra'] - all_stars['radius']
        all_stars['max_dec'] = all_stars['dec'] + all_stars['radius']
        all_stars['min_dec'] = all_stars['dec'] - all_stars['radius']

        # ra, dec, and radius in pixels
        # TODO check if off by one is needed?
        all_stars['ra_pix'] = np.round((all_stars['ra'] - coords[0]) * self.dim).astype(int) - 1
        all_stars['dec_pix'] = np.round((all_stars['dec'] - coords[2]) * self.dim).astype(int) - 1
        all_stars['rad_pix'] = np.ceil(all_stars['radius'] * self.dim).astype(int)

        return all_stars
    
    def seg_map(self, star_data:pd.DataFrame):
        """Creates segementation map of shape (`dim`, `dim`) based on the mask locations and pixel data of `star_data`"""

        array = np.zeros(self.dim**2, dtype=int)
        radec = np.asarray([star_data['dec_pix'],star_data['ra_pix']]).T
        circle_points = self.distance_tree.query_radius(radec, star_data['rad_pix'])

        for circle_array in circle_points:
            np.put(array, circle_array, 1)
        
        array = array.reshape((self.dim, self.dim))

        return array
    
    def define_grid(self):
        """Creates gridlines and centers on pixels for the initialized dimension and field of view"""
        self.gridlines = np.arange(0, self.dim+1, (self.fov/3600 * self.dim))
        centers = []

        for i in range(len(self.gridlines[:-1])):
            centers.append(int((self.gridlines[i] + self.gridlines[i+1])/2 + 0.5))

        self.x_cen, self.y_cen = np.meshgrid(centers, centers)
        return
    
    # @timer
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

    # @timer
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
    
    # @timer
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
    
    # @timer
    def find_overlapping_extent(self, all_stars):
        # grab everything in a 1 degree square
        # print(f"Finding everything within the square RA=({ra}, {ra+1}) and DEC=({dec}, {dec+1})")
        
        # degree_masks = all_stars.query(f'({ra} < ra < {ra+1}) & ({dec} < dec < {dec+1})')

        min_ra = all_stars['min_ra'].min()
        min_dec = all_stars['min_dec'].min()
        max_ra = all_stars['max_ra'].max()
        max_dec = all_stars['max_dec'].max()
        
        return [min_ra, min_dec, max_ra, max_dec]

    # @timer
    def create_degree_square(self, ra, dec, catalog_df, plot_image=False):
        """Generates dark sky positions for a 1 x 1 degree region of the sky with lower "corner" given by (ra,dec)
        """
        if self.mode=="corner":
            coords = [ra, ra+self.map_dist, dec, dec+self.map_dist]
        if self.mode=="centre":
            coords=[ra-self.map_dist/2, ra+self.map_dist/2, dec-self.map_dist/2, dec+self.map_dist/2]
        # print(">>>> Generating dark sky positions of 1-degree square...")
        print(">>>> Combining mask and queried stars...")
        all_stars = self.combine_data(catalog_df, coords)
        print(">>>> Calculating pixel values for stars....")
        all_stars = self.create_pixel_columns(all_stars, coords)

        print(">>>> Creating segmentation map...")
        segmentation_map = self.seg_map(all_stars)
        
        print(">>>> Finding dark regions in segmentation map...")
        dr_trans, dark_regions = self.find_dark_regions(segmentation_map)

        if plot_image:
            print(">>>> Plotting dark regions...")
            pix_coords = [all_stars['ra_pix'], all_stars['dec_pix'], all_stars['rad_pix']]
            self.create_plot(segmentation_map, coords, pix_coords, dr_trans)

        print(">>>> Converting dark regions to coordinates...")
        dark_catalogue = self.create_data_frame(dark_regions, coords)
        
        print(">>>> Finding maximum extent of stars beyond the degree-square bounds...")
        overlap = self.find_overlapping_extent(all_stars)
        
        print(">>>> Done!")
        return dark_catalogue, overlap

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
            
            # everything within square bounded by ra / dec and bounds that you're checking
            smaller_box = catalogue.query(f'({ra} < ra < {ra + bounds}) & ({dec} < dec < {dec + bounds})') 
            
            # select everything within square bounded by the min/max ra/dec of the overlap store
            bigger_box = catalogue.query(f'({min_ra} <= ra <= {max_ra}) & ({min_dec} <= dec <= {max_dec})')
            
            # only perform removal if there are actually sky positions in the "overlap region"
            if smaller_box.shape < bigger_box.shape:
                # everything that is in the bigger box but not the smaller one (within overlapping region)
                overlap_region = pd.concat((bigger_box, smaller_box)).drop_duplicates(keep=False)
                # concatenate the two and drop everything within the overlapping regions
                catalogue = pd.concat((catalogue, overlap_region)).drop_duplicates(keep=False)
                        
        return catalogue
        
    # @timer
    def create_catalogue(self, ra, dec, query_dist, plot_image=False, allsky=False):
        """Creates catalog of sky positions in a square starting from a bottom-left corner of (ra, dec)
        up to (ra+query_dist, dec+query_dist) using a single query to the LSDR10 tractor catalog.
        
        Parameters
        ----------
        ra: `float`
            Right ascension of bottom left corner of square (degrees)
        dec: `float`
            Declination of bottom left corner of square (degrees)
        query_dist: `float`
            Side length of square to query (degrees)
        plot_image: `bool`
            Plot each square-degree analyzed for dark sky positions
        all_sky: `bool`
            Whether to return a list of min/max ra/dec radial extents for stars in the square
            (used for all-sky generation)
        
        Returns
        -------
        catalogue: `DataFrame`
            DataFrame of dark sky positions containing columns: `ra`, `dec`
        """
        if self.mode=="corner":
            print(f"> Creating sky catalog from one {query_dist}-degree square starting from ({ra}, {dec}) to ({ra+query_dist}, {dec+query_dist})")
            # query sky for some amount
            print(f">> Querying the tractor catalog for stars from RA/DEC({ra}, {dec}) to ({ra+query_dist}, {dec+query_dist})...")
            query_df = self.query_tractor(ra, dec, query_dist)
            # make array of ra / dec starting points for degree cubes
            dec_range = np.arange(dec, dec+query_dist)
            ra_range = np.arange(ra, ra+query_dist)

        if self.mode=="centre":
            print(f"> Creating sky catalog from one {query_dist}-degree square starting from ({ra-query_dist/2}, {dec-query_dist/2}) to ({ra+query_dist/2}, {dec+query_dist/2})")
            # query sky for some amount
            print(f">> Querying the tractor catalog for stars from RA/DEC({ra-query_dist/2}, {dec-query_dist/2}) to ({ra+query_dist/2}, {dec+query_dist/2})...")
            query_df = self.query_tractor(ra, dec, query_dist)
            # make array of ra / dec starting points for degree cubes
            dec_range = np.arange(dec-query_dist/2, dec+query_dist/2)
            ra_range = np.arange(ra-query_dist/2, ra+query_dist/2)
        
        coord_grid = np.meshgrid(ra_range, dec_range)
        ra_coords = coord_grid[0].flatten()
        dec_coords = coord_grid[1].flatten()
        overlap_store = []
        larger_catalogue = pd.DataFrame(columns=['ra','dec'])
        
        print(">> Looping through sky coordinates...")
        for ra_c, dec_c in zip(ra_coords,dec_coords):
            print(f">>> Generating sky catalog for square RA,DEC ({ra_c}, {dec_c}) to ({ra_c+1}, {dec_c+1})...")
            if self.galactic_check(ra_c, dec_c, 1):
                cat, overlap = self.create_degree_square(ra_c, dec_c, query_df, plot_image)
                larger_catalogue = pd.concat([larger_catalogue.astype(cat.dtypes),cat],axis=0).reset_index(drop=True)
                overlap_store.append(overlap)
            else:
                print(f">>> 1-degree square intersects with the galactic plane!")
                overlap_store.append([ra, dec, ra+1, dec+1])
            # print('Added (' + str(ra) + ', ' + str(dec) + ') to catalogue')
        
        print(">> Removing positions from overlapping regions...")
        catalogue = self.remove_overlap_positions(ra_coords, dec_coords, overlap_store, larger_catalogue)
        
        if allsky:
            print(f">> Finding largest overlap for whole {query_dist}-degree square...")
            overlap_store = np.asarray(overlap_store)
            if overlap_store.shape[0] > 1:
                min_ra = np.min(overlap_store[:, 0])
                min_dec = np.min(overlap_store[:, 1])
                max_ra = np.max(overlap_store[:, 2])
                max_dec = np.max(overlap_store[:, 3])
            else: 
                min_ra = overlap_store[0]
                min_dec = overlap_store[1]
                max_ra = overlap_store[2]
                max_dec = overlap_store[3]
            # print(f">> min RA/DEC = ({min_ra}, {min_dec})    max RA/DEC = ({max_ra}, {max_dec})")
            print(f"> Done!")
            return catalogue, [min_ra, min_dec, max_ra, max_dec]
        
        print(f"> Done!")
        return catalogue
    
    # @timer
    def all_sky(self, bands=('g','r','i','z'), query_dist=5.0, min_ra=0, min_dec=-90, max_ra=360, max_dec=30, **kwargs):
        """Loop through the entire sky."""
        
        if 'bands' in kwargs.keys():
            self.bands = kwargs['bands']
        
        print("================= WHOLE SKY =================")
        print(f"===== From {min_ra},{min_dec} to {max_ra},{max_dec} in {query_dist}^2 squares ======")
        print(f"===== Bands used: {self.bands} =====")
        # use 5 degree squares
        
        dec_range = np.arange(min_dec, max_dec, query_dist)
        ra_range = np.arange(min_ra, max_ra, query_dist)
        
        coord_grid = np.meshgrid(ra_range, dec_range)
        ra_coords = coord_grid[0].flatten()
        dec_coords = coord_grid[1].flatten()
        overlap_store = []
        larger_catalogue = pd.DataFrame(columns=['ra','dec'])
        
        print("====== WHOLE SKY: Looping through sky coordinates... =====")
        for ra_c, dec_c in zip(ra_coords, dec_coords):
            print(f"====== {query_dist}-degree square starting from RA,DEC = {ra_c}, {dec_c} ======")
            # if self.galactic_check(ra_c, dec_c, query_dist):
            cat, overlap = self.create_catalogue(ra_c, dec_c, query_dist=query_dist, allsky=True)
            larger_catalogue = pd.concat([larger_catalogue.astype(cat.dtypes),cat],axis=0).reset_index(drop=True)
            overlap_store.append(overlap)
            # else:
            #     print(f"{query_dist}-degree square starting from RA,DEC = {ra_c}, {dec_c} intersects with the galactic plane!")
            
        print("====== WHOLE SKY: Removing positions from overlapping regions... ======")

        catalogue = self.remove_overlap_positions(ra_coords, dec_coords, overlap_store, larger_catalogue, bounds=query_dist)
        print("================= WHOLE SKY: Done! =================")
        print(f"===== WHOLE SKY: Found {catalogue.size} dark sky positions =====")
        return catalogue
        
        
        
if __name__=="__main__":
    
    # catalog = SkyCatalogue()
    catalog_g_band = SkyCatalogue(all_bands=False)
    # cProfile.run('catalog_g_band.create_catalogue(3, -4, 2)', 'gstats')
    catalog_g_band.all_sky(query_dist=2.0, min_ra=212, max_ra=216, min_dec=16, max_dec=20)
    # cProfile.run('catalog_g_band.all_sky(query_dist=2.0, min_ra=212, max_ra=216, min_dec=16, max_dec=20)', 'gstats')
    
    # gstats = pstats.Stats('gstats')
    # gstats.sort_stats(SortKey.TIME).print_stats(20)
    # gstats.print_stats()
    # gstats.strip_dirs().sort_stats(-1).print_stats()
    # positions_gband = catalog_g_band.create_catalogue(3, -4, 2)
    # positions = catalog.all_sky(query_dist=30.0)
    # positions = catalog.create_catalogue()