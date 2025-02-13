# imports
import matplotlib.pyplot as plt
from astropy.io import fits as fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import pkg_resources
pkg_resources.require("numpy==1.26.3")
import numpy as np

from dl import queryClient as qc
import pandas as pd
from sklearn.neighbors import KDTree

class SkyCatalogue():

    def __init__(self, bands=('g','r','i','z'), mag_limit=21, map_dist=1.0, mask_radius=20, fov=45, verbose=False):
        """ Initialise the SkyCatalogue object.

        Parameters
        ----------
        mode : `str` (default="corner")
            If "corner" then defines sky regions using user specified ra, dec as the bottom left corner of each square region
            If "centre" then defines sky regions using user specified ra, dec as the centre of each square region
        bands : `tuple` (default=('g', 'r', 'i', 'z'))
            Selection of passbands applied for object detection
        map_dist : `float` (default=1.0)
            Side length of unit square region for initial dark sky identification handling (degrees)
        mask_radius : `float` (default=20)
            Minimum exlusion zone around object for telescope fov centre (arcseconds)
            Recommended value is half telescope maximum field of view (i.e. 20 arcseconds for ANU 2.3m telescope dimensions)
        fov : `float` (default=45)
            Maximum telescope field of view, corresponding to side length of safe zone around identified dark sky positions (arcseconds)
        """

        print("Initialising...")
        self.verboseprint = print if verbose else lambda *a, **k: None

        self.bands = bands
        if mag_limit >= 16:
            self.mag_limit = mag_limit
        else:
            self.mag_limit = 21
        self.map_dist = map_dist
        # self.dim = int(self.map_dist * (3600/0.262))
        self.dim = int((3600*4) * self.map_dist)
        # degrees per pixel
        self.pixscale = 1 / 14400
        self.mask_radius = mask_radius
        self.fov = fov
        # amount of pixels corresponding to fov at a 0.262 arcsec per 1 pix scale
        # same as used in LSDR10 brick images
        self.pixfov = self.fov // 0.262
        
        # load all masked stars
        self.verboseprint("Loading masked star data....")
        self.load_mask_data()
        
        # define grid lines from fov
        self.verboseprint("Defining grid lines...")
        self.define_grid()
        
        # create distance grid for this dimension
        self.verboseprint("Creating KDTree for distance calculations...")
        self.dist_array = np.indices((self.dim, self.dim), dtype=int)
        self.dist_array = np.dstack((self.dist_array[0], self.dist_array[1]))
        self.dist_array = np.concatenate(self.dist_array, axis=0)
        self.distance_tree = KDTree(self.dist_array)
        self.verboseprint("KDTree created!")
        
        print("Initialisation complete!")
        pass

    def galactic_check(self, ra, dec, dist, mode='corner'):
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
        
        # define square coordinate regions from user specified mode
        if mode=="corner":
            ra_min=ra
            ra_max = ra + dist
            dec_min=dec
            dec_max = dec + dist
        if mode=="centre":
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
        
        # check if in Legacy Survey gap near the galactic plane
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

    
    def query_tractor(self, ra, dec, dist=1.0, mode='corner', **kwargs):
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
            
        optional keyword arguments:
        bands: `tuple` `str`
            Bands to query for objects from the selection of ('g', 'r', 'i', 'z')
            Where objects are detected in multiple bands, the one with the brightest magnitude will be selected
        
        Returns
        -------
        brick_info: `pd.DataFrame`
            Pandas DataFrame containing columns: `ra`, `dec`, `mag`, `passband`
        """
        
        if 'bands' in kwargs.keys():
            self.bands = kwargs['bands']
        
        # Bounds of the square we are querying objects for based on the mode
        if mode=="corner":
            ra_min=ra
            ra_max = ra + dist
            dec_min=dec
            dec_max = dec + dist

        if mode=="centre":
            ra_min=ra-dist/2
            ra_max = ra + dist/2
            dec_min=dec-dist/2
            dec_max = dec + dist/2
      

        # run query based on user input
        mags = [f"mag_{b}" for b in self.bands]
        conditions = [f"({mag}<={self.mag_limit} AND {mag}>=16)" for mag in mags]
        
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
        catalog_box.loc[:, 'radius'] = self.calculate_mask_radius(catalog_box.loc[:,'mag'])
        
        # remove mag column
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
        coords : `list`
            List of bounds of the selected region in order ra_min, ra_max, dec_min, dec_max

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

        # create wcs cornered/centered on that ra/dec
        
        wcs_dict = {
            "CTYPE1": "RA---TAN",
            "CUNIT1": 'deg',
            'CDELT1': self.pixscale,
            'CRPIX1': 0,
            "CRVAL1": coords[0],
            "NAXIS1": self.dim,
            "CTYPE2": "DEC--TAN",
            "CUNIT2": 'deg',
            'CDELT2': self.pixscale,
            'CRPIX2': 0,
            "CRVAL2": coords[2],
            "NAXIS2": self.dim
        }
        
        w = WCS(wcs_dict)
        # pix_arrays = w.wcs_world2pix(all_stars['ra'], all_stars['dec'], 0)
        pixarrays = w.world_to_array_index_values(all_stars['ra'], all_stars['dec'])
        # all_stars['ra_pix'], all_stars['dec_pix'] = w.world_to_array_index_values(np.column_stack([all_stars['ra'], all_stars['dec']]), 0)
        # w.printwcs()
        # print(pixarrays)
        all_stars['ra_pix'], all_stars['dec_pix'] = pixarrays[1], pixarrays[0]
        # ra, dec, and radius in pixels

        all_stars['rad_pix'] = all_stars['radius'] / self.pixscale

        return all_stars
    
    def seg_map(self, star_data:pd.DataFrame):
        """Creates segementation map of shape (`dim`, `dim`) based on the mask locations and pixel data of `star_data`.
        
        Parameters
        ----------
        star_data : `pd.DataFrame`
            Minimum and maxiumum pixel values of all object coordinates along with radius
            Contains columns ra, dec, radius, max_ra, min_ra, max_dec, min_dec, ra_pix, dec_pix, rad_pix

        Returns
        -------
        array : `np.ndarray`
            Binary array of the forbidden and allowed regions (segmentation map)
            Value 1 if forbidden, 0 if allowed
        """

        array = np.zeros(self.dim**2, dtype=int)
        # the index of a particular ra / dec in this array is:
        # array[ dec*dim + ra ]
        # "rows" of dec with "columns" of ra
        radec = np.asarray([star_data['dec_pix'],star_data['ra_pix']]).T
        circle_points = self.distance_tree.query_radius(radec, star_data['rad_pix'])

        for circle_array in circle_points:
            np.put(array, circle_array, 1)
        
        array = array.reshape((self.dim, self.dim))

        return array
    
    def define_grid(self):
        """Creates gridlines and centers on pixels for the initialized dimension and field of view."""

        # self.pixscale = self.fov // 0.272
        self.gridlines = np.arange(0, self.dim+1, self.pixfov)
        centers = np.arange(self.pixfov//2, self.dim - (self.pixfov//2), self.pixfov)
        # centers = []

        # for i in range(len(self.gridlines[:-1])):
        #     centers.append(int((self.gridlines[i] + self.gridlines[i+1])/2 + 0.5))

        self.x_cen, self.y_cen = np.meshgrid(centers, centers)
        return
    
    def find_dark_regions(self, segmap):
        """Iterates through grid squares to collect those which do not intersect with forbidden regions.
        
        Parameters
        ----------
        segmap : `np.ndarray`
            Binary array of forbidden and allowed regions (segmentation map)
            Value 1 if forbidden, 0 if allowed

        Returns
        -------
        dr_trans : `np.ndarray`
            Transverse coordinates of grid square centres designated as allowed dark regions
        dark_regions: `list`
            List of coordinates (ra,dec) that represent centres of grid squares of allowed dark regions
        """

        dark_regions = []

        # define bounds of each grid square and iterate through
        for i in range(len(self.gridlines) - 1):
            for j in range(len(self.gridlines) - 1):
                x_start, x_end = (self.gridlines[i]).astype(int), (self.gridlines[i + 1]).astype(int)
                y_start, y_end = (self.gridlines[j]).astype(int), (self.gridlines[j + 1]).astype(int)
                # check for intersection between grid square and object, accept as dark region if none
                if np.all(segmap[y_start:y_end, x_start:x_end] == 0):
                    dark_regions.append([self.x_cen[j, i], self.y_cen[j, i]])

        # convert to transverse coordinates for easier plot handling
        dr_trans = np.array(dark_regions).transpose()

        return dr_trans, dark_regions

    def create_plot(self, array, coords, pix_coords, dr_trans):
        """Draws a plot of forbidden and allowed grid squares to centre the fov on to obtain a dark position.
        
        Parameters
        ----------
        array : `np.ndarray'
            Binary array of forbidden and allowed regions (segmentation map)
            Value 1 if forbidden, 0 if allowed
        coords : `list`
            List of bounds of the selected region in order ra_min, ra_max, dec_min, dec_max
        pix_coords : `list`
            Pixel values of plot bounds in order ra_min, ra_max, dec_min, dec_max
        dr_trans : `np.ndarray`
            Transverse coordinates of grid square centres designated as allowed dark regions
        """

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

    def create_data_frame(self, dark_regions, coords):
        """Creates a dataframe of allowed dark region within certain coordinate bounds.
        
        Parameters
        ----------
        dark_regions : `list`
            List of coordinates (ra,dec) that represent centres of grid squares of allowed dark regions
        coords : `list`
            List of bounds of the selected region in order ra_min, ra_max, dec_min, dec

        Returns
        -------
        dark_catalogue : `pd.DataFrame`
            Contains coordinates of allowed dark regions within the selected region
            Columns are ra, dec
        """
        dark_ra = []
        dark_dec = []

        # converting pixels back to sky coordinates
        for i in dark_regions:
            ra = i[0] / (self.dim) + coords[0]
            dec = i[1] / (self.dim) + coords[2]
            dark_ra.append(ra)
            dark_dec.append(dec)

        dark_catalogue = pd.DataFrame({'ra':dark_ra, 'dec':dark_dec})
        return dark_catalogue
    
    def find_overlapping_extent(self, all_stars):
        """Identifies the maximum extent to which the masked exclusion zone of an object within
        one section of the sky may overlap into another section of the sky.

        Parameters
        ----------
        all_stars : `pd.DataFrame`
            Columns min_ra, max_ra, min_dec, max_dec which give the extent of the masked exclusion zone of each object

        Returns
        -------
        'list'
            Minimum and maximum right ascension and declinations of the maximum overlap of one sky segment into another
        """

        # identify the maximum mask extent which overlaps with neighbouring regions on each side of square 
        min_ra = all_stars['min_ra'].min()
        min_dec = all_stars['min_dec'].min()
        max_ra = all_stars['max_ra'].max()
        max_dec = all_stars['max_dec'].max()
        
        return [min_ra, min_dec, max_ra, max_dec]

    def create_degree_square(self, ra, dec, catalog_df=None, plot_image=False, add_query=False, mode='corner', **kwargs):
        """Generates dark sky positions for a 1 x 1 degree region of the sky with user specified
        mode "corner" or "centre" determining the position of given ra, dec coordinate within square region.

        Parameters
        ----------
        ra : `float`
            If mode "corner" then right ascension coordinate of the bottom left corner of square
            If mode "centre" then right ascension coordinate of centre of square region
        dec : `float`
            If mode "corner" then declination coordinate of bottom left corner of square
            If mode "centre" then declination coordinate of centre of square region
        catalog_df : `pd.DataFrame`
            Tractor catalogue of objects and magnitudes within specified 1 x 1 degree region
            Columns are ra, dec, mag
        plot_image : `bool` (default=False)
            If True then plots the square region and the allowed dark regions
        
        optional keyword arguments:
        bands: `tuple` `str`
            Bands to query for objects from the selection of ('g', 'r', 'i', 'z')
            Where objects are detected in multiple bands, the one with the brightest magnitude will be selected

        Returns
        -------
        dark_catalogue : `pd.DataFrame`
            Contains catalogue of allowed dark sky regions within the selected 1 x 1 degree region
            Columns are ra, dec

        overlap : `list`
            Minimum and maximum right ascension and declinations of the maximum overlap of one 1 x 1 degree region into another
            In order; min_ra, min_dec, max_ra, max_dec
        """
            
        if 'bands' in kwargs.keys():
            self.bands = kwargs['bands']
        
        if mode=="centre":
            coords=[ra-self.map_dist/2, ra+self.map_dist/2, dec-self.map_dist/2, dec+self.map_dist/2]
        elif mode=="corner":
            coords = [ra, ra+self.map_dist, dec, dec+self.map_dist]
            
        if add_query:
            self.verboseprint(f">> Querying the tractor catalog for stars from RA/DEC({coords[0]}, {coords[2]}) to ({coords[1]}, {coords[3]})...")
            catalog_df = self.query_tractor(ra, dec, dist=self.map_dist, mode=mode)
        # print(">>>> Generating dark sky positions of 1-degree square...")
        self.verboseprint(">>>> Combining mask and queried stars...")
        all_stars = self.combine_data(catalog_df, coords)
        self.verboseprint(">>>> Calculating pixel values for stars....")
        all_stars = self.create_pixel_columns(all_stars, coords)

        self.verboseprint(">>>> Creating segmentation map...")
        segmentation_map = self.seg_map(all_stars)
        
        self.verboseprint(">>>> Finding dark regions in segmentation map...")
        dr_trans, dark_regions = self.find_dark_regions(segmentation_map)

        if plot_image:
            self.verboseprint(">>>> Plotting dark regions...")
            pix_coords = [all_stars['ra_pix'], all_stars['dec_pix'], all_stars['rad_pix']]
            self.create_plot(segmentation_map, coords, pix_coords, dr_trans)

        self.verboseprint(">>>> Converting dark regions to coordinates...")
        dark_catalogue = self.create_data_frame(dark_regions, coords)
        
        self.verboseprint(">>>> Finding maximum extent of stars beyond the degree-square bounds...")
        overlap = self.find_overlapping_extent(all_stars)
        
        self.verboseprint(">>>> Done!")
        return dark_catalogue, overlap

    def remove_overlap_positions(self, ra_coords, dec_coords, overlap_store, dark_catalogue, bounds=1, mode='corner'):
        """Deletes dark sky positions on the edges of regions that fall into the masks of objects in neighbouring regions.

        Parameters
        ----------
        ra_coords : `list`
            Contains embedded list of right ascensions of all allowed dark sky positions for each coordinate within user specified query region
        dec_coords : `list`
            Contains embedded list of declinations of all allowed dark sky positions for each coordinate within user specified query region
        overlap_store : `list`
            Contains embedded list of maximum overlaps in ra, dec of each coordinate within user specified query region
        dark_catalogue : `pd.DataFrame`
            Contains catalogue of allowed dark sky regions within user specified query region
            Columns are ra, dec

        Returns
        -------
        dark_catalogue : `pd_DataFrame`
            Overwritten catalogue of dark sky positions, with positions on the edges of regions that fall into the masks of objects in neighbouring regions removed
            Columns are ra, dec
        """

        # for each ra / dec square and associated overlap
        for ra, dec, overlap in zip(ra_coords, dec_coords, overlap_store):
            
            # make sure this works if ra/dec extents are less than the actual bounds!
            if mode=="corner":
                min_ra = overlap[0] if overlap[0] < ra else ra
                min_dec = overlap[1] if overlap[1] < dec else dec
                max_ra = overlap[2] if overlap[2] > ra+bounds else ra+bounds
                max_dec = overlap[3] if overlap[3] > dec+bounds else dec+bounds
            elif mode=="centre":
                min_ra = overlap[0] if overlap[0] < ra-bounds/2 else ra-bounds/2
                min_dec = overlap[1] if overlap[1] < dec-bounds/2 else dec-bounds/2
                max_ra = overlap[2] if overlap[2] > ra+bounds/2 else ra+bounds/2
                max_dec = overlap[3] if overlap[3] > dec+bounds/2 else dec+bounds/2
            
            # everything within square bounded by ra / dec and bounds that you're checking
            smaller_box = dark_catalogue.query(f'({ra} < ra < {ra + bounds}) & ({dec} < dec < {dec + bounds})') 
            
            # select everything within square bounded by the min/max ra/dec of the overlap store
            bigger_box = dark_catalogue.query(f'({min_ra} <= ra <= {max_ra}) & ({min_dec} <= dec <= {max_dec})')
            
            # only perform removal if there are actually sky positions in the "overlap region"
            if smaller_box.shape < bigger_box.shape:
                # everything that is in the bigger box but not the smaller one (within overlapping region)
                overlap_region = pd.concat((bigger_box, smaller_box)).drop_duplicates(keep=False)
                # concatenate the two and drop everything within the overlapping regions
                dark_catalogue = pd.concat((dark_catalogue, overlap_region)).drop_duplicates(keep=False)
                        
        return dark_catalogue
        
    def create_catalogue(self, ra, dec, query_dist, plot_image=False, return_overlaps=False, mode='corner', **kwargs):
        """Creates catalog of dark sky positions in a square defined by ra, dec, and query_dist.
        
        Parameters
        ----------
        ra : `float`
            If mode "corner" then right ascension of bottom left corner of square region (degrees)
            If mode "centre" then right ascension of centre coordinate of square region (degrees)
        dec : `float`
            If mode "corner" then declination of bottom left corner of square region (degrees)
            If mode "centre" then declination of centre coordinate of square region (degrees)
        query_dist : `float`
            Side length of square to query (degrees)
        plot_image : `bool` (default=False)
            If True, plot the image of the region
        return_overlaps : `bool` (default=False)
            If True, then return list of mask overlaps between regions
            Useful if finding dark sky positions in a larger sky region (5 x 5 degree and greater)
        
        optional keyword arguments:
        bands: `tuple` `str`
            Bands to query for objects from the selection of ('g', 'r', 'i', 'z')
            Where objects are detected in multiple bands, the one with the brightest magnitude will be selected
        
        Returns
        -------
        dark_catalogue : `pd.DataFrame`
            DataFrame of all dark sky positions withing user specified region containing columns ra, dec
        """
        
        if 'bands' in kwargs:
            self.bands = kwargs['bands']
            
        if mode == 'centre':
            self.verboseprint(f"> Creating sky catalog from one {query_dist}-degree square starting from ({ra-query_dist/2}, {dec-query_dist/2}) to ({ra+query_dist/2}, {dec+query_dist/2})")
            # query sky for some amount
            self.verboseprint(f">> Querying the tractor catalog for stars from RA/DEC({ra-query_dist/2}, {dec-query_dist/2}) to ({ra+query_dist/2}, {dec+query_dist/2})...")
            query_df = self.query_tractor(ra, dec, query_dist, mode='centre')
            # make array of ra / dec starting points for degree cubes
            dec_range = np.arange(dec-query_dist/2, dec+query_dist/2, self.map_dist)
            ra_range = np.arange(ra-query_dist/2, ra+query_dist/2, self.map_dist)
        elif mode == 'corner': 
            print(f"> Creating sky catalog from one {query_dist}-degree square starting from ({ra}, {dec}) to ({ra+query_dist}, {dec+query_dist})")
            # query sky for some amount
            self.verboseprint(f">> Querying the tractor catalog for stars from RA/DEC({ra}, {dec}) to ({ra+query_dist}, {dec+query_dist})...")
            query_df = self.query_tractor(ra, dec, query_dist)
            # make array of ra / dec starting points for degree cubes
            dec_range = np.arange(dec, dec+query_dist, self.map_dist)
            ra_range = np.arange(ra, ra+query_dist, self.map_dist)
        
        coord_grid = np.meshgrid(ra_range, dec_range)
        ra_coords = coord_grid[0].flatten()
        dec_coords = coord_grid[1].flatten()
        overlap_store = []
        dark__catalogue = pd.DataFrame(columns=['ra','dec'])
        
        self.verboseprint(">> Looping through sky coordinates...")
        for ra_c, dec_c in zip(ra_coords,dec_coords):
            self.verboseprint(f">>> Generating sky catalog for square RA,DEC ({ra_c}, {dec_c}) to ({ra_c+self.map_dist}, {dec_c+self.map_dist})...")
            if self.galactic_check(ra_c, dec_c, self.map_dist, mode=mode):
                cat, overlap = self.create_degree_square(ra_c, dec_c, query_df, plot_image)
                dark__catalogue = pd.concat([dark__catalogue.astype(cat.dtypes),cat],axis=0).reset_index(drop=True)
                overlap_store.append(overlap)
            else:
                self.verboseprint(f">>> {self.map_dist}-degree square intersects with the galactic plane!")
                overlap_store.append([ra, dec, ra+self.map_dist, dec+self.map_dist])
            # print('Added (' + str(ra) + ', ' + str(dec) + ') to catalogue')
        
        self.verboseprint(">> Removing positions from overlapping regions...")
        dark_catalogue = self.remove_overlap_positions(ra_coords, dec_coords, overlap_store, dark__catalogue, mode=mode)
        
        if return_overlaps:
            self.verboseprint(f">> Finding largest overlap for whole {query_dist}-degree square...")
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
            self.verboseprint(f"> Done!")
            return dark_catalogue, [min_ra, min_dec, max_ra, max_dec]
        
        self.verboseprint(f"> Done!")
        return dark_catalogue

    def all_sky(self, ra_allsky=0, dec_allsky=-90, sky_dist=10.0, query_dist=2.0, full_sky=False, mode='corner', **kwargs):
        """Loop through the entire sky.
        
        Parameters
        ----------
        ra_allsky
            starting ra (default=0)
        dec_allsky
            starting dec (default=-90)
        sky_dist
            size of ra/dec square to create a catalog over (default=10)
        query_dist
            size of smaller query-sized subdivisions (default=2)
        full_sky
            If true, generate positions over the range of the entire sky from (0, -90) to (360, 30)
            
        optional keyword argument:
        bands: `tuple` `str`
            Bands to query for objects from the selection of ('g', 'r', 'i', 'z')
            Where objects are detected in multiple bands, the one with the brightest magnitude will be selected
        mode: `str`
            (default) If "corner" then defines sky regions using user specified ra, dec as the bottom left corner of each square region
            If "centre" then defines sky regions using user specified ra, dec as the centre of each square region
        """
        
        print("================= WHOLE SKY =================")
        
        if 'bands' in kwargs:
            self.bands = kwargs['bands']
            
        # make sure sky distance is larger than query distance for consistency
        if query_dist > sky_dist:
            print(f"Your query distance ({query_dist}) is larger than the sky distance ({sky_dist}) you're trying to cover!")
            print(f"Either reduce your query distance, or use the SkyCatalogue.create_catalogue() method instead.")
            return
        
        # use corner mode by default
        if mode == 'centre':
            # TODO: check bounds of ra/dec range to be within 0-360, -90-30 
            dec_range = np.arange(dec_allsky-sky_dist/2, dec_allsky+sky_dist/2, query_dist)
            ra_range = np.arange(ra_allsky-sky_dist/2, ra_allsky+sky_dist/2, query_dist)
            print(f"===== From ({ra_allsky-sky_dist/2}, {dec_allsky-sky_dist/2}) to ({ra_allsky+sky_dist/2}, {dec_allsky+sky_dist/2}) in {len(dec_range)} {query_dist}^2 squares ======")
        elif mode == 'corner':
            dec_range = np.arange(dec_allsky, dec_allsky+sky_dist, query_dist)
            ra_range = np.arange(ra_allsky, ra_allsky+sky_dist, query_dist)
            print(f"===== From ({ra_allsky}, {dec_allsky}) to ({ra_allsky+sky_dist}, {dec_allsky+sky_dist}) in {len(dec_range)} {query_dist}^2 squares ======")
    
            
        if full_sky:
            dec_range = np.arange(-90, 30, query_dist)
            ra_range = np.arange(0, 360, query_dist)
            print(f"===== Whole sky from (0, -90) to (360, 30) =====")
        
        print(f"===== Bands used: {self.bands} =====")
        # use 5 degree squares
        
        # dec_range = np.arange(min_dec, max_dec, query_dist)
        # ra_range = np.arange(min_ra, max_ra, query_dist)
        
        coord_grid = np.meshgrid(ra_range, dec_range)
        ra_coords = coord_grid[0].flatten()
        dec_coords = coord_grid[1].flatten()
        overlap_store = []
        larger_catalogue = pd.DataFrame(columns=['ra','dec'])
        
        print("====== WHOLE SKY: Looping through sky coordinates... =====")
        for ra_c, dec_c in zip(ra_coords, dec_coords):
            print(f"====== {query_dist}-degree square starting from RA,DEC = {ra_c}, {dec_c} ======")
            # if self.galactic_check(ra_c, dec_c, query_dist):
            cat, overlap = self.create_catalogue(ra_c, dec_c, query_dist=query_dist, return_overlaps=True)
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

    catalog_g_band = SkyCatalogue(all_bands=False)
    catalog_g_band.all_sky(query_dist=2.0, min_ra=212, max_ra=216, min_dec=16, max_dec=20)