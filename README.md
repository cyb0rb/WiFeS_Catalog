# DarkSkyPALS
## A Python software to generate dark sky positions from the DESI Legacy Survey, customised for the ANU 2.3m telescope

**WARNING: This software is broken!**

Reliability of produced sky positions decreases significantly as declination approaches the poles. This is due to a bug which incorrectly converts between spherical cartesian coordinates (ra, dec) and their linear projections onto square regions used to locate dark sky positions. The user will need to investigate the implementation of `wcs.world_to_array_index_values` and its inverse function rectify this issue.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ANU SSO Summer Intership Project 2025
Alannah Falvo, Cyrus Worley, Vernica Mehta
