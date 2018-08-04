README for cleaned-ish thesis file directory.

Short demo is Demo_galaxybin.ipynb best viewed in notebook viewer so you can interact with plots using the javascript plugin.

The main work-horse files for my second paper are in notebooks, New_0910_cats_NUVRRJ_mod.ipynb and Rotate_PreGordo_cat_all_rev4_mod.ipynb, which document the steps taken, all in Pandas, to prepare the catalogs and examine and analyze the data, respectively. The other two notebooks are exercises in calculating tidal force for merging clusters and making the Voronoi plot for the second paper. The two .py files are used to make composite spectra plots for the paper (ApJ_compositeplot_pregordo_rev2_mod.py) and consolidate all the functions I wrote for data analysis which are called in the notebooks (Cluster_functs_mod.py). Once I have determine which galaxy spectra to co-add I export the catalogs to IDL programs to make the composites and do the fitting and measurements. These IDL codes, along with the Voronoi tesselation codes, are not public. I would love to convert everything from IDL to Python for the greater good if someone wanted to fund me to do it :)

It is important to note that the notebooks are in static html only as the interactive plots made with mplD3 do not display on github. The interactive plots using mplD3, in combination with the jupyter notebooks, have proved very useful for vizualizing data and presenting results during research meetings. There is a basic example I uploaded a long time ago to git_random, or I can provide the interactive version of the plots in the notebook if you want, or can re-run and upload the enabled notebooks so you can view them with notebook viewer.

These notebooks cover only the analysis of my second paper, however I started assempling Cluster_functs while doing the analysis for my first paper, once I had decided to convert as many codes as possible to be compatible with Pandas Dataframes. If you want to see the exciting results of both papers (and/or my dissertation) you can check out the arXiv versions for free:

This is a link to the papers on NASA ADS:

Mansheim et. al 2017 (MNRAS letter on the Pre-Gordo cluster merger)
http://adsabs.harvard.edu/abs/2017MNRAS.469L..20M

Mansheim et. al 2017 (ApJ paper on the Musket Ball cluster merger)
http://adsabs.harvard.edu/abs/2017ApJ...834..205M

Dissertation
http://adsabs.harvard.edu/abs/2016PhDT.......275M
