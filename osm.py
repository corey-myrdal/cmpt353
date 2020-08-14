import pandas as pd
import numpy as np
from scipy import stats
import os, sys 
from wikidata.client import Client
import wikipediaapi as wk
from wikitables import import_tables
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import colors
import gmplot
    
def main(in_directory):
    # google maps api key
    gmaps_api = 'AIzaSyAm4vrzQVlfDxV2L13PuWgHEiwVaS8IPHc'
    
    # load osm data
    osmpath = os.path.join(in_directory, 'amenities-vancouver.json')
    osm_data = pd.read_json(osmpath, lines=True)
    
    # load wikidata
    #wikiclient = Client()
    #wikient = wikiclient.get('Q3472954', load=True)
    
    # load wikipedia data
    wiki = wk.Wikipedia('en')
    can_chains = wiki.page('List of Canadian restaurant chains')
    can_chains = pd.DataFrame({'restaurant_name': can_chains.sections[0].sections})
    can_chains = can_chains.applymap(lambda x: x.title)
    
    wiki_table = import_tables('List of restaurant chains in the United States')
    us_chains = pd.DataFrame(wiki_table[0].rows)

    # necessary loop due to wikitable library
    for table in wiki_table[1:29]:
        us_chains = us_chains.append(pd.DataFrame(table.rows), sort=False)
    
    us_chains = us_chains.reset_index()
    
    # gather restaurant and fast food osm data
    osm_rest = osm_data[osm_data['amenity'].isin(['restaurant','fast_food'])]
    
    # drop any restaurants without names
    osm_rest = osm_rest.dropna(subset=['name'])
    
    # detect chains utilizing multiple methods
    # first method, find restaurants that occur multiple times in osm data
    osm_chains = osm_rest[osm_rest.duplicated(['name'])]
    osm_indep = osm_rest[~osm_rest.duplicated(['name'])]
    
    # second method, include restaurants in list of US and CAN chains
    osm_chains_can = osm_rest[osm_rest['name'].isin(can_chains['restaurant_name'])]
    osm_chains_us = osm_rest[osm_rest['name'].isin(us_chains['Name'])]
    
    # combine lists
    osm_chains_can = pd.concat([osm_chains_can,osm_chains_us], sort='True')#.drop_duplicates().reset_index(drop=True)
    osm_chains_can = osm_chains_can.drop_duplicates(subset=['lat','lon']).reset_index()
    
    # combine methods
    osm_chains = pd.concat([osm_chains,osm_chains_can], sort='True')#.drop_duplicates().reset_index(drop=True)
    osm_chains = osm_chains.drop_duplicates(subset=['lat','lon']).reset_index()
    
    
    # segment osm data into grid
    # find bounds of grid
    ymin = osm_rest['lat'].min()
    ymax = osm_rest['lat'].max()
    xmin = osm_rest['lon'].min()
    xmax = osm_rest['lon'].max()
    
    # build grid
    grid_density = 120
    x_grid = np.linspace(xmin, xmax, grid_density*1.3)
    y_grid = np.linspace(ymin, ymax, grid_density)
    
    dens_rest, x_edges, y_edges = np.histogram2d(osm_rest['lon'].values, osm_rest['lat'].values, bins=[x_grid, y_grid])
    dens_chain, _, _ = np.histogram2d(osm_chains['lon'].values, osm_chains['lat'].values, bins=[x_grid, y_grid])
    dens_indep, _, _ = np.histogram2d(osm_indep['lon'].values, osm_indep['lat'].values, bins=[x_grid, y_grid])
    
    dens_rest = dens_rest.T
    dens_chain = dens_chain.T
    dens_indep = dens_indep.T

    # density of chains over independents
    dens_chain_vs_indep = dens_chain - dens_indep

    dens_chain_vs_indep[dens_chain_vs_indep == 0] = np.nan
    
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
    
    # plot heatmap
    fig, ax = plt.subplots()
    ax.set_axis_off()
    
    # setup separate colors for locations with more chains vs independents
    min_dens = np.nanmin(dens_chain_vs_indep)
    max_dens = np.nanmax(dens_chain_vs_indep)
    color_chain = plt.cm.coolwarm(np.linspace(0.15, 0, max_dens))
    color_indep = plt.cm.coolwarm(np.linspace(0.5, 0.95, abs(min_dens)))
    color_chain_indep = np.vstack((color_indep, color_chain))
    color_chain_indep = colors.LinearSegmentedColormap.from_list('color_chain_indep', color_chain_indep)
    
    # seperate colormap for chains and independent restaurants
    divnorm = colors.TwoSlopeNorm(vmin=min_dens, vcenter=0, vmax=max_dens)
    
    # plot heatmap
    ax.pcolormesh(x_mesh, y_mesh, dens_chain_vs_indep, cmap=color_chain_indep, norm=divnorm)
    
    # save heatmap
    img_bound = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('chainvsindep_heatmap.png', format='png', transparent=True, bbox_inches=img_bound, pad_inches=0)
    
    # overlap heatmap on google map
    lat_mid = np.mean(osm_rest['lat'])
    lon_mid = np.mean(osm_rest['lon'])
    
    gmap = gmplot.GoogleMapPlotter(lat_mid, lon_mid, 11, apikey=gmaps_api)
    
    # adjust for grid spacing
    grid_size = grid_density/(grid_density-1)
    north = (ymax-lat_mid)*grid_size + lat_mid
    south = (ymin-lat_mid)*grid_size + lat_mid
    east = (xmax-lon_mid)*grid_size + lon_mid
    west = (xmin-lon_mid)*grid_size + lon_mid
    
    bounds = {'north':north, 'south':south, 'east':east, 'west':west}
    gmap.ground_overlay('chainvsindep_heatmap.png', bounds, opacity=0.8)

    gmap.draw('map_chainvsindep.html')
    
    # heatmap for chains and independts separately on two maps
    gmap = gmplot.GoogleMapPlotter(lat_mid, lon_mid, 11, apikey=gmaps_api)
    
    gmap.heatmap(osm_chains['lat'].values,osm_chains['lon'].values)
    gmap.draw('map_chain.html')
    
    gmap.heatmap(osm_indep['lat'].values,osm_indep['lon'].values)
    gmap.draw('map_indep.html')
    
    ############################################################
    #### heatmap zoom in ###
    grid_density = 300
    x_grid = np.linspace(xmin, xmax, grid_density*1.5)
    y_grid = np.linspace(ymin, ymax, grid_density)
    
    dens_rest, x_edges, y_edges = np.histogram2d(osm_rest['lon'].values, osm_rest['lat'].values, bins=[x_grid, y_grid])
    dens_chain, _, _ = np.histogram2d(osm_chains['lon'].values, osm_chains['lat'].values, bins=[x_grid, y_grid])
    dens_indep, _, _ = np.histogram2d(osm_indep['lon'].values, osm_indep['lat'].values, bins=[x_grid, y_grid])
    
    dens_rest = dens_rest.T
    dens_chain = dens_chain.T
    dens_indep = dens_indep.T

    # density of chains over independents
    dens_chain_vs_indep = dens_chain - dens_indep
    
    dens_chain_vs_indep[dens_chain_vs_indep == 0] = np.nan
    
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
    
    # plot heatmap
    fig, ax = plt.subplots()
    ax.set_axis_off()
    
    # setup separate colors for locations with more chains vs independents
    min_dens = np.nanmin(dens_chain_vs_indep)
    max_dens = np.nanmax(dens_chain_vs_indep)
    color_chain = plt.cm.coolwarm(np.linspace(0.15, 0, max_dens))
    color_indep = plt.cm.coolwarm(np.linspace(0.5, 0.95, abs(min_dens)))
    color_chain_indep = np.vstack((color_indep, color_chain))
    color_chain_indep = colors.LinearSegmentedColormap.from_list('color_chain_indep', color_chain_indep)
    
    # seperate colormap for chains and independent restaurants
    divnorm = colors.TwoSlopeNorm(vmin=min_dens, vcenter=0, vmax=max_dens)
    
    # plot heatmap
    ax.pcolormesh(x_mesh, y_mesh, dens_chain_vs_indep, cmap=color_chain_indep, norm=divnorm)
    
    # save heatmap
    img_bound = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('chainvsindep_heatmap_zoom.png', format='png', dpi=1000, transparent=True, bbox_inches=img_bound, pad_inches=0)
    
    # overlap heatmap on google map
    lat_mid = np.mean(osm_rest['lat'])
    lon_mid = np.mean(osm_rest['lon'])
    
    gmap = gmplot.GoogleMapPlotter(lat_mid, lon_mid, 11, apikey=gmaps_api)
    
    # adjust for grid spacing
    grid_size = grid_density/(grid_density-1)
    north = (ymax-lat_mid)*grid_size + lat_mid
    south = (ymin-lat_mid)*grid_size + lat_mid
    east = (xmax-lon_mid)*grid_size + lon_mid
    west = (xmin-lon_mid)*grid_size + lon_mid
    
    bounds = {'north':north, 'south':south, 'east':east, 'west':west}
    gmap.ground_overlay('chainvsindep_heatmap_zoom.png', bounds, opacity=0.8)

    gmap.draw('map_chainvsindep_zoom.html')
    
    
    

if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
    
    
##### Other attempts #####
    # subtract to show locations with more chains than independent restaurants
    '''dens_chain_vs_indep = dens_chain - dens_indep
    dens_chain_vs_indep[dens_chain_vs_indep<1] = np.nan
    
    # create heatmap
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.tick_params(which='both', direction='in')
    ax.set_axis_off()
    
    ax.imshow(np.rot90(dens_chain_vs_indep), extent=[xmin, xmax, ymin, ymax], alpha=0.5, cmap='coolwarm')
    
    # remove padding on image and save
    img_bound = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    fig.savefig('chainvsindep_heatmap.png', format='png', transparent=True, bbox_inches=img_bound, pad_inches=0)
    
    # need to adjust image dimensions since xmin, ymin etc are locations of the
    # grid squares not the actual corner of the images
    lat_mid = ((ymax-ymin)/2)+ymin
    lon_mid = ((xmax-xmin)/2)+xmin
    print(lat_mid)
    
    grid_size = grid_density/(grid_density-1)
    north = (ymax-lat_mid)*grid_size + lat_mid
    south = (ymin-lat_mid)*grid_size + lat_mid
    east = (xmax-lon_mid)*grid_size + lon_mid
    west = (xmin-lon_mid)*grid_size + lon_mid
    bounds = {'north':north, 'south':south, 'east':east, 'west':west}
    
    
    
    gmap = gmplot.GoogleMapPlotter(lat_mid, lon_mid, 11, apikey=gmaps_api)
    
    gmap.ground_overlay('chainvsindep_heatmap.png', bounds)
    gmap.draw('map_chainvsindep.html')'''
    
    
    #print(wikient.description)
    #print(pd.unique(osm_data['amenity']))
    #print(can_chains)
    #print(us_chains)
    #print(osm_chains)
    
    # create map for visualization
    '''gmap = gmplot.GoogleMapPlotter(49.154904, -122.779240, 11, apikey=gmaps_api)
    
    gmap.heatmap(osm_chains['lat'].values,osm_chains['lon'].values)
    gmap.draw('map_chain.html')
    
    gmap.heatmap(osm_indep['lat'].values,osm_indep['lon'].values)
    gmap.draw('map_indep.html')'''
    
    
    
    #plt.plot(osm_chains['lat'], osm_chains['lon'], 'o')
    #plt.show()
    '''fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.tick_params(which='both', direction='in')
    ax.set_axis_off()
    
    ax.imshow(np.rot90(dens_chain_vs_indep), extent=[xmin, xmax, ymin, ymax], alpha=0.5, cmap='coolwarm')
    
    # remove padding on image and save
    img_bound = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    fig.savefig('chainvsindep_heatmap.png', format='png', transparent=True, bbox_inches=img_bound, pad_inches=0)
    
    # need to adjust image dimensions since xmin, ymin etc are locations of the
    # grid squares not the actual corner of the images
    lat_mid = ((ymax-ymin)/2)+ymin
    lon_mid = ((xmax-xmin)/2)+xmin
    print(lat_mid)
    
    grid_size = grid_density/(grid_density-1)
    north = (ymax-lat_mid)*grid_size + lat_mid
    south = (ymin-lat_mid)*grid_size + lat_mid
    east = (xmax-lon_mid)*grid_size + lon_mid
    west = (xmin-lon_mid)*grid_size + lon_mid
    bounds = {'north':north, 'south':south, 'east':east, 'west':west}
    
    
    
    gmap = gmplot.GoogleMapPlotter(lat_mid, lon_mid, 11, apikey=gmaps_api)
    
    gmap.ground_overlay('chainvsindep_heatmap.png', bounds)
    gmap.draw('map_chainvsindep.html')'''