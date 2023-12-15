## Importing Libraries

### flask
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from markupsafe import escape
import base64

### URASpace
from seleniumwire import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import gzip
import brotli

import re
import json
import numpy as np
import pandas as pd
import random

### OSM
import osmnx as ox
from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiLineString, MultiPolygon , shape
from shapely.ops import linemerge, split, snap, nearest_points

import pyproj
import networkx as nx
import math

import requests
import credentials


### RL import
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from shapely.geometry import Polygon, Point, LineString
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from RL.utils import polygon_to_Poly3DCollection
import rasterio.features
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from shapely import affinity
from shapely.affinity import scale, translate

#p2p import
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf


#use mapbox api to get the static image
access_token = credentials.mapbox_token

### Selenium to run through the website and extract PR and Parcel Polygon
def URAMapSelenium(searchkey):
    """
    input searchkey: string (e.g. "clavon", "640761", "732786")
    output: dictionary
    {
        "search_result": searchkey,
        "name": name,
        "address": address,
        "postal_code": postal_code,
        "PR": plot_ratio,
        "geometry": geometry,
        'URA_GFA': gross_floor_area,
        'URA_site_area': site_area,
        'URA_building_height': building_height,
        'URA_dwelling_units': no_of_dwelling_units
    }
    """
    output = {}
    # Initialize the web driver (specify the path to your web driver)
    # driver = webdriver.Chrome()
    options = webdriver.ChromeOptions()
    options.add_argument("headless")  # run headless chrome
    driver = webdriver.Chrome(options=options)

    # Open the website
    emit_log("Intializing scraping bots...")
    driver.get("https://www.ura.gov.sg/maps/")
    class_name = "us-wel-closesplash-icon"  # Hide the left menu from the start
    element = driver.find_element(By.CLASS_NAME, class_name)
    element.click()

    # Locate the search bar element by its HTML attributes
    search_bar = driver.find_element(By.ID, "us-s-txt")

    # Input the number and perform the search
    search_bar.send_keys(searchkey)

    driver.implicitly_wait(10)
    search_result = driver.find_element(By.CLASS_NAME, "list-group-item.us-sr-item")
    search_result.send_keys(Keys.RETURN)

    driver.implicitly_wait(10)

    # click on the  button us-ip-svcs-l2-other-services
    Other_element = driver.find_element(By.CLASS_NAME, "us-ip-svcs-l2-other-services")
    Other_element.click()
    # Wait for the results to load (you may need to specify a longer wait time)
    driver.implicitly_wait(5)

    captured_requests = driver.requests


    # Extract parcel polygon -------------------------------------------------------------------------------------------------
    output["search_result"] = searchkey

    for request in captured_requests:
        #check that the url matches https://www.onemap.gov.sg/api/....
        url = request.url
        #Extract from response body (spefically for landlot api from URA) which returns a gzip format of the geometry
        # if (re.match(r"https://maps.ura.gov.sg/arcgis/rest/services/maps2/landlot/*",url) != None):
        if (re.match(r"https://www\.onemap\.gov\.sg/api/public/revgeocode\?location=",url) != None):
            print("Name URL:", request.url)
            compressed_data = request.response.body
            try: #json
                # print("Response Content (bytes):", request.response.body)
                data = json.loads(compressed_data)
            except Exception as e:
                print('Invalid JSON format - Error:', e)
                try: #gzip
                    decompressed_data = gzip.decompress(request.response.body)
                    # print("Decompressed Content:", decompressed_data)
                    converted_data = decompressed_data.decode("utf-8")
                    # print("Converted Content:", converted_data)
                    data = json.loads(converted_data)
                except Exception as e:
                    print('Invalid brotli format - Error:', e)

            try:
                print(data)
                output['name'] = data['GeocodeInfo'][0]['BUILDINGNAME']
                output['postal_code'] = data['GeocodeInfo'][0]['POSTALCODE']
                output['address'] = data['GeocodeInfo'][0]['BLOCK'] + " "+ data['GeocodeInfo'][0]['ROAD']
                print("Collected location info from OneMap")
            except Exception as e:
                print("Location Info Error:", e)

            pass
            print("Collected Name, Address and Postal Code from OneMap")
        if (re.match(r"https://maps\.ura\.gov\.sg/arcgis/rest/services/MP19/Updated_Landuse_gaz/MapServer/45/query\?returnGeometry=true*",url) != None):
            print("Geometry URL:", request.url)
            # print("Method:", request.method)
            # print("Request Headers:", request.headers)
            # print("Response Content (bytes):", request.response.body)
            compressed_data = request.response.body
            decompressed_data = gzip.decompress(compressed_data)
            # print("Decompressed Content:", decompressed_data)
            converted_data = decompressed_data.decode("utf-8")
            # print("Converted Content:", converted_data)

            #convert str in to dict
            data = json.loads(converted_data)
            # print(type(data))
            count = 0
            geometry = None
            features = data['features']
            # print(features[0])
            output["PR"] = features[0]['attributes']["GPR_NUM"]

            for feature in features:
                feature_ring = feature['geometry']['rings']
                if len(feature_ring) > count:
                    # print("feature_ring:",feature_ring)
                    geometry = feature_ring
                    count = len(feature_ring)
                # else:
                #     geometry = feature_ring
            # print(data['features'][0]['geometry']['rings'])
            # geometry = data['features'][1]['geometry']['rings']
            print(geometry)
            print("Collected Polygon from URA Space")
            output['geometry'] = geometry
        if (re.match(r"https://www\.ura\.gov\.sg/eDevtRegister/map/service/getSiteInfo\.do\?polygon_id",url) != None): #only works for MUKIM number
            print("Dev Info:", request.url)
            # print("Response Content (bytes):", request.response.body)
            compressed_data = request.response.body
            decompressed_data = brotli.decompress(request.response.body)
            # print("Decompressed Content:", decompressed_data)
            converted_data = decompressed_data.decode("utf-8")
            # print("Converted Content:", converted_data)
            # Extract the JSON part using regular expressions
            match = re.search(r'\((.*?)\);$', converted_data)
            if match:
                json_string = match.group(1)
                # Parse the JSON string
                data = json.loads(json_string)
                # print(data)
                try: 
                    print(data)
                    output['URA_GFA'] = data['result'][0]['gfa']
                    output['URA_site_area'] = data['result'][0]['site_area']
                    output['URA_building_height'] = data['result'][0]['bldg_height']
                    output['URA_dwelling_units'] = data['result'][0]['resi_units']
                    print("Collected development information from URA Space")
                except Exception as e:
                    print("Dev Info Error:", e)
            else:
                print("Invalid JSONP format")

    driver.quit()
    emit_log("URA scrapping process completed...")
    return output


def calculate_distance(line, feature):
    emit_log("Calculating distances...")
    return line.distance(feature)


def compare_points(line,centroid):  # identify which point is to the left and which is to the right
    # print(line)
    point1 = np.array(line.coords)[0]
    point2 = np.array(line.coords)[-1]
    # Calculate the angles formed by point1 and point2 with the reference point
         # Creating vectors from centroid to the points
    vector1 = np.array([point1[0] - centroid.x, point1[1] - centroid.y])
    vector2 = np.array([point2[0] - centroid.x, point2[1] - centroid.y])

    # Calculate cross product
    cross_product = np.cross(vector1, vector2)

    # Determine left and right based on the cross product
    if cross_product > 0:
        left_point, right_point = point2, point1
        indexL, indexR = -1, 0
    else:
        left_point, right_point = point1, point2
        indexL, indexR = 0, -1

    return left_point, right_point, indexL, indexR


### Set setback distance from context
def setSetbacks(input_dict, StoreyHeight, setbacks):
    """
    input: input_dict: dictionary
    {
        "search_result": searchkey,
        "address": address,
        "postal_code": postal_code,
        "PR": plot_ratio,
        "geometry": geometry,
    }
    export: setback_data: dictionary
    {
        "search_result": searchkey,
        "address": address,
        "postal_code": postal_code,
        "PR": plot_ratio,
        "geometry": geometry,
        "setback": setback,
    }
    """
    emit_log("Calculating setback...")
    place_data = input_dict.copy()  # place_data for manipulation

    # Add buffers -------------------------------------------------------------------------------------------------
    # convert place_data['geometry] to shapely polygon
    place_data["geometry_poly"] = Polygon(place_data["geometry"][0])
    try:
        place_name = place_data['name']+', Singapore'
        OSM_place_data = ox.geocoder.geocode_to_gdf(place_name)
        print(place_name, 'found')
    except:
        
        print('Error: Name > OSM place not found')
        try:
            place_name = place_data['search_result']+', Singapore'
            OSM_place_data = ox.geocoder.geocode_to_gdf(place_name)
            print(place_name, 'found')
        except:
            print('Error: Query > OSM place not found')
            try:
                place_name = place_data['address']+', Singapore'
                OSM_place_data = ox.geocoder.geocode_to_gdf(place_name)
                print(place_name, 'found')
            except:
                print('Error: Address > OSM place not found')

    buffer = 0.00045  # in  geodetic coordinates
    buffer2 = 0.00015  # in  geodetic coordinates
    buffer3 = 0.00015  # in  geodetic coordinates

    # add the buffer to the original geometry
    place_data["geometry_buffered"] = place_data["geometry_poly"].buffer(
        buffer, cap_style=3, join_style=2
    )

    OG_polygon = Polygon(place_data["geometry_poly"])

    # Simplify the polygon ----------------------------------------------------------------------------------------
    tolerance = 0.00025
    simplified_polygon = OG_polygon.simplify(tolerance, preserve_topology=True)

    # Minor buffer (Buffer2)
    simplified_polygon_buffer2 = simplified_polygon.buffer(buffer2, cap_style=3,join_style=2)
    while True: # increase buffer2 until it contains OG_polygon
        if OG_polygon.intersects(simplified_polygon_buffer2.exterior):
            buffer2+=0.0001
            simplified_polygon_buffer2 = simplified_polygon.buffer(buffer2, cap_style=3,join_style=2)
        else:
            break

    print('Minor buffer size: ',buffer2)

    # Outer buffer (Buffer2*2)
    simplified_polygon_buffer = simplified_polygon.buffer(buffer2*2, cap_style=3,join_style=2)

    #  Inner buffer (-Buffer)
    simplified_polygon_bufferin = simplified_polygon.buffer(-buffer3, cap_style=3,join_style=2)
    while True: # decrease buffer3 until it does not overlaps OG_polygon
        if simplified_polygon_bufferin.overlaps(OG_polygon):
            buffer3+=0.0001
            simplified_polygon_bufferin = simplified_polygon.buffer(-buffer3, cap_style=3,join_style=2)
        else:
            break

    print('Inner buffer size: ',buffer3)

    # tags to extract
    tags = {
        "landuse": True,
        "highway": [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
        ],
        # 'highway': True,
    }

    # Create boundary zones for simplifed line -------------------------------------------------------------------------------------------------

    vertices_simplified = simplified_polygon_buffer2.exterior.coords[:-1] #middle ring
    vertices_buffer = simplified_polygon_buffer.exterior.coords[:-1]  # outer ring
    vertices_in = simplified_polygon_bufferin.exterior.coords[:-1]  # inner ring (for capturing the features missing from the middle ring)

    osm_polygon = OSM_place_data["geometry"]
    # Snap the simplified polygon to the OSM polygon to get features that are missing from the simplified polygon
    snapped_polygon = snap(simplified_polygon_buffer2, osm_polygon, tolerance=0.0002)
    snapped_polygon = snapped_polygon[0]

    lines_simplified = []
    lines_buffer = []
    lines_in = []

    # ------------------------------ Construct lines from the vertices ------------------------------
    for i in range(len(vertices_simplified)):
        # line = LineString([vertices_simplified[i],vertices_simplified[i+1]])
        if i == len(vertices_simplified)-1:
            line = [vertices_simplified[i],vertices_simplified[0]]
        else:
            line = [vertices_simplified[i],vertices_simplified[i+1]]
        
        lines_simplified.append(line)
        # print(i)
    for i in range(len(vertices_buffer)):
        if i == len(vertices_buffer)-1:
            line = [vertices_buffer[i],vertices_buffer[0]]
        else:
            line = [vertices_buffer[i],vertices_buffer[i+1]]
        lines_buffer.append(line)
        # print(i)
    for i in range(len(vertices_in)):
        if i == len(vertices_in)-1:
            line = [vertices_in[i],vertices_in[0]]
        else:
            line = [vertices_in[i],vertices_in[i+1]]
        lines_in.append(line)
        # print(i)

    # ------------------------------ Construct connecting lines from mid-outer ------------------------------

    MO_lines = {}
    OM_lines = {}
    lines = []

    # find the nearest point on the outer ring to the middle ring
    for vertex in vertices_buffer:
        # print(vertex)
        pointv = Point(vertex)
        point1, point2 = nearest_points(pointv,MultiPoint(vertices_simplified)) #MutliPoint to allow the vertex to be returned instead of the inbetweens

        # retrieve the point2 that is from middle ring
        # snapping to nearest vertex in middle ring  (nearest_point returns approx which is not in vertices_simplified)
        dist = math.inf
        closest_v = None
        for v in vertices_simplified: #loops through to find the vertex on the vertices_simplified that is closest to point2
            vp = Point(v)
            if point2.distance(vp) < dist:
                dist = point2.distance(vp)
                closest_v = v

        MO_lines[closest_v] =  vertex
        OM_lines[vertex] = closest_v #for extracting the lines from outer to middle

        # print(point1,point2)
        line = [vertex,closest_v]
        lines.append(line)

    # ------------------------------ Construct connecting lines from inner-outer ------------------------------

    # lines2 = []
    lines3 = []  # for long branch lines that connects O - M_OG - I
    # boundinglines = []

    #find the nearest point on the outer ring to the inner ring
    for vertex in vertices_simplified:
        # print(vertex)
        pointv = Point(vertex)
        point1, point2 = nearest_points(pointv,MultiPoint(vertices_in)) #MutliPoint to allow the vertex to be returned instead of the inbetweens

        # retrieve the point2 that is from vertices_simplified
        # find point2 with shortest distance to all coordinates in vertices_simplified (nearest_point returns approx which is not in vertices_simplified)
        dist = math.inf
        closest_v = None
        for v in vertices_in: #loops through to find the vertex on the vertices_simplified that is closest to point2
            vp = Point(v)
            if point2.distance(vp) < dist:
                dist = point2.distance(vp)
                closest_v = v

        # extracting closest point from OG by intersecting the line with OG
        intersect_line = LineString([closest_v,point1])
        # intersect_line = LineString([closest_v,MO_lines[vertex]])

        intersect_OG = OG_polygon.exterior.intersection(intersect_line)
        closest_OG = snap(intersect_OG, OG_polygon, tolerance=0.000002)
        if (closest_OG.type == 'MultiPoint'):
            # # extracting closest point from OG
            dist = math.inf
            closest = None
            # loop through all the points in the multipoint and find the closest point
            for og in closest_OG.geoms:
            
                ogp = Point(og)
                if point1.distance(ogp) < dist:
                    dist = point1.distance(ogp)
                    closest_OG = og
        closest_OG = (closest_OG.x,closest_OG.y)
        line3 = [closest_v,closest_OG,MO_lines[vertex]]
        # print(line3)
        lines3.append(line3)

    # Create Graphs -------------------------------------------------------------------------------------------------
    # Outer-Middle Graph
    G = nx.Graph()
    # Iterate through new polylines and add nodes and edges to the graph
    for line in lines:  # connecting lines from mid-outer
        # Iterate through coordinates in the line and add nodes and edges
        for i in range(len(line) - 1):
            G.add_edge(line[i], line[i + 1], weight=0)

    for line in lines_buffer:  # add edges to the buffer lines
        for i in range(len(line) - 1):
            G.add_edge(line[i], line[i + 1], weight=0)

    for line in lines_simplified:
        for i in range(len(line) - 1):
            G.add_edge(line[i], line[i + 1], weight=0)

    cycles = list(nx.simple_cycles(G))

    # Filter the cycles with at most 4 nodes
    filtered_cycles_temp = [cycle for cycle in cycles if len(cycle) == 4]
    filtered_cycles = []
    # loop to remove those that are the same as the simplified polygon buffer
    for cycle in filtered_cycles_temp:
        bb1, bb2, bb3, bb4 = cycle
        checking_polygon = Polygon([bb1, bb2, bb3, bb4])

        # print(cycle)
        if simplified_polygon_buffer2.equals(checking_polygon):
            pass
        elif simplified_polygon_buffer.equals(checking_polygon):
            pass
        else:
            filtered_cycles.append(cycle)

    # Inner-OG-Outer Graph
    G3 = nx.Graph()
    # Iterate through new polylines and add nodes and edges to the graph
    for line in lines3:  # O-M_OG-I
        # Iterate through coordinates in the line and add nodes and edges
        G3.add_edge(line[0], line[1], weight=0)
        G3.add_edge(line[1], line[2], weight=0)

    for line in lines_buffer:  # add edges to the buffer lines
        for i in range(len(line) - 1):
            G3.add_edge(line[i], line[i + 1], weight=0)

    for line in lines_in:
        for i in range(len(line) - 1):
            G3.add_edge(line[i], line[i + 1], weight=0)

    cycles3 = list(nx.simple_cycles(G3))

    # Filter the cycles with at most 6 nodes
    filtered_cycles3 = [cycle for cycle in cycles3 if len(cycle) == 6]

    # Matching the cycles from the 2 graphs ---------------------------------------------------------------------------------
    matching_pairs = []

    # Iterate through lists in 1 (Outer-Mid)
    for cyc2 in filtered_cycles3:
        # print(cycle1)
        # Iterate through lists in 2 (Mid-Inner)
        setB = set(cyc2)
        for cyc1 in filtered_cycles:
            # Check if there are matching tuple pairs in the current lists
            setA = set(cyc1)
            common_elements = setA.intersection(setB)
            if len(common_elements) == 2:
                # print(len(setC))
                matching_pairs.append((cyc1, cyc2))

    bb_tags = {}  # store the tags,  for each bounding box

    for i, pair in enumerate(matching_pairs, 0):
        # print(i)
        cycle, cycle2 = pair

        # print(f"Cycle {i} a: {cycle}")
        # print(f"Cycle {i} b: {cycle2}")
        weighted_edges = []
        bb1, bb2, bb3, bb4 = cycle
        # print(bb1,bb2,bb3,bb4)
        bounding_box = Polygon([bb1, bb2, bb3, bb4])

        # print(polygon.covers(Point(bb1)))
        if simplified_polygon_buffer2.covers(Point(bb1)):
            weighted_edges.append(bb1)
        if simplified_polygon_buffer2.covers(Point(bb2)):
            weighted_edges.append(bb2)
        if simplified_polygon_buffer2.covers(Point(bb3)):
            weighted_edges.append(bb3)
        if simplified_polygon_buffer2.covers(Point(bb4)):
            weighted_edges.append(bb4)

        # print(weighted_edges)
        polyline = LineString(weighted_edges)

        # try:
        # --------------------------------- Extract closest feature from each M-O boundary ---------------------------------

        # df_building = ox.features.features_from_bbox(north-0.0003,south+0.0003,east-0.0003,west+0.0003,{'building': True})
        gdf = ox.features.features_from_polygon(bounding_box, tags)
        gdf["distance_to_polyline"] = gdf["geometry"].apply(
            lambda feature: calculate_distance(polyline, feature)
        )
        gdf_neg = ox.features.features_from_polygon(snapped_polygon, tags)
        gdf_neg['geometry'] = snapped_polygon

        # remove features that are within the snapped polygon
        mask = ~gdf["geometry"].isin(gdf_neg["geometry"])
        gdf = gdf.loc[mask].copy()

        closest_feature = gdf[
            gdf["distance_to_polyline"] == gdf["distance_to_polyline"].min()
        ]

        # --------------------------------- Extract OG lines within I-O boundary ---------------------------------
        bounding_box_in = Polygon(cycle2)

        if OG_polygon.intersects(bounding_box_in):
            # Extract the intersection of the polygon and the bounding box
            # difference = shapely.difference(OG_polygon,bounding_box_in)
            # print(difference)
            intersection = OG_polygon.intersection(bounding_box_in)
            #remove bounding box from intersection
            intersection2 = bounding_box_in.difference(intersection)


            if isinstance(intersection2, MultiPolygon):
                print(f'is MultiPolygon: {intersection2}')
                area = 0
                for element in intersection2.geoms:
                    if element.area > area:
                        area = element.area
                        temp = element
                intersection2 = temp

            #get the common points between the intersection2 and the OG_polygon
            og_mls = OG_polygon.exterior.intersection(intersection2.exterior)

            og = None
            #if is multiLineString, merge them
            if isinstance(og_mls, MultiLineString):
                og = linemerge(og_mls)
            else:
                og = og_mls

            # point1 = Point(np.array(og.coords)[0])
            # point2 = Point(np.array(og.coords)[-1])

        # save
        bb_tags[i] = {
            "gdf": gdf,
            "tag": closest_feature,
            "bb": bounding_box,  # bounding box
            "cycle": cycle,
            "innerline": polyline,  # innerline is the polyline that is inside the polygon to be used as the inner boundary
            "og": og,  # og is the intersection of the polygon and the bounding box
            # "og_point1": point1,
            # "og_point2": point2,
        }

    # Left Right adjacency -------------------------------------------------------------------------------------------------
    list_of_sides = list(bb_tags.keys())

    centroid = OG_polygon.centroid

    # identify Left and Right from the list of sides
    for side in list_of_sides:

        # print(side)
        line = bb_tags[side]['og'] #get the og line for that side
        #get list of sides that does not include the current side
        othersides = list_of_sides.copy()
        othersides.remove(side)

        x,y = line.coords.xy
        
        L_point, R_point, indexL, indexR = compare_points(line,centroid)
        L_point, R_point = Point(L_point), Point(R_point)
        bb_tags[side]['L_point'] = L_point
        bb_tags[side]['L_index'] = indexL
        bb_tags[side]['R_point'] = R_point
        bb_tags[side]['R_index'] = indexR
        # Find the corresponding sides for the left and right points
        Lflag,Rflag = False,False
        for key in othersides:
            if (Lflag and Rflag):
                break
            else:
                # print(Lflag,Rflag)
                polyline = bb_tags[key]['og'] # to check which point is touching the consequtive polyline

                if L_point.touches(polyline): #Left point
                    # print('Lyes')
                    bb_tags[side]['L'] = key
                    # bb_tags[key]['R'] = side
                    Lflag = True
                if R_point.touches(polyline): #Right point
                    # print('Ryes')
                    bb_tags[side]['R'] = key
                    # bb_tags[key]['L'] = side
                    Rflag = True
   
    utm_zone = pyproj.Proj(init="EPSG:3857")

    # Apply setback values -------------------------------------------------------------------------------------------------
    def applySetback(bb_tags, StoreyHeight, setback_name):
        setback_common = pd.DataFrame(
            setbacks["extracted"]["common"], columns=setbacks["index"]["common"]
        )
        setback_road = pd.DataFrame(
            setbacks["extracted"]["road"], columns=setbacks["index"]["road"]
        )

        for i in bb_tags:
            # try:
            # if 'highway' in bb_tags[i]['tag'].columns and not(bb_tags[i]['tag']['highway'].isna().any()): #if key_to_check in gdf.columns:
            if "highway" in bb_tags[i]["tag"].columns and not(bb_tags[i]["tag"]["highway"].isna().any()):  # if key_to_check in gdf.columns:
                setback_type = "road"
                # print(bb_tags[i]['tag']['highway'])
                road_type = bb_tags[i]["tag"]["highway"].iloc[0]
                if StoreyHeight > 5:
                    if road_type in ["motorway", "trunk", "primary", "secondary"]:
                        Cat = "Category 1"
                        Dev = "Residential 6 storeys and above"
                    elif road_type in ["tertiary"]:
                        Cat = "Category 2"
                        Dev = "Residential 6 storeys and above"
                    elif road_type == "residential":
                        Cat = "Category 4 - 5 and slip road"
                        Dev = "Residential"
                else:
                    if road_type in ["motorway", "trunk", "primary", "secondary"]:
                        Cat = "Category 1"
                        Dev = "Residential up to 5 storeys"
                    elif road_type in ["tertiary"]:
                        Cat = "Category 2"
                        Dev = "Residential up to 5 storeys"
                    elif road_type == "residential":
                        Cat = "Category 4 - 5 and slip road"
                        Dev = "Residential"
                        

                raw_val = setback_road.loc[
                    (setback_road["Road Category"] == Cat)
                    & (setback_road["Type of Development"] == Dev)
                ][setbacks["index"]["road"][2]].iloc[0]
                match = re.search(r"\d+", raw_val)

                if match:
                    setback_val = float(match.group())
                else:
                    # print("No float value found.")
                    pass

            elif "landuse" in bb_tags[i]["tag"].columns:
                setback_type = "common"
                # type = bb_tags[i]['tag']['landuse'].iloc[0]
                setback_val = setback_common.loc[
                    setback_common["Storey Height"] == f"{StoreyHeight}",
                    "Common Boundary Setback for Flats",
                ].iloc[0]
                setback_val = float(setback_val.split("m")[0])
                # print('is landuse')
            bb_tags[i][setback_name] = {
                "type": setback_type,
                "value": setback_val,
            }
        return bb_tags


    # Apply setback values -------------------------------------------------------------------------------------------------
    def get_adjacent_side(side, direction, bb_tag):
        # try:
        if direction == "L":
            return bb_tag[side]["L"]
        elif direction == "R":
            return bb_tag[side]["R"]
        # except:
        #     return None

    def set_setback(side, bb_tag, setback_name):
        og = bb_tag[side]["og"]
        buffer_val = bb_tag[side][setback_name]["value"]
        # print(og)

        # Convert meters to geodetic coordinates
        _c, converted_buffer = utm_zone(5, buffer_val, inverse=True)

        # print(converted_buffer)
        ogL = og.parallel_offset(converted_buffer, 'left', join_style= 1, mitre_limit=0.5)
        ogR = og.parallel_offset(converted_buffer, 'right', join_style= 1, mitre_limit=0.5)

        #get the center of the buffer_line
        ogL_center = ogL.centroid
        ogR_center = ogR.centroid

        setback_line = None
        #check which center is within the polygon
        if OG_polygon.contains(ogL_center) and not(OG_polygon.contains(ogR_center)):
            print('ogL is within polygon')
            setback_line = ogL
        else:
            print('ogL is not within polygon')
            setback_line = ogR

        #save the new og to the dictionary
        if isinstance(setback_line, MultiLineString):
            setback_line = LineString([point for line in setback_line.geoms for point in line.coords])

        bb_tags[side][setback_name]['line'] = setback_line
        return bb_tags

    def loopsetback(side,Olist, setback_name, bb_tags):
        if len(Olist) == 0: #base case
            return
        else:
            #remove side from list
            Olist.remove(side)
            print(f"Side {side} removed from list.")
            # offset the setback of the side
            bb_tags = set_setback(side,bb_tags,setback_name)
            print(f"Setback of side {side} set.")
            Left = get_adjacent_side(side,'L',bb_tags)
            Right = get_adjacent_side(side,'R',bb_tags)
            print(f"Left: {Left}, Right: {Right}")
            if Left in Olist and Right in Olist:
                indexL = Olist.index(Left)
                indexR = Olist.index(Right)

                if indexL < indexR: # left is bigger than right
                    loopsetback(Right,Olist, setback_name,bb_tags) #Go small first
                    # loopsetback(Left,Olist)
                else:
                    loopsetback(Left,Olist,setback_name,bb_tags) #Go small first
                    # loopsetback(Right,Olist)
            else:
                if Left in Olist:
                    loopsetback(Left,Olist,setback_name,bb_tags)
                elif Right in Olist:
                    loopsetback(Right,Olist,setback_name,bb_tags)
                else:
                    pass
    
    for storey in StoreyHeight:
        setback_name = f"setback{storey}"
    
        bb_tags = applySetback(bb_tags, storey, setback_name)
        # print(bb_tags)
        bb_tags = dict(sorted(bb_tags.items(), key=lambda x: x[1][setback_name]['value'], reverse=True))
        for i, value in enumerate(bb_tags):
            print(f"{i}\t|{value}: {bb_tags[value]['L']} {bb_tags[value]['R']} {bb_tags[value]['L_point']}, {bb_tags[value]['R_point']}")
        ordered_list = list(bb_tags.keys())
        loopsetback(ordered_list[0], ordered_list, setback_name, bb_tags)

    # print(sorted_ls)
    def getextrapolatedLine(p1, p2, factor=2):
        "Creates a line extrapolated in p1->p2 direction"
        # EXTRAPOL_RATIO = 10
        a = p1
        b = (p1[0] + factor * (p2[0] - p1[0]), p1[1] + factor * (p2[1] - p1[1]))
        return LineString([a, b])

    def extendtoSide(main, direction, bb_tag, setback_name,factor=2):
        line = bb_tag[main][setback_name]["line"]
        point, start_index = (
            bb_tag[main][f"{direction}_point"],
            bb_tag[main][f"{direction}_index"],
        )

        if start_index == 0:
            # next_index = 1
            scaled_last_segment = getextrapolatedLine(
                line.coords[1], line.coords[0], factor
            )

        else:  # start_index == -1
            # next_index = -2
            scaled_last_segment = getextrapolatedLine(
                line.coords[-2], line.coords[-1], factor
            )

        # print(last_segment.boundary.geoms[1])
        return scaled_last_segment
    def LonLat_To_XY(Lon, Lat): #convert lat long to coordinates
        utm_zone = pyproj.Proj(init="EPSG:3857")
        return utm_zone(Lon, Lat) 

    def convert_to_SVG(coordinates,svg_height=150,svg_width=150):
        #convert all coordinates to UTM
        utm_coordinates = []
        for x,y in coordinates:
            utm_coordinates.append(LonLat_To_XY(x,y))

        #normalise coordinates to svg_height
        latitudes = [lat for lat, lon in utm_coordinates]
        longitudes = [lon for lat, lon in utm_coordinates]

        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)

        translated_coords = [(lat - min_lat, lon - min_lon) for lat, lon in utm_coordinates]

        # Scale factors
        scale_x = svg_width / (max_lon - min_lon)
        scale_y = svg_height / (max_lat - min_lat)

        scaled_coords = [(lon * scale_x, lat * scale_y) for lat, lon in translated_coords]
        #return in svg format
        return ' '.join([f'{x},{y}' for x, y in scaled_coords])

    # Extend lines to create a connected polygon -----------------------------------------------------------------------
    def extendtrimsetback(bb_tags, setback_name):
        bb_tags = dict(sorted(bb_tags.items(), key=lambda x: x[1][setback_name]['value'], reverse=True))
        sorted_ls = list(bb_tags.keys())
        
        # Identify the sides that are are not connected to the L and R
        for i in sorted_ls:
            # plot setbacks
            setback_line = bb_tags[i][setback_name]["line"]
            if isinstance(setback_line, MultiLineString):
                setback_line = LineString([point for line in setback_line.geoms for point in line.coords])

            polyline_m = bb_tags[i][setback_name]["line"] 
            polyline_l = bb_tags[bb_tags[i]["L"]][setback_name]["line"]
            polyline_r = bb_tags[bb_tags[i]["R"]][setback_name]["line"]
            poly_temp = polyline_m

            # all connected
            if polyline_m.intersects(polyline_l) and polyline_m.intersects(polyline_r):
                print(i, "all connected")
                pass
            else:
                factor_limit = 10 
                merged = None
                factor = 1.2
                # only L connected
                if polyline_m.intersects(polyline_l) and not(polyline_m.intersects(polyline_r)):
                    factor = 1.2
                    print(i, "R not connected")
                    while (not(poly_temp.intersects(polyline_r))) and factor < factor_limit:
                        poly_temp = extendtoSide(i,'R',bb_tags, setback_name,factor)
                        factor += 0.2

                    print('factor:',factor)
                    # merged = linemerge([polyline_m,poly_temp])
                    xp, yp = poly_temp.coords[1]
                    if bb_tags[i]['R_index'] == 0:
                        merged = LineString([(xp,yp)]+list(polyline_m.coords))
                    else:
                        merged = LineString(list(polyline_m.coords)+[(xp,yp)])

                # only R connected
                elif not (polyline_m.intersects(polyline_l)) and polyline_m.intersects(polyline_r):
                    print(i, "L not connected")
                    while (not (poly_temp.intersects(polyline_l))) and factor < factor_limit:
                        poly_temp = extendtoSide(i, "L", bb_tags, setback_name, factor)
                        factor += 0.2
                    print("factor:", factor)
                    xp, yp = poly_temp.coords[1]
                    if bb_tags[i]['L_index'] == 0:
                        merged = LineString(list(polyline_m.coords)+[(xp,yp)])
                    else:
                        merged = LineString([(xp,yp)]+list(polyline_m.coords))

                # all not connected
                elif not (polyline_m.intersects(polyline_l)) and not (polyline_m.intersects(polyline_r)):
                    print(i, "all not connected")
                    poly_tempL = polyline_m
                    poly_tempR = polyline_m

                    print(i, "R not connected")
                    while (
                        not (poly_tempR.intersects(polyline_r))
                    ) and factor < factor_limit:
                        poly_tempR = extendtoSide(i, "R", bb_tags, setback_name, factor)
                        print("factor:", factor)
                        factor += 0.2

                    print(i, "L not connected")
                    while (
                        not (poly_tempL.intersects(polyline_l))
                    ) and factor < factor_limit:
                        poly_tempL = extendtoSide(i, "L", bb_tags, setback_name, factor)
                        print("factor:", factor)
                        factor += 0.2

                    merged = linemerge([poly_tempL, polyline_m, poly_tempR])

                if isinstance(merged, MultiLineString):
                    print("is MultiLineString")
                    # print(merged)
                    # merged = LineString([point for line in merged.geoms for point in line.coords])
                    merged = LineString([point for line in merged.geoms for point in line.coords])
                    # merged = linemerge(merged)
                    print(merged)

                bb_tags[i][setback_name]["line"] = merged
        

    # Trimming Excess  -------------------------------------------------------------------------------------------------
        emit_log(f"Trimming excess setback for {setback_name}...")
        for i in bb_tags:
            setback_line = (bb_tags[i][setback_name]['line'])
            setback_lineL = (bb_tags[bb_tags[i]['L']][setback_name]['line'])
            setback_lineR = (bb_tags[bb_tags[i]['R']][setback_name]['line'])

            #find intersection with R
            if setback_line.intersects(setback_lineR):
                print(f'{i} intersects R')
                intersectionR = setback_line.intersection(setback_lineR)
                #if multiple points, find the nearest point to the setback distance
                if isinstance(intersectionR, MultiPoint):
                    print(intersectionR,'is MultiPoint')
                    R_point = bb_tags[i]['R_point']
                    setback_dist = bb_tags[i][setback_name]['value']
                    #find the nearest point
                    closest = float('inf') #set closest to infinity
                    temp_point = None
                    for point in intersectionR.geoms:
                        dist = calculate_distance(point, R_point)
                        if abs(dist - setback_dist) < closest:
                            closest = abs(dist - setback_dist)
                            temp_point = point
                    intersectionR = temp_point

                minimum_distance = nearest_points(intersectionR, setback_line)[1]

                #split the setback line by the point
                geom = split(snap(setback_line, minimum_distance, 0.000001), minimum_distance)
                if len(geom.geoms) >= 2:
                    setback_temp = geom.geoms[0]
                else:
                    setback_temp = geom.geoms[1]
                
            # find intersection with L
            if setback_line.intersects(setback_lineL):
                print(f'{i} intersects L')
                intersectionL = setback_line.intersection(setback_lineL)
                if isinstance(intersectionL, MultiPoint):
                    print(intersectionL,'is MultiPoint')
                    L_point = bb_tags[i]['L_point']
                    setback_dist = bb_tags[i][setback_name]['value']
                    #find the nearest point
                    closest = float('inf')
                    temp_point = None
                    for point in intersectionL.geoms:
                        dist = calculate_distance(point, L_point)
                        if abs(dist - setback_dist) < closest:
                            closest = abs(dist - setback_dist)
                            temp_point = point
                    intersectionL = temp_point
                minimum_distance = nearest_points(intersectionL, setback_line)[1]
                # print(minimum_distance)
                #split the setback line by the point
                geom2 = split(snap(setback_temp, minimum_distance, 0.000001), minimum_distance)
                if len(geom2.geoms) >= 2:
                    setback_temp = geom2.geoms[1]
                else:
                    setback_temp = geom2.geoms[0]


            bb_tags[i][setback_name]['trimmed'] = setback_temp

    # Create the final setback polygon -----------------------------------------------------------------------------------

        emit_log(f"Joining setback polygon for {setback_name}...")
        index_list = list(bb_tags.keys())
        annotated_setback = {}

        for side in index_list:

            # print(side)
            line = bb_tags[side][setback_name]['trimmed'] #get the og line for that side
            center_point = line.interpolate(line.length / 2)
            #get list of sides that does not include the current side
            annotated_setback[side] = { 
                'coord': [center_point.x,center_point.y],
                'setback': bb_tags[side][setback_name]['value']
                }
            othersides = index_list.copy()
            othersides.remove(side)
            
            L_point, R_point, indexL, indexR = compare_points(line,centroid)
            L_point, R_point = Point(L_point), Point(R_point)
            bb_tags[side][setback_name]['L_point'] = L_point
            bb_tags[side][setback_name]['L_index'] = indexL
            bb_tags[side][setback_name]['R_point'] = R_point
            bb_tags[side][setback_name]['R_index'] = indexR
            # Find the corresponding sides for the left and right points
            Lflag,Rflag = False,False
            for key in othersides:
                if (Lflag and Rflag):
                    break
                else:
                    # print(Lflag,Rflag)
                    polyline = LineString(bb_tags[key][setback_name]['trimmed']) # to check which point is touching the consecutive polyline

                    if L_point.buffer(0.000001).intersects(polyline) and not(Lflag): #Left point
                        # print('Lyes')
                        bb_tags[side][setback_name]['L'] = key
                        # bb_tags[key]['R'] = side
                        Lflag = True
                    if R_point.buffer(0.000001).intersects(polyline) and not(Rflag): #Right point
                        # print('Ryes')
                        bb_tags[side][setback_name]['R'] = key
                        # bb_tags[key]['L'] = side
                        Rflag = True
        

        coordinates = []
        i = index_list[0]
        # isFirst = True
        while index_list != []:
            current_i = i
            setback_line = bb_tags[current_i][setback_name]['trimmed']
            x,y = setback_line.coords.xy

            #check if the first point is already in the list of coordinates
            # left = bb_tags[current_i][setback_name]['L_point']
            # right = bb_tags[current_i][setback_name]['R_point']
            left_index = bb_tags[current_i][setback_name]['L_index']
            # right_index = bb_tags[current_i][setback_name]['R_index']
            if left_index == 0:
                for element in list(setback_line.coords):
                    coordinates.append(element)
            else:
                for element in reversed(list(setback_line.coords)):
                    coordinates.append(element)
            index_list.remove(i)
            i = bb_tags[current_i][setback_name]["R"]

        return coordinates,annotated_setback

    setback_data = input_dict

    for storey in StoreyHeight:
        setback_name = f"setback{storey}"
        coordinates,annotated_setback = extendtrimsetback(bb_tags, setback_name)
        setback_data[setback_name] = {'polygon':coordinates,'annotation':annotated_setback,'svg':convert_to_SVG(coordinates)}
        print(f"{setback_name} is set.")
    # setback_polygon = Polygon(coordinates)
    

    plotratio = float(setback_data["PR"])

    centroid = OG_polygon.centroid
    # OG_coords = list(OG_polygon.exterior.coords)
    # setback_data["svg"] = {
    #     'site':convert_to_SVG(OG_coords)
    #     }

    setback_data["centroid"] = [centroid.x,centroid.y] #get the centroid of the OG polygon
    area = OG_polygon.area*(111000**2) #get the area of the OG polygon in m2
    setback_data['site_area'] = area
    setback_data['GFA'] = area*plotratio

    emit_log(f"Setback and map data is packaged!")
    return setback_data

def getMapboxStaticImg(centroid,access_token):

    emit_log("Loading mapbox image...")

    # style = 'mapbox/streets-v11'
    # style = 'mapbox/dark-v11'
    style = 'yeokewei/cloybodi8014m01o49qs0776q'

    longitude = centroid[0]
    latitude = centroid[1]
    zoom = 15
    bearing = 0
    width = 250
    height = 250

    # Create the URL for the request
    # url = f'https://api.mapbox.com/styles/v1/{style}/static/geojson({encoded_geojson})/{longitude},{latitude},{zoom},{bearing}/{width}x{height}@2x?access_token={access_token}'
    url = f'https://api.mapbox.com/styles/v1/{style}/static/{longitude},{latitude},{zoom},{bearing}/{width}x{height}@2x?access_token={access_token}'

    # Make the GET request to the Mapbox API
    response = requests.get(url)
    # print(response.content)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the image to a file
        # with open('mapbox_static_image.png', 'wb') as file:
        #     file.write(response.content)
        # Return the image
        print('Mapbox: Successfully retrieved image.')
        encoded_image = base64.b64encode(response.content).decode('utf-8')
        return encoded_image
    else:
        print('Mapbox: Failed to download image:', response.content)
        return {'error': 'Failed to fetch image from Mapbox'}


# RL Class -----------------------------------------------------------------------------------------------------------------------
class SpaDesPlacement(gym.Env):
    def __init__(self, sites_info, building, grid_size=(50, 50),name='clavon'):
        super(SpaDesPlacement, self).__init__()
        self.max_boxes = 10
        self.box_placed = 0
        self.grid_size = grid_size
        self.sites_info = sites_info
        self.name = name  # None
        self.site_boundary, self.site = self._generate_site(
            sites_info, name=self.name)
        self.site_coverage = self.site['site_coverage']  # Next: GFA ...
        # Next:Meters value ...
        self.building_scale = self.site['building_scale']
        self.grid = self.update_grid_with_polygon(
            self.site_boundary, init_site=True)
        self.site_pixel = np.count_nonzero(self.grid == 1)
        self.total_footprint_pixel = 0
        self.building = building  # Next: More buildings ...
        self.building_list = []
        self.action_space = spaces.Box(low=np.array(
            [-1, -1, 3, -1]), high=np.array([1, 1, 10, 1]), shape=(4,), dtype=float)
        self.observation_space = spaces.Box(
            0, 1, shape=(np.prod(grid_size),), dtype=np.float32)
        self.boxes = np.empty((1, 4), dtype=float)
        self.state = self._get_state()

    def reset(self, seed=1, **kwargs):
        self.box_placed = 0
        if kwargs.get('name') is not None:
            self.site_boundary, self.site = self._generate_site(
                self.sites_info, name=kwargs.get('name'))
        else:
            self.site_boundary, self.site = self._generate_site(
                self.sites_info, name=self.name)
        self.site_coverage = self.site['site_coverage']  # Next: GFA ...
        # Next:Meters value ...
        self.building_scale = self.site['building_scale']
        self.building_list = []
        self.total_footprint_pixel = 0
        self.boxes = np.empty((1, 4), dtype=float)
        self.grid = self.update_grid_with_polygon(
            self.site_boundary, init_site=True)
        self.site_pixel = np.count_nonzero(self.grid == 1)
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        ''' 
        1. reward (+ve) for placing more building                                        
        2. reward (-ve) for placing building outside boundary                                                    Terminate
        3. reward (-ve) for placing building that collide with other buildings                                   Terminate
        4. reward (-ve) for placing building that violate interblock distance (short)    
        5. reward (-ve) for placing building that violate interblock distance (long)      
        6. reward (+ve) for not violating any interblock distance and collison                       
        7. reward (+ve) for placing building that is fulfiled site coverage                                       Terminate
        '''
        done = False
        is_valid = True
        x, y, height, angle = action
        x, y = ((x+1)*self.grid_size[0]/2)-0.1, ((y+1)*self.grid_size[0]/2)-0.1
        X, Y = self._resize_polygon(
            self.building, self.building_scale, (x, y), angle)
        building = Polygon(zip(X, Y))
        reward = 100  # * (len(self.boxes)) # reward for placing more building

        if self._building_outside_boundary(building):
            reward -= 400
            done = True
            is_valid = False

        else:
            reward_, is_valid = self._check_no_collision_and_interblock_distance(
                building, self.building_list)
            if reward_ < 0:
                reward = reward_
            else:
                reward += reward_
            if is_valid == False:
                done = True

        # store buildings properties for calculation of reward
        box = np.array([[x, y, height, angle]])
        self.boxes = np.append(self.boxes, box, axis=0)
        self.building_list.append(building)
        self.box_placed += 1

        # update state
        self.grid = self.update_grid_with_polygon(building)
        self.state = self._get_state()
        if self._site_coverage_covered() and is_valid:
            reward += 200
            print('site coverage covered')
            done = True

        # limit
        if self.box_placed >= 10:
            done = True

        return self.state, reward, done, None, {}

    def _site_coverage_covered(self):
        if (self.total_footprint_pixel / self.site_pixel > self.site_coverage):
            return True

    def _check_no_collision_and_interblock_distance(self, building, building_list):
        """
            Input: 
                building (shapely.Polygon): Polygon to be placed and checked for interblock distance and collision 
            Output:
                reward (int): Panalty for violation of interblock distance and collision, reward otherwise
                is_valid (bool): For Termination if building collide
        """

        bounds = building.minimum_rotated_rectangle.exterior.xy
        # get longest side of the building boundary
        x1, y1 = bounds[0][0], bounds[1][0]
        x2, y2 = bounds[0][1], bounds[1][1]
        x3, y3 = bounds[0][2], bounds[1][2]
        x4, y4 = bounds[0][3], bounds[1][3]
        w1 = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        w2 = math.sqrt((y3-y2)**2 + (x3-x2)**2)

        # extend both sides of the longest sides of the building boundary by interblock distance scaled to grid size
        interblock_distance = {"facing": 30, "non_facing": 10}
        grid_to_metre_ratio = 60/self.building_scale  # To be replaced ...
        interblock_dist = interblock_distance["facing"]/grid_to_metre_ratio
        if w1 > w2:
            projection_line = LineString([(x1, y1), (x2, y2)])
            projection_line2 = LineString([(x3, y3), (x4, y4)])
        else:
            projection_line = LineString([(x2, y2), (x3, y3)])
            projection_line2 = LineString([(x4, y4), (x1, y1)])
        buffer = projection_line.buffer(
            distance=-interblock_dist, cap_style="square", single_sided=True)
        buffer2 = projection_line2.buffer(
            distance=-interblock_dist, cap_style="square", single_sided=True)

        def calculate_angle(line1, line2):
            # Get vectors representing the lines
            vector1 = np.array(line1.coords[1]) - np.array(line1.coords[0])
            vector2 = np.array(line2.coords[1]) - np.array(line2.coords[0])

            # Calculate the dot product and magnitude of the vectors
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)

            # Calculate the cosine of the angle
            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Calculate the angle in radians and convert to degrees
            angle_in_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle_in_degrees = np.degrees(angle_in_radians)

            return min(angle_in_degrees, 180 - angle_in_degrees)

        reward = 0
        is_valid = True
        # check if building intersects with other buildings or violate interblock distance
        for other_box in building_list:
            if building.intersects(other_box):
                reward -= 400
                is_valid = False
            else:
                reward += 400

            if building.distance(other_box) < interblock_distance["non_facing"]/grid_to_metre_ratio:
                reward -= 200
            else:
                reward += 400
            other_box_bounds = other_box.minimum_rotated_rectangle.exterior.xy
            # get longest side of the other building boundary
            other_building_x1, other_building_y1 = other_box_bounds[0][0], other_box_bounds[1][0]
            other_building_x2, other_building_y2 = other_box_bounds[0][1], other_box_bounds[1][1]
            other_building_x3, other_building_y3 = other_box_bounds[0][2], other_box_bounds[1][2]
            other_building_w1 = math.sqrt(
                (other_building_y2-other_building_y1)**2 + (other_building_x2-other_building_x1)**2)
            other_building_w2 = math.sqrt(
                (other_building_y3-other_building_y2)**2 + (other_building_x3-other_building_x2)**2)

            if other_building_w1 > other_building_w2:
                other_building_projection_line = LineString(
                    [(other_building_x1, other_building_y1), (other_building_x2, other_building_y2)])
            else:
                other_building_projection_line = LineString(
                    [(other_building_x2, other_building_y2), (other_building_x3, other_building_y3)])
            if calculate_angle(projection_line, other_building_projection_line) < 30:
                if buffer.intersects(other_box) or buffer2.intersects(other_box):
                    reward -= 300
                else:
                    reward += 400
        return reward, is_valid

    def _building_outside_boundary(self, building):
        # Check if the building is outside the site boundary
        if not self.site_boundary.contains(building):
            return True
        return False

    def update_grid_with_polygon(self, polygon, init_site=False):
        rasterized = rasterio.features.geometry_mask(
            [polygon],
            out_shape=self.grid_size,
            transform=rasterio.transform.from_bounds(0, 0, self.grid_size[0], self.grid_size[1], width=self.grid_size[0], height=self.grid_size[1]), invert=True)
        if init_site:
            grid = np.full(self.grid_size, -10)
            updated_grid = grid + rasterized.astype(int) * 11
            return updated_grid
        else:
            grid = np.copy(self.grid)
            updated_grid = grid + rasterized.astype(int)
            self.total_footprint_pixel = np.count_nonzero(updated_grid > 1)
            # plt.imshow(updated_grid)
            # plt.show()
            return updated_grid

    def _resize_polygon(self, poly, desired_scale=50, center=None, angle=0):
        poly = affinity.rotate(poly, (angle+1)*180, origin='centroid')
        X, Y = poly.exterior.xy
        current_width = max(X) - min(X)
        current_height = max(Y) - min(Y)
        longest_axis = max(current_width, current_height)
        scale_factor = desired_scale / longest_axis
        if center is None:
            center_x = current_width / 2
            center_y = current_height / 2
        else:
            center_x, center_y = center
        scaled_polygon_x = [(x - min(X)) * scale_factor for x in X]
        scaled_polygon_y = [(y - min(Y)) * scale_factor for y in Y]
        scaled_polygon = Polygon(list(zip(scaled_polygon_x, scaled_polygon_y)))
        x_off = scaled_polygon.centroid.x - center_x
        y_off = scaled_polygon.centroid.y - center_y
        centered_polygon_x = [x - x_off for x in scaled_polygon_x]
        centered_polygon_y = [y - y_off for y in scaled_polygon_y]
        return centered_polygon_x, centered_polygon_y

    def _get_state(self):
        flat_grid = self.grid.flatten()
        return flat_grid

    def _generate_site(self, sites_info, name):
        if name is not None and name in sites_info.keys():
            site = sites_info[name]
        else:
            site = sites_info[random.choice(list(sites_info.keys()))]
        scale = max(self.grid_size[0], self.grid_size[1])
        x, y = self._resize_polygon(
            site['site_boundary'], scale, (self.grid_size[0]/2, self.grid_size[1]/2))
        return Polygon(list(zip(x, y))), site
    
def p2p_inference(query,svg_width=100,svg_height=100):
    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    inferences={}
    value_list = [256, 512, 768, 1024] #256, 512, 768, 1024 defines image size (higher the value list, the more building (red) pixels)
    for i in range(3): #number of images to generate
        inferences[f'output{i+1}'] = {"svg":{},"geojson":{}}
    # for i in range(3): #number of images to generate
    
        # randomly choose value from a list 
        value = random.choice(value_list)
        value_list.remove(value)
        print(value,'round') 
        IMG_HEIGHT = value
        IMG_WIDTH = value


        def load_and_preprocess_image(file_path):
            # Read the image
            image = cv2.imread(file_path)
            
            # Determine the target size (256x256)
            # target_size = (256, 256)
            target_size = (IMG_WIDTH, IMG_HEIGHT)
            
            # Calculate padding values
            height, width, _ = image.shape
            max_dim = max(height, width)
            pad_height = max_dim - height
            pad_width = max_dim - width
            
            # Create a constant white color image
            white_color = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
            
            # Insert the original image into the center of the white image
            white_color[pad_height//2:pad_height//2+height, pad_width//2:pad_width//2+width, :] = image
            
            # Resize the padded image to 256x256
            resized_image = cv2.resize(white_color, target_size)
            
            return resized_image

        def pad_and_resize(image , target_size=(256, 256), fill_color=(255, 255, 255)):
            # Determine the size of the padding
            height, width,channel = image.shape
            print('hw',height,width)
            padding_size = abs(height - width) // 2
            # max_dim = max(height, width)
            # pad_height = max_dim - height
            # pad_width = max_dim - width
            max_dim = max(height, width)
            pad_height = max_dim - height
            pad_width = max_dim - width
            white_color = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
            white_color[pad_height//2:pad_height//2+height, pad_width//2:pad_width//2+width, :] = image

        def resize(input_image, real_image, height, width):
            input_image = tf.image.resize(input_image, [height, width],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_image = tf.image.resize(real_image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return input_image, real_image

        # Normalizing the images to [-1, 1]
        def normalize(input_image, real_image):
            input_image = (input_image / 127.5) - 1
            real_image = (real_image / 127.5) - 1

            return input_image, real_image

        def load(image_file):
            print('image',image_file)
            
            # Read and decode an image file to a uint8 tensor
            image_f = tf.io.read_file(image_file)
            print(image_f)
            image = tf.io.decode_jpeg(image_f)

        

            w = tf.shape(image)[1]
            w = w // 2
            input_image = image[:, :w, :]
            real_image = image[:, w:, :]
            input_image = tf.image.resize(input_image,(IMG_WIDTH,IMG_HEIGHT)) #reduce
            real_image = tf.image.resize(real_image,(IMG_WIDTH,IMG_HEIGHT))
            # pad and resize to 256 256 
            # input_image = pad_and_resize(input_image)
            # real_image = pad_and_resize(real_image)

            # Convert both images to float32 tensors
            input_image = tf.cast(input_image, tf.float32)
            real_image = tf.cast(real_image, tf.float32)

            return input_image, real_image
        
        def load_image_test(image_file):
            input_image, real_image = load(image_file)
            
            input_image, real_image = resize(input_image, real_image,
                                            IMG_HEIGHT, IMG_WIDTH)
            input_image, real_image = normalize(input_image, real_image)

            return input_image, real_image

        def random_crop(input_image, real_image):
            stacked_image = tf.stack([input_image, real_image], axis=0)
            cropped_image = tf.image.random_crop(
            stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

            return cropped_image[0], cropped_image[1]   

        @tf.function()
        def random_jitter(input_image, real_image):
        # Resizing to 286x286
            input_image, real_image = resize(input_image, real_image, 286, 286)

            # Random cropping back to 256x256
            input_image, real_image = random_crop(input_image, real_image)

            if tf.random.uniform(()) > 0.5:
                # Random mirroring
                input_image = tf.image.flip_left_right(input_image)
                real_image = tf.image.flip_left_right(real_image)

            return input_image, real_image

        def generate_images_test(model, test_input, output_path):
            prediction = model(test_input, training=True)
            plt.figure(figsize=(30, 30))

            display_list = [prediction[0]]
            title = [ 'Predicted Image']

            for i in range(1):
                plt.subplot(1, 1, i+1)
                plt.title(title[i])
                # Getting the pixel values in the [0, 1] range to plot.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
                plt.savefig(output_path, dpi=300,bbox_inches='tight')
            # plt.show()
            # cv2.imwrite('/Users/jefflai/SpaDS/house_diffusion/data_clavon/prediction.png', prediction[0])
            # plt.savefig('/Users/jefflai/SpaDS/house_diffusion/data_clavon/prediction.png', dpi=300,bbox_inches='tight')
            return prediction[0]

        OUTPUT_CHANNELS = 3
        def upsample(filters, size, apply_dropout=False):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False))

            result.add(tf.keras.layers.BatchNormalization())

            if apply_dropout:
                result.add(tf.keras.layers.Dropout(0.5))

            result.add(tf.keras.layers.ReLU())

            return result

        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())

            result.add(tf.keras.layers.LeakyReLU())

            return result

        def Generator():
            inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3])

            down_stack = [
                downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
                downsample(128, 4),  # (batch_size, 64, 64, 128)
                downsample(256, 4),  # (batch_size, 32, 32, 256)
                downsample(512, 4),  # (batch_size, 16, 16, 512)
                downsample(512, 4),  # (batch_size, 8, 8, 512)
                downsample(512, 4),  # (batch_size, 4, 4, 512)
                downsample(512, 4),  # (batch_size, 2, 2, 512)
                downsample(512, 4),  # (batch_size, 1, 1, 512)
            ]

            up_stack = [
                upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
                upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
                upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
                upsample(512, 4),  # (batch_size, 16, 16, 1024)
                upsample(256, 4),  # (batch_size, 32, 32, 512)
                upsample(128, 4),  # (batch_size, 64, 64, 256)
                upsample(64, 4),  # (batch_size, 128, 128, 128)
            ]

            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    activation='tanh')  # (batch_size, 256, 256, 3)

            x = inputs

            # Downsampling through the model
            skips = []
            for down in down_stack:
                x = down(x)
                skips.append(x)

            skips = reversed(skips[:-1])

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])

            x = last(x)

            return tf.keras.Model(inputs=inputs, outputs=x)
        def Discriminator():
            initializer = tf.random_normal_initializer(0., 0.02)

            inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3], name='input_image')
            tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3], name='target_image')

            x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

            down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
            down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
            down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

            zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
            conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                            kernel_initializer=initializer,
                                            use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

            batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

            leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

            zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

            last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                            kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

            return tf.keras.Model(inputs=[inp, tar], outputs=last)

        if query == 'CLAVON' or query == 'CLEMENTI PEAKS':
            if query == 'CLAVON':
                siteselected = 'clavon'
                sitefilename = 'clavon'
            elif query == 'CLEMENTI PEAKS':
                siteselected = 'clementi peaks'
                sitefilename = 'clementi_peaks'

        # inference ----------------------------------------------


            checkpoint_dir = 'P2P/training_checkpoints'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            generator = Generator()
            discriminator = Discriminator()
            # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
            # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                            discriminator_optimizer=discriminator_optimizer,
                                            generator=generator,
                                            discriminator=discriminator)
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir ))


            # Predicting on the test dataset
            #change the path to the test dataset (clavon/clementi peaks)
            try:
                test_dataset_clavon = tf.data.Dataset.list_files(f'P2P/data_{sitefilename}/test/*')
            except tf.errors.InvalidArgumentError:
                # test_dataset = tf.data.Dataset.list_files(PATH + 'val/*.jpg')
                print('error')
            test_dataset_clavon = test_dataset_clavon.map(load_image_test)
            test_dataset_clavon = test_dataset_clavon.batch(BATCH_SIZE)

            for inp, tar in test_dataset_clavon .take(1): #only 1 image currently, but can increase
                pred = generate_images_test(generator, inp, output_path=f'P2P/data_{sitefilename}/prediction_{i}.png')

        #end of p2p ------------------------------------------------------------------------------------------------------------------------------------

            # plt.figure()

            # CV & Shapely -----------------------------------------------------------------------------------
            # Load the image
            image = cv2.imread(f'P2P/data_{sitefilename}/prediction_{i}.png')

            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Red color mask
            # [Red color range definitions as before]
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([170, 120, 70])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            # Yellow color mask
            lower_yellow = np.array([22, 93, 0])
            upper_yellow = np.array([45, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Find contours for red
            # [Contour finding for red as before]
            # Combine masks
            mask = mask1 + mask2

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(len(contours))

            # Assuming the largest 2 contours is the red polygon (buildings) ---------------------------------
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:2]
            red_polygon = []
            for contour in contours:
                epsilon = 0.020 * cv2.arcLength(contour, True) #higher thevalue, the smoother (0.02 is max)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Convert to Shapely Polygon
                polygon_points = [point[0] for point in approx]
                shapely_polygon = Polygon(polygon_points)
                red_polygon.append(shapely_polygon)
                X, Y = shapely_polygon.exterior.xy
                # Plotting
                # plt.plot(X, Y)
                # plt.fill(X, Y, alpha=0.3)  # Optional: fill the polygon with a semi-transparent color
            


            # Find contours for yellow (site)
            yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assuming the largest yellow contour is the target
            yellow_contour = max(yellow_contours, key=cv2.contourArea)
            yellow_contour = np.concatenate(yellow_contour, axis = 0)
            yellow_polygon = Polygon(yellow_contour)

            X, Y = yellow_polygon.exterior.xy
            # Plotting
            # plt.plot(X, Y)
            # plt.fill(X, Y, alpha=0.3)  # Optional: fill the polygon with a semi-transparent color


            x, y, w, h = cv2.boundingRect(yellow_contour)  # Bounding box for yellow

            yellow_bb = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]) 
            X, Y = yellow_bb.exterior.xy
            # Plotting
            # plt.plot(X, Y)
            # plt.fill(X, Y, alpha=0.3)  # Optional: fill the polygon with a semi-transparent color

            # Calculate the relative scale and position of the red polygon
            center_x, center_y = x + w/2, y + h/2
            relative_position = (np.mean([point[0] for point in polygon_points], axis=0) - [center_x, center_y])
            relative_scale = (len(polygon_points) / (w * h))

            # Smoothen the red polygon
            t = np.linspace(0, 1, len(polygon_points), endpoint=False)
            t_interp = np.linspace(0, 1, 500, endpoint=False)
            x, y = zip(*polygon_points)

            # Example Shapely polygons
            clavon_site = yellow_polygon
            # print("yellow",yellow_polygon)
            bounding_box_site = yellow_bb
            buildings = red_polygon
            # fig, ax = plt.subplots()
            # Convert to Lat Long coordinates
            sites_info = {'clavon': {'name': 'clavon',
                                    'site_boundary': Polygon(((103.76762137358979, 1.3087990670131122), (103.76695888021099, 1.3091033941901744), (103.76695888021099, 1.3091033941901782), (103.76680089609894, 1.3087280864870512), (103.766792747434, 1.3087078172631332), (103.76678557945053, 1.3086899861164933), (103.76677812454516, 1.3086684317627417), (103.76677122299097, 1.3086453681151176), (103.7667670091718, 1.308629112978794), (103.76676302875474, 1.308610442969295), (103.76675844145478, 1.308585861093), (103.76675564607066, 1.308568042468877), (103.7667533421893, 1.3085446293791023), (103.76675190226392, 1.30852567236893), (103.76675180617868, 1.3085231659641638), (103.76675217242752, 1.3085144072545394), (103.76675207742, 1.3084925885686336), (103.76675194337425, 1.3084860243757694), (103.76675179918865, 1.3084809478597594), (103.7667517106069, 1.3084758790108302), (103.7667516777277, 1.30847081910914), (103.76675170047343, 1.3084657425356094), (103.76675178278158, 1.3084603788857558), (103.76675190539342, 1.3084553122515075), (103.766759415393, 1.3084502477405577), (103.76675234045226, 1.3084448982895578), (103.76675262913469, 1.3084398468137872), (103.76675297264875, 1.30843479804762), (103.76675337164095, 1.308429760924249), (103.7667538265147, 1.3084247131985483), (103.76675436383049, 1.3084194076934133), (103.766754929798, 1.3084143621108462), (103.76675554278873, 1.3084093157891215), (103.76675623518352, 1.3084040881540013), (103.76675695186114, 1.3083991144012195), (103.76675772636361, 1.3083941199926599), (103.7667585981971, 1.3083888850082213), (103.76675947173706, 1.3083839153246697), (103.76676040667252, 1.3083789760982625), (103.76676143482773, 1.3083738029708472), (103.76676252125338, 1.3083686400989731), (103.76676359985571, 1.3083637364392524), (103.76676473694107, 1.3083588713631675), (103.76676598859025, 1.3083537327630061), (103.76676721542269, 1.3083489250335345), (103.7667685661309, 1.30834388081516), (103.76676998123311, 1.3083387781575588), (103.7667713777606, 1.3083339540246992), (103.76677288501628, 1.3083289535549412), (103.76677431495274, 1.3083243728196843), (103.76677594002432, 1.308319412844347), (103.7667776044232, 1.3083144450305626), (103.76677923565617, 1.3083097919766917), (103.76678109463339, 1.3083046375442602), (103.76678287287089, 1.30829985562966), (103.76678465129328, 1.3082952392626963), (103.76678653966118, 1.3082904938452273), (103.7667884690412, 1.3082857914938295), (103.76679050494603, 1.3082809810350113), (103.766792652022, 1.3082760417058086), (103.76679468914062, 1.3082714786919092), (103.76679669853525, 1.308267125485), (103.76679893802313, 1.308262412675837), (103.76680124155497, 1.3082576830310426), (103.76680345530123, 1.3082532600238392), (103.76680579233569, 1.3082487256194255), (103.76680831269016, 1.3082439502444518), (103.76681077662153, 1.308239388683786), (103.76681316405632, 1.3082350862344916), (103.76681367260619, 1.3082341941146542), (103.7670369584743, 1.307915310591265), (103.7670369584743, 1.3079153105912689), (103.76766580058731, 1.308298877127881), (103.76766580058731, 1.3082988771278814), (103.7676428514021, 1.30838189675668), (103.76764179810895, 1.3083863255490304), (103.76762282479181, 1.3084812464296733), (103.7676218261851, 1.3084883025243677), (103.76761371619476, 1.30858474385251), (103.76761349686825, 1.3085895323950083), (103.76761275665643, 1.3086863455250057), (103.76761291779121, 1.3086913506228472), (103.7676198824296, 1.308787898815235), (103.767645505, 1.3087927382936007), (103.76762137358979, 1.3087990670131138))),
                                    'site_coverage': 0.25,
                                    'building_scale': 18,
                                    'postal_code': "129962",
                                    "PR": 3.5,
                                    "URA_GFA": 62247.2,
                                    "URA_site_area": 16542.7,
                                    "URA_building_height": 140,
                                    "URA_dwelling_units": 640
                                    },
                        'clementi peaks': {'name': 'clementi peaks',
                                            'site_boundary': Polygon(((103.76881799558069, 1.3113251436959874), (103.76881140669404, 1.3113255727539448), (103.76873396113677, 1.3113315665393301), (103.76872784263504, 1.3113321102857496), (103.76865738056709, 1.3113391824336829), (103.76839126471891, 1.3113533462825622), (103.76838106854085, 1.3113540831328867), (103.76835900052754, 1.3113560993222997), (103.76833891785218, 1.3113586981033494), (103.76831699267106, 1.3113623775367729), (103.76830393468646, 1.3113649016606184), (103.76822916610874, 1.311381278147748), (103.76822326065329, 1.3113826413806544), (103.76814908496362, 1.3114006451884348), (103.76814328579329, 1.3114021211198539), (103.76806951434554, 1.3114217711079816), (103.76806369241301, 1.3114233919786933), (103.76805063357534, 1.3114271859332673), (103.76805063357534, 1.3114271859332676), (103.76793435831124, 1.3112195678874503), (103.76785570785277, 1.3110483549485854), (103.767854909063, 1.3110466321041037), (103.76783283523227, 1.3109994591068697), (103.76778345246646, 1.3108599918491781), (103.7677810570058, 1.310853493215834), (103.767775684187, 1.3108394750826222), (103.76777019997111, 1.310826231496659), (103.76776409052887, 1.3108125214049136), (103.76776114891919, 1.3108061449635922), (103.7677035103156, 1.3106853657516602), (103.7676989924413, 1.3106763377694812), (103.76768916217716, 1.3106575804661729), (103.7676792703816, 1.3106402915368032), (103.76766809354281, 1.3106223308305385), (103.76766267840111, 1.3106139769623915), (103.76760055230274, 1.3105219074519354), (103.767594802784, 1.3105107430616825), (103.767537054323, 1.3103985563959337), (103.767557039474, 1.310366533902596), (103.76751937486969, 1.3103642385678935), (103.76744440558485, 1.3102219736491236), (103.76743225026283, 1.310198905858367), (103.76743066661837, 1.3101959453563914), (103.76734150792808, 1.3100317376402688), (103.7673397693827, 1.3100285860524563), (103.76725723774882, 1.3098813090027575), (103.767004307234, 1.309318039510132), (103.767004307234, 1.3093180395101274), (103.76767692458766, 1.309016942764499), (103.76767692458766, 1.3090169427645042), (103.76769177935024, 1.30905584345123), (103.76769263097744, 1.309057957826232), (103.7677184530503, 1.3091188584835525), (103.76771935865, 1.30918791988285), (103.76797250724987, 1.3096639600470876), (103.76798035681244, 1.309681308153539), (103.76810629485455, 1.3099595184365), (103.7681429619824, 1.3100405173730216), (103.76818872560791, 1.31014162506384), (103.76824352221166, 1.3102626758022258), (103.768244409765, 1.310264537846306), (103.76830279528608, 1.3103830476364076), (103.76830295441698, 1.3103833682895258), (103.76834752882135, 1.3104725485665667), (103.76834849990372, 1.3104744128344818), (103.76839591388273, 1.31056183915754), (103.76839618181492, 1.3105623280126735), (103.76844411771513, 1.3106488769082296), (103.76844515425438, 1.3106506775505256), (103.76849661155047, 1.31073674686852), (103.76849738485257, 1.3107380064697685), (103.76855069357646, 1.3108225930541009), (103.76855129084, 1.3108235225460931), (103.76860488702482, 1.3109053421425187), (103.76860502290766, 1.3109055486933838), (103.76865875501163, 1.3109868763656687), (103.76870692933043, 1.3110702854992), (103.76875105891605, 1.3111566848241125), (103.76878968068795, 1.311244243504477), (103.76881799558069, 1.3113251436959852))),
                                            'site_coverage': 0.15,
                                            'building_scale': 12,
                                            "postal_code": "120463",
                                            "PR": 4,
                                            "URA_GFA": 144701.58,
                                            "URA_site_area": 35550,
                                            "URA_building_height": 137,
                                            "URA_dwelling_units": 1104
                                            }}

        # Scale and translate the p2p polygons based on actual site boundary ---------------------------------
            clavon_poly_coord = sites_info.get(siteselected)['site_boundary']
            x,y,w,h= clavon_poly_coord.bounds
            clavon_poly_bbx =Polygon([(x, y), (w, y), (w, h), (x, h)]) 
            # print('bound',clavon_poly_coord.bounds)
            scale_factor_x = (clavon_poly_bbx.bounds[2]-clavon_poly_bbx.bounds[0]) / (bounding_box_site.bounds[2]-bounding_box_site.bounds[0])  # Using width of bounding box
            scale_factor_y = (clavon_poly_bbx.bounds[3]-clavon_poly_bbx.bounds[1]) / (bounding_box_site.bounds[3]-bounding_box_site.bounds[1])  # Using height of bounding box

            scaled_clavon = scale(clavon_site, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))
            # ax.plot(scaled_clavon.exterior.xy[0],scaled_clavon.exterior.xy[1],color='red')
            # scaled_clavon = scale(clavon_site, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))

            scaled_polygon_B = scale(bounding_box_site, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))
            
            translation_vector = (clavon_poly_bbx.centroid.x - scaled_polygon_B.centroid.x, 
                                clavon_poly_bbx.centroid.y - scaled_polygon_B.centroid.y)
            transformed_clavon_poly = translate(scaled_clavon, xoff=translation_vector[0], yoff=translation_vector[1])

            transformed_clavon_poly_bbx = translate(scaled_polygon_B, xoff=translation_vector[0], yoff=translation_vector[1])
            # print(transformed_clavon_poly_bbx)
            # print(clavon_poly_bbx)

            polygon_coords ={}
            # ax.plot(transformed_clavon_poly_bbx.exterior.xy[0],transformed_clavon_poly_bbx.exterior.xy[1])
            # ax.plot(transformed_clavon_poly.exterior.xy[0],transformed_clavon_poly.exterior.xy[1])
            # ax.plot(clavon_poly_coord.exterior.xy[0],transformed_clavon_poly.exterior.xy[1])
            svg_buildings = list(clavon_poly_coord.exterior.coords)
            print("site:", clavon_poly_coord.exterior.coords)
            for j in range(len(buildings)):
                # inferences[f'model_{i}'] =[]
                poly = buildings[j]
                
                # print(poly)
                    #scale and translate building 
                scale_poly = scale(poly, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))
                translate_poly = translate(scale_poly, xoff=translation_vector[0], yoff=translation_vector[1])
                # ax.plot(translate_poly.exterior.xy[0],translate_poly.exterior.xy[1])
                polygon_coords[f'building{j+1}'] = list(translate_poly.exterior.coords)
                print("building:", clavon_poly_coord.exterior.coords)
                # inferences[f'model_{i}'].append(translate_poly)
                svg_buildings+= list(translate_poly.exterior.coords)

            # convert to SVG -------------------------------------------------------------
            latitudes = [lat for lat, lon in svg_buildings]
            longitudes = [lon for lat, lon in svg_buildings]

            min_lat, max_lat = min(latitudes), max(latitudes)
            min_lon, max_lon = min(longitudes), max(longitudes)
            # print(min_lat, max_lat,min_lon, max_lon)

            scale_x = svg_width / (max_lon - min_lon)
            scale_y = svg_height / (max_lat - min_lat)
            # write to svg
            count2=1
            for key, value in polygon_coords.items():
                translated_coords = [(lat - min_lat, lon - min_lon) for lat, lon in value]
                scaled_coords = [(lon * scale_x, lat * scale_y) for lat, lon in translated_coords]
                inferences[f'output{i+1}']['svg'][f'{key}'] = ' '.join([f'{x},{y}' for x, y in scaled_coords])
                count2+=1
            site_geom = clavon_poly_coord.exterior.coords
            translated_coords = [(lat - min_lat, lon - min_lon) for lat, lon in site_geom]
            scaled_coords = [(lon * scale_x, lat * scale_y) for lat, lon in translated_coords]
            inferences[f'output{i+1}']['svg']['site'] = ' '.join([f'{x},{y}' for x, y in scaled_coords])


            #create geojson from model. value
            geojson = {
                "type": "FeatureCollection"
                }
            list_coord = []
            area = 0
            for key, value in polygon_coords.items():
                # print(key)
                list_coord.append({
                    "type": "Feature",
                    "properties": { "type": "building" },
                    "geometry": {
                    "type": "Polygon",
                    "coordinates": [value]
                    }})
                building_polygon = Polygon(value)
                area += building_polygon.area
            # site_polygon = Polygon(site)
            site_area = clavon_poly_coord.area
            site_coverage = round(100*area/site_area,2)
            inferences[f'output{i+1}']['sitecoverage'] = site_coverage

            geojson['features'] = list_coord                
            inferences[f'output{i+1}']['geojson'] = geojson
            # # Prepare the data to be saved in JSON format
            # data = {
            #     "site": clavon_poly_coord, #clavon_site_geojson,
            #     # "bounding_box_site": transformed_clavon_poly_bbx, #bounding_box_site_geojson,
            #     "buildings":  buildings_output, #buildings_geojson list of 2
            # }
            # output["buildings"][f"output{i+1}"] = data #list of 3 models
    return inferences


# RL inference -----------------------------------------------------------------------------------------------------------------------
def RLinference(query,site,svg_height=100,svg_width=100):
    if query == 'CLAVON' or query == 'CLEMENTI PEAKS':
        if query == 'CLAVON':
            siteselected = 'clavon'
        elif query == 'CLEMENTI PEAKS':
            siteselected = 'clementi peaks'
        else:
            siteselected = None


    # inference
        building_list = [Polygon(((0.0, 0.0), (0.0, 1.1), (1.5, 1.1), (1.5, 2.0), (0.0, 2.0), (0.0, 3.0), (7.0, 3.0), (7.0, 2.0), (5.0, 2.0), (5.0, 1.1), (7.0, 1.1), (7.0, 0.0), (4.5, 0.0), (4.5, 0.5), (3.5, 0.5), (3.5, 0.0), (0.0, 0.0))),
                        #  Polygon(((0.0,0.0),(0.0,1.3),(1.5,1.3),(1.5,2.0),(0.0,2),(0.0,3),(5.0,3.0),(5.0,2.0),(3.0,2.0),(3.0,1.3),(5.0,1.3),(5.0,0.0),(0.0,0.0))),
                        ]
        sites_info = {'clavon': {'name': 'clavon',
                                'site_boundary': Polygon(((103.76762137358979, 1.3087990670131122), (103.76695888021099, 1.3091033941901744), (103.76695888021099, 1.3091033941901782), (103.76680089609894, 1.3087280864870512), (103.766792747434, 1.3087078172631332), (103.76678557945053, 1.3086899861164933), (103.76677812454516, 1.3086684317627417), (103.76677122299097, 1.3086453681151176), (103.7667670091718, 1.308629112978794), (103.76676302875474, 1.308610442969295), (103.76675844145478, 1.308585861093), (103.76675564607066, 1.308568042468877), (103.7667533421893, 1.3085446293791023), (103.76675190226392, 1.30852567236893), (103.76675180617868, 1.3085231659641638), (103.76675217242752, 1.3085144072545394), (103.76675207742, 1.3084925885686336), (103.76675194337425, 1.3084860243757694), (103.76675179918865, 1.3084809478597594), (103.7667517106069, 1.3084758790108302), (103.7667516777277, 1.30847081910914), (103.76675170047343, 1.3084657425356094), (103.76675178278158, 1.3084603788857558), (103.76675190539342, 1.3084553122515075), (103.766759415393, 1.3084502477405577), (103.76675234045226, 1.3084448982895578), (103.76675262913469, 1.3084398468137872), (103.76675297264875, 1.30843479804762), (103.76675337164095, 1.308429760924249), (103.7667538265147, 1.3084247131985483), (103.76675436383049, 1.3084194076934133), (103.766754929798, 1.3084143621108462), (103.76675554278873, 1.3084093157891215), (103.76675623518352, 1.3084040881540013), (103.76675695186114, 1.3083991144012195), (103.76675772636361, 1.3083941199926599), (103.7667585981971, 1.3083888850082213), (103.76675947173706, 1.3083839153246697), (103.76676040667252, 1.3083789760982625), (103.76676143482773, 1.3083738029708472), (103.76676252125338, 1.3083686400989731), (103.76676359985571, 1.3083637364392524), (103.76676473694107, 1.3083588713631675), (103.76676598859025, 1.3083537327630061), (103.76676721542269, 1.3083489250335345), (103.7667685661309, 1.30834388081516), (103.76676998123311, 1.3083387781575588), (103.7667713777606, 1.3083339540246992), (103.76677288501628, 1.3083289535549412), (103.76677431495274, 1.3083243728196843), (103.76677594002432, 1.308319412844347), (103.7667776044232, 1.3083144450305626), (103.76677923565617, 1.3083097919766917), (103.76678109463339, 1.3083046375442602), (103.76678287287089, 1.30829985562966), (103.76678465129328, 1.3082952392626963), (103.76678653966118, 1.3082904938452273), (103.7667884690412, 1.3082857914938295), (103.76679050494603, 1.3082809810350113), (103.766792652022, 1.3082760417058086), (103.76679468914062, 1.3082714786919092), (103.76679669853525, 1.308267125485), (103.76679893802313, 1.308262412675837), (103.76680124155497, 1.3082576830310426), (103.76680345530123, 1.3082532600238392), (103.76680579233569, 1.3082487256194255), (103.76680831269016, 1.3082439502444518), (103.76681077662153, 1.308239388683786), (103.76681316405632, 1.3082350862344916), (103.76681367260619, 1.3082341941146542), (103.7670369584743, 1.307915310591265), (103.7670369584743, 1.3079153105912689), (103.76766580058731, 1.308298877127881), (103.76766580058731, 1.3082988771278814), (103.7676428514021, 1.30838189675668), (103.76764179810895, 1.3083863255490304), (103.76762282479181, 1.3084812464296733), (103.7676218261851, 1.3084883025243677), (103.76761371619476, 1.30858474385251), (103.76761349686825, 1.3085895323950083), (103.76761275665643, 1.3086863455250057), (103.76761291779121, 1.3086913506228472), (103.7676198824296, 1.308787898815235), (103.767645505, 1.3087927382936007), (103.76762137358979, 1.3087990670131138))),
                                'site_coverage': 0.25,
                                'building_scale': 18,
                                'postal_code': "129962",
                                "PR": 3.5,
                                "URA_GFA": 62247.2,
                                "URA_site_area": 16542.7,
                                "URA_building_height": 140,
                                "URA_dwelling_units": 640
                                },
                    'clementi peaks': {'name': 'clementi peaks',
                                        'site_boundary': Polygon(((103.76881799558069, 1.3113251436959874), (103.76881140669404, 1.3113255727539448), (103.76873396113677, 1.3113315665393301), (103.76872784263504, 1.3113321102857496), (103.76865738056709, 1.3113391824336829), (103.76839126471891, 1.3113533462825622), (103.76838106854085, 1.3113540831328867), (103.76835900052754, 1.3113560993222997), (103.76833891785218, 1.3113586981033494), (103.76831699267106, 1.3113623775367729), (103.76830393468646, 1.3113649016606184), (103.76822916610874, 1.311381278147748), (103.76822326065329, 1.3113826413806544), (103.76814908496362, 1.3114006451884348), (103.76814328579329, 1.3114021211198539), (103.76806951434554, 1.3114217711079816), (103.76806369241301, 1.3114233919786933), (103.76805063357534, 1.3114271859332673), (103.76805063357534, 1.3114271859332676), (103.76793435831124, 1.3112195678874503), (103.76785570785277, 1.3110483549485854), (103.767854909063, 1.3110466321041037), (103.76783283523227, 1.3109994591068697), (103.76778345246646, 1.3108599918491781), (103.7677810570058, 1.310853493215834), (103.767775684187, 1.3108394750826222), (103.76777019997111, 1.310826231496659), (103.76776409052887, 1.3108125214049136), (103.76776114891919, 1.3108061449635922), (103.7677035103156, 1.3106853657516602), (103.7676989924413, 1.3106763377694812), (103.76768916217716, 1.3106575804661729), (103.7676792703816, 1.3106402915368032), (103.76766809354281, 1.3106223308305385), (103.76766267840111, 1.3106139769623915), (103.76760055230274, 1.3105219074519354), (103.767594802784, 1.3105107430616825), (103.767537054323, 1.3103985563959337), (103.767557039474, 1.310366533902596), (103.76751937486969, 1.3103642385678935), (103.76744440558485, 1.3102219736491236), (103.76743225026283, 1.310198905858367), (103.76743066661837, 1.3101959453563914), (103.76734150792808, 1.3100317376402688), (103.7673397693827, 1.3100285860524563), (103.76725723774882, 1.3098813090027575), (103.767004307234, 1.309318039510132), (103.767004307234, 1.3093180395101274), (103.76767692458766, 1.309016942764499), (103.76767692458766, 1.3090169427645042), (103.76769177935024, 1.30905584345123), (103.76769263097744, 1.309057957826232), (103.7677184530503, 1.3091188584835525), (103.76771935865, 1.30918791988285), (103.76797250724987, 1.3096639600470876), (103.76798035681244, 1.309681308153539), (103.76810629485455, 1.3099595184365), (103.7681429619824, 1.3100405173730216), (103.76818872560791, 1.31014162506384), (103.76824352221166, 1.3102626758022258), (103.768244409765, 1.310264537846306), (103.76830279528608, 1.3103830476364076), (103.76830295441698, 1.3103833682895258), (103.76834752882135, 1.3104725485665667), (103.76834849990372, 1.3104744128344818), (103.76839591388273, 1.31056183915754), (103.76839618181492, 1.3105623280126735), (103.76844411771513, 1.3106488769082296), (103.76844515425438, 1.3106506775505256), (103.76849661155047, 1.31073674686852), (103.76849738485257, 1.3107380064697685), (103.76855069357646, 1.3108225930541009), (103.76855129084, 1.3108235225460931), (103.76860488702482, 1.3109053421425187), (103.76860502290766, 1.3109055486933838), (103.76865875501163, 1.3109868763656687), (103.76870692933043, 1.3110702854992), (103.76875105891605, 1.3111566848241125), (103.76878968068795, 1.311244243504477), (103.76881799558069, 1.3113251436959852))),
                                        'site_coverage': 0.15,
                                        'building_scale': 12,
                                        "postal_code": "120463",
                                        "PR": 4,
                                        "URA_GFA": 144701.58,
                                        "URA_site_area": 35550,
                                        "URA_building_height": 137,
                                        "URA_dwelling_units": 1104
                                        }
                    }

        # load model
        # load_path = 'RL/best_model_3_w_interblock_distance_ppo96.zip'
        load_path = 'RL/best_model_3_w_interblock_distance_ppo97.zip'
        if os.path.exists(load_path):
            model = PPO.load(load_path)
        else:
            print(f"The file {load_path} does not exist.")

        assert model is not None, "Model not found"


        top_models = []

        env = SpaDesPlacement(sites_info, building_list[0],name=siteselected)
        best_obs = ''
        best_reward = -float('inf')
        best_rewards = []
        best_boxes = []
        episode =0

        # while valid and episode <1000:
        for i in range(5000):
            obs,info = env.reset(seed =1 , name = siteselected)
            episode_reward = 0
            episode +=1
            rewards = []
            while True:
                action, _ = model.predict(obs)
                obs, reward, done, _,_ = env.step(action)
                episode_reward +=reward
                rewards.append(reward)
            
                if done :
                    current_model_info = (episode_reward, obs, env.boxes, rewards)
                    top_models.append(current_model_info)
                    top_models.sort(key=lambda x: x[0], reverse=True)
                    if len(top_models) > 3:
                        top_models = top_models[:3]
                    break
                        
        model_buildings =[]
        for model in top_models:
            best_boxes = model[2] 
            best_rewards = model[3]
            best_obs = model[1]
            best_reward = model[0]
            boxes = best_boxes[:1]
            rewards = []
            buildings_poly =[]
            for i in range(1,len(best_boxes)):
                X,Y = env._resize_polygon(env.building, env.building_scale, (best_boxes[i][0], best_boxes[i][1]), best_boxes[i][3])
                poly = Polygon(zip(X,Y))
                if env._building_outside_boundary(poly):
                    pass
                else:
                    boxes = np.append(boxes,np.array([best_boxes[i]]),axis=0)
                    rewards.append(best_rewards[i-1])
                    buildings_poly.append(poly)
            model_buildings.append(buildings_poly)
            best_reward = sum(rewards)
            obs_1 = np.reshape(best_obs, (50,50))

        inferences={}
        clavon_poly_coord = sites_info.get(siteselected)['site_boundary']
        clavon_poly_coord_centroid = clavon_poly_coord.centroid
        clavon_poly_grid = env.site_boundary
        # print('bound',clavon_poly_coord.bounds)
        scale_factor_x = (clavon_poly_coord.bounds[2]-clavon_poly_coord.bounds[0]) / (clavon_poly_grid.bounds[2]-clavon_poly_grid.bounds[0])  # Using width of bounding box
        scale_factor_y = (clavon_poly_coord.bounds[3]-clavon_poly_coord.bounds[1]) / (clavon_poly_grid.bounds[3]-clavon_poly_grid.bounds[1])  # Using height of bounding box

        # Scale Polygon B
        scaled_polygon_B = scale(clavon_poly_grid, xfact=-scale_factor_x, yfact=-scale_factor_y, origin=(clavon_poly_coord_centroid))

        # Calculate translation vector
        translation_vector = (clavon_poly_coord.centroid.x - scaled_polygon_B.centroid.x, 
                            clavon_poly_coord.centroid.y - scaled_polygon_B.centroid.y)

        # Translate scaled Polygon B
        transformed_clavon_poly_grid = translate(scaled_polygon_B, xoff=translation_vector[0], yoff=translation_vector[1])
        # print(clavon_poly_coord)
        # print(transformed_clavon_poly_grid)
    
        for i in range(len(model_buildings)):
            # inferences[f'building{i+1}'] = []
            inferences[f'output{i+1}'] = {"svg":{},"geojson":{}}
            coord_list = []
            # svg_coord = {}
            polygon_coords = {}
            count = 1
            for poly in model_buildings[i]:
                polygon_list = []
                #scale and translate building 
                scale_poly = scale(poly, xfact=-scale_factor_x, yfact=-scale_factor_y, origin=(clavon_poly_coord_centroid))
                translate_poly = translate(scale_poly, xoff=translation_vector[0], yoff=translation_vector[1])
                
                for x,y in translate_poly.exterior.coords:
                    polygon_list.append([x,y])
                #add polygon coords for geojson
                polygon_coords[f'building{count}'] = polygon_list
                #add polygon coords for svg
                # svg_coord[count] = [LonLat_To_XY(x,y) for x,y in polygon_coords[f'building{count}']]
                # svg_coord[count] = [x,y for x,y in polygon_coords[f'building{count}']]
                #add for svg conversion and scaling
                coord_list+=polygon_list
                count+=1 
            coord_list+=site #add site

            # convert to SVG -------------------------------------------------------------
            latitudes = [lat for lat, lon in coord_list]
            longitudes = [lon for lat, lon in coord_list]

            min_lat, max_lat = min(latitudes), max(latitudes)
            min_lon, max_lon = min(longitudes), max(longitudes)
            # print(min_lat, max_lat,min_lon, max_lon)

            scale_x = svg_width / (max_lon - min_lon)
            scale_y = svg_height / (max_lat - min_lat)
            # print(scale_x,scale_y)
            # output = {}
            count2 = 1
            #for buildings
            for key, value in polygon_coords.items():
                translated_coords = [(lat - min_lat, lon - min_lon) for lat, lon in value]
                scaled_coords = [(lon * scale_x, lat * scale_y) for lat, lon in translated_coords]
                inferences[f'output{i+1}']['svg'][f'{key}'] = ' '.join([f'{x},{y}' for x, y in scaled_coords])
                count2+=1
            #for site
            site_geom = site
            translated_coords = [(lat - min_lat, lon - min_lon) for lat, lon in site_geom]
            scaled_coords = [(lon * scale_x, lat * scale_y) for lat, lon in translated_coords]
            inferences[f'output{i+1}']['svg']['site'] = ' '.join([f'{x},{y}' for x, y in scaled_coords])
            
            #create geojson from model. value
            geojson = {
                "type": "FeatureCollection"
                }
            list_coord = []
            area = 0
            for key, value in polygon_coords.items():
                # print(key)
                list_coord.append({
                    "type": "Feature",
                    "properties": { "type": "building" },
                    "geometry": {
                    "type": "Polygon",
                    "coordinates": [value]
                    }})
                building_polygon = Polygon(value)
                area += building_polygon.area
            site_polygon = Polygon(site)
            site_area = site_polygon.area
            site_coverage = round(100*area/site_area,2)
            inferences[f'output{i+1}']['sitecoverage'] = site_coverage

            geojson['features'] = list_coord                
            inferences[f'output{i+1}']['geojson'] = geojson
        # print(inferences)
    else:
        inferences = None
    return inferences

def model_inference(setback_data,query,model='OP'):
    # Adding buildings & SVG -------------------------------------------------------------------------------------------------
    if model == 'RL': # Reinforcement Learning, one output
        emit_log(f"Generating Buildings..")  # Emit log message
        buildings = RLinference(query,setback_data['geometry'][0])
        # output = buildings_to_SVG(buildings,setback_data['geometry'][0])
        setback_data['buildings'] = buildings
    elif model == 'P2P':
        emit_log(f"Generating Buildings..")
        buildings = p2p_inference(query)
        setback_data['buildings'] = buildings
    elif model == 'OP':
        emit_log(f"Generating Buildings..")  # Emit log message
        if query == "CLEMENTI PEAKS":
            setback_data["buildings"] = {"output1": {}, "output2": {}, "output3": {}}
            #read the svg file
            for i in range(1,4):
                with open(f'svg/clementi{i}svg.json', 'r',encoding='utf-8') as f:
                    setback_data["buildings"][f"output{i}"]["svg"] = json.load(f)
                with open(f'svg/clementi{i}.geojson', 'r') as f:
                    setback_data["buildings"][f"output{i}"]["geojson"] = json.load(f)
                    json_data = setback_data["buildings"][f"output{i}"]["geojson"]
                    area = 0
                    for feature in json_data['features']:
                        if feature["properties"]["type"] == "building":
                            geom = shape(feature['geometry'])
                            area += geom.area

                    site_polygon = Polygon(setback_data['geometry'][0])
                    # ax.plot(*peaks_polygon.exterior.xy, color='black')
                    site_coverage = round(100*area/site_polygon.area,2)
                    print("Site Coverage:" , site_coverage)
                    setback_data["buildings"][f"output{i}"]["sitecoverage"] = site_coverage
            
        elif query == "CLAVON":
            clavon_site = None
            with open('svg/clavon.geojson', 'r') as f:
                clavon_site = json.load(f)
                # print("read ",clavon_site)
            setback_data["buildings"] = {"output1": {}, "output2": {}, "output3": {}}
            for i in range(1,4):
                with open(f'svg/clavon{i}svg.json', 'r',encoding='utf-8') as f:
                    setback_data["buildings"][f"output{i}"]["svg"] = json.load(f)
                with open(f'svg/clavon{i}.geojson', 'r') as f:
                    setback_data["buildings"][f"output{i}"]["geojson"] = json.load(f)
                    json_data = setback_data["buildings"][f"output{i}"]["geojson"]
                    area = 0
                    for feature in json_data['features']:
                        if feature["properties"]["type"] == "building":
                            geom = shape(feature['geometry'])
                            area += geom.area

                    site_polygon = shape(clavon_site['features'][0]['geometry'])
                    print("site poly",site_polygon)
                    # ax.plot(*peaks_polygon.exterior.xy, color='black')
                    site_coverage = round(100*area/site_polygon.area,2)
                    print("Site Coverage:" , site_coverage)
                    setback_data["buildings"][f"output{i}"]["sitecoverage"] = site_coverage
                
            # #read the svg file
            # with open('svg/clavon1svg.json', 'r',encoding='utf-8') as f:
            #     setback_data["buildings"]["output1"]["svg"] = json.load(f)
            # with open('svg/clavon1.geojson', 'r') as f:
            #     setback_data["buildings"]["output1"]["geojson"] = json.load(f)

            # with open('svg/clavon2svg.json', 'r',encoding='utf-8') as f:
            #     setback_data["buildings"]["output2"]["svg"] = json.load(f)
            # with open('svg/clavon2.geojson', 'r') as f:
            #     setback_data["buildings"]["output2"]["geojson"] = json.load(f)

            # with open('svg/clavon3svg.json', 'r',encoding='utf-8') as f:
            #     setback_data["buildings"]["output3"]["svg"] = json.load(f)
            # with open('svg/clavon3.geojson', 'r') as f:
            #     setback_data["buildings"]["output3"]["geojson"] = json.load(f)
        emit_log(f"Generation completed!")  # Emit log message
    else:
        print('No model type')
    return setback_data


# run the code ----------------------------------------------------------------------------------------------------------
def main(searchVal,StoreyHeight=[30,5],model=None):
    with open("json/Setback.json", "r", encoding="utf-8") as f:
        # json_string_setbacks = json.load(f)
        json_string_setbacks = f.read()

    # setbacks = pd.json_normalize(json.loads(json_string_setbacks))
    setbacks = pd.read_json(json_string_setbacks, orient="split")
    input_dict = URAMapSelenium(searchVal)
    output_dict = setSetbacks(input_dict, StoreyHeight, setbacks)
    output_dict = model_inference(output_dict,output_dict["name"],model) #kiv
    # print(output_dict)

    #extract static img from mapbox
    # encoded_image = getMapboxStaticImg(output_dict['centroid'],access_token)
    
    with open("json/Output.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f)

    response_data = {
        # 'image': encoded_image,
        'json': output_dict
    }

    # package as a response
    response = jsonify(response_data)
    # save output to json
    emit_log("Package received!")
    return response


# run flask app ----------------------------------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def emit_log(message):
    """ Emit log messages to the frontend """
    print(message)  # Keep the print statement if you want to see logs in the server terminal too
    socketio.emit('log', {'data': message})

@app.route('/query/<searchVal>')
def index(searchVal):
    print(f"Received value: {searchVal}")
    try:
        emit_log(f"Received query: {searchVal}")  # Emit log message
        response = main(escape(searchVal),StoreyHeight=[30,5],model=None)
        return response
    except Exception as e:
        print(f"Error: {e}")
        emit_log(f"Error: {e}")  # Emit error message
        return {"error": str(e)}, 500
    
# ... other routes ...
@app.route('/query/<searchVal>/<model>') #array is a string of comma separated values (for multiple storey heights)
def index2(searchVal,model):
    modeltype = None
    if model == 'RL':
        modeltype = 'RL'
    # Split the array string into a Python list
    # array_list = array.split(',')
    elif model == 'OP':
        modeltype = 'OP'
    elif model == 'PP':
        modeltype = 'P2P'
    # Convert to the appropriate data type if necessary (e.g., to integers)
    # array_list = [int(element) for element in array_list]
    # print(f"Received value: {searchVal} and {array_list}")
    try:
        emit_log(f"Received query: {searchVal}")  # Emit log message
        response = main(escape(searchVal),model=modeltype)
        return response
    except Exception as e:
        print(f"Error: {e}")
        emit_log(f"Error: {e}")  # Emit error message
        return {"error": str(e)}, 500


# Replace `app.run(debug=True)` with the following:
if __name__ == '__main__':
    socketio.run(app, debug=True)
    # main("Clementi Peaks", StoreyHeight=[30,5])
    # main("clavon")
