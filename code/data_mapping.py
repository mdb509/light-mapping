import os
import gpxpy
import pandas as pd
import bisect


class DataMapper():
    """ handels data mapping and parsing of data from gpx and biolog data"""
    
    def save_as_csv(self, data: list, filename: str, record=1):
        """Convert to DataFrame and save to CSV"""
        df = pd.DataFrame(data)
        df.to_csv("code/data_output/"+filename, index=False)
        print(f"CSV saved as {filename}")

    def gpx_to_csv(self, gpx_file):
        """ Parse gpx file and converts it to csv and saves it"""
        # Parse GPX file
        with open(gpx_file, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        # Extract data points
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'timestamp': point.time.timestamp() if point.time else None,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation
                    })

        self.save_as_csv(data, "output_gpx.csv")

    
    def biolog_to_csv(self, biolog_file):
        """ Parse biolog file and converts it to csv and saves it"""
        
        # separte records
        record = 1
        data = []

        # Parse Biolog file
        with open(biolog_file, 'r') as file:
            for line in file:
                line = ' '.join(line.split())
                point = line.split(' ')
                if "//" in point:
                    if "Autorange" in point and data != []:
                        self.save_as_csv(data, f"output_biolog_{record}.csv")
                        record += 1
                        data = []
                    continue
            
                # example for one data point
                # point: ['ATbio04', '1746527033', '6.5.2025', '10:23', '3860', '23.25', '121437', '2']
                data.append({
                    'timestamp': point[1],
                    'date': point[2],
                    'time': point[3],
                    'battery_voltage_mV': point[4],
                    'temperature_celsius': point[5],
                    'measurement_µV': point[6],
                    'gain_factor': point[7]
                })
        self.save_as_csv(data, f"output_biolog_{record}.csv")


    def map_bio_to_gpx(self, gpx_filename, biolog_filename):
        """ map gpx and biolog data using unix timestaps"""
        # Read csv for bio and gpx
        df_gpx = pd.read_csv(gpx_filename)
        df_biolog = pd.read_csv(biolog_filename)

        # Ensure GPS/Bio timestamps are sorted (if not already sorted)
        df_gpx = df_gpx.sort_values(by='timestamp')
        df_biolog = df_biolog.sort_values(by='timestamp')
        
        # GPS-Timestamps
        gpx_timestamps = df_gpx['timestamp'].values
        
        # store mapped points here
        mapped_points = []
        
        # count all matched points for ratio calculation
        matched_points = 0
        all_points = len(df_biolog['timestamp'])

        for _, biolog in df_biolog.iterrows():
            
            # Find the position of the biolog timestamp in the GPS timestamps
            # TODO find correct offset, sensor unix timestamp is off by 2h
            corrected_time = biolog['timestamp'] - 7210  
            pos = bisect.bisect_left(gpx_timestamps, corrected_time)
            
            # Previous GPS timestamp (if available), else skip this datapoint
            if pos > 0:
                prev_gps = df_gpx.iloc[pos - 1]
            else:
                continue
            
            # Next GPS timestamp (if available), else skip this datapoint
            if pos < len(gpx_timestamps):
                next_gps = df_gpx.iloc[pos]
            else:
                continue
            
            # check if target time is between bove gps times
            t1 = prev_gps.loc['timestamp']
            t2 = next_gps.loc['timestamp']
            target_time  = corrected_time

            if not (t1 <= target_time <= t2):
                raise ValueError("target_time liegt nicht zwischen den beiden GPS-Punkten")
            
            matched_points += 1

            # Ratio between target timestamp and above GPS timestamps
            # Calculate actuall gps point of measurment point
            ratio = (target_time - t1) / (t2 - t1)
            
            # Interpolate latitude, longitude, and elevation
            prev_lat = prev_gps.loc['latitude']
            next_lat = next_gps.loc['latitude']
            prev_lon = prev_gps.loc['longitude']
            next_lon = next_gps.loc['longitude']

            lat = prev_lat + (next_lat - prev_lat) * ratio
            lat = round(float(lat), 7)

            lon = prev_lon + (next_lon - prev_lon) * ratio
            lon = round(float(lon), 7)

            ele = None
            if prev_gps.loc['elevation'] is not None and next_gps.loc['elevation'] is not None:
                ele = prev_gps.loc['elevation'] + (next_gps.loc['elevation'] - prev_gps.loc['elevation']) * ratio
                ele = round(float(ele),7)
            
            # Create a new data point with all values from biolog but updated latitude, longitude, and elevation
            new_data_point = biolog.to_dict()
            new_data_point.update({
                'latitude': lat,
                'longitude': lon,
                'elevation': ele
            })
            # Append new datapoint
            mapped_points.append(new_data_point)

        # State ratio of how many datapoints from BIOLOG.TXT are mapped
        print(f"Matched {matched_points/all_points*100}%")
        print(f"match {matched_points}")
        print(f"all {all_points}")
        df_mapped = pd.DataFrame(mapped_points)
        self.save_as_csv(df_mapped, "output.csv")
        
if __name__ == '__main__':
    d = DataMapper()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    d.gpx_to_csv(current_folder +"/data_input/Wanderung-2.gpx")
    d.biolog_to_csv(current_folder + "/data_input/BIOLOG.TXT")
    d.map_bio_to_gpx(current_folder + "/data_output/output_gpx.csv", current_folder + "/data_output/output_biolog_6.csv")

