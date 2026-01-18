
from pyproj import Transformer
import math
from typing import Any, List

class GridMapperWebMercator:
                            
    def __init__(self, lon_min:float, lat_min:float, 
                        lon_max:float, lat_max:float, grid_size_m:int)->None:
        if lat_max <= lat_min or lon_max <= lon_min:
            raise ValueError("The coordinates of the top right corner must be \
                             greater than the coordinates of the bottom left corner.")
        if grid_size_m < 0:
            raise ValueError("The grid spacing must be a positive number.")

        self.lat_min, self.lon_min = lat_min, lon_min
        self.lat_max, self.lon_max = lat_max, lon_max
        self.grid_size_m = grid_size_m

                                        
        self.transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.transformer_to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

                
        self.x_min, self.y_min = self.transformer_to_utm.transform(lon_min, lat_min)
        self.x_max, self.y_max = self.transformer_to_utm.transform(lon_max, lat_max)

                 
        self.num_cols = math.ceil((self.x_max - self.x_min) / grid_size_m)
        self.num_rows = math.ceil((self.y_max - self.y_min) / grid_size_m)
        self.total_grids = self.num_cols * self.num_rows

    def latlon_to_grid(self, lon:float, lat:float)->int:

        x, y = self.transformer_to_utm.transform(lon, lat)

        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            raise ValueError("The input latitude and longitude are outside the initial area range.")

        col = int((x - self.x_min) // self.grid_size_m)
        row = int((y - self.y_min) // self.grid_size_m)
        grid_id = row * self.num_cols + col + 1          
        return grid_id

    def grid_to_latlon(self, grid_id:int):

        if not (1 <= grid_id <= self.total_grids):
            raise ValueError(f"The input raster ID is out of range.(1 ~ {self.total_grids})")

        grid_id -= 1             
        row = grid_id // self.num_cols
        col = grid_id % self.num_cols

        x_center = self.x_min + (col + 0.5) * self.grid_size_m
        y_center = self.y_min + (row + 0.5) * self.grid_size_m

        lon_center, lat_center = self.transformer_to_geo.transform(x_center, y_center)
        return lon_center, lat_center

class GridMapperUTM:

    def __init__(self, min_lon, min_lat, max_lon, max_lat, cell_size_m=1000):
        if max_lon <= min_lon or max_lat <= min_lat:
            raise ValueError("Invalid latitude and longitude range")
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")

        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.cell_size_m = cell_size_m

                               
        center_lon = (min_lon + max_lon) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        is_northern = (min_lat + max_lat) / 2 >= 0             
        epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone

                        
        self.to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        self.to_lonlat = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

                       
        self.x_min, self.y_min = self.to_utm.transform(min_lon, min_lat)
        self.x_max, self.y_max = self.to_utm.transform(max_lon, max_lat)

                
        self.num_x = math.ceil((self.x_max - self.x_min) / cell_size_m)
        self.num_y = math.ceil((self.y_max - self.y_min) / cell_size_m)

    def lonlat_to_grid(self, lon, lat):
        x, y = self.to_utm.transform(lon, lat)
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            raise ValueError("over range")

        gx = int((x - self.x_min) // self.cell_size_m)
        gy = int((y - self.y_min) // self.cell_size_m)
        return gx * self.num_y + gy

    def grid_to_lonlat(self, grid_id):
        total_grids = self.num_x * self.num_y
        if not (0 <= grid_id < total_grids):
            raise ValueError(f"grid_id over range (0 ~ {total_grids-1})")

        gx = grid_id // self.num_y
        gy = grid_id % self.num_y

                    
        x = self.x_min + (gx + 0.5) * self.cell_size_m
        y = self.y_min + (gy + 0.5) * self.cell_size_m

               
        lon, lat = self.to_lonlat.transform(x, y)
        return lon, lat
    

import math

             
WGS84_A = 6378137.0                          
WGS84_F = 1.0 / 298.257223563         

def m_per_deg_at_lat(lat_deg, a=WGS84_A, f=WGS84_F):
                        
    phi = math.radians(lat_deg)
    e2 = 2*f - f*f       
    sinphi = math.sin(phi)
    cosphi = math.cos(phi)

                         
    m_per_deg_lat = (math.pi / 180.0) * (a * (1 - e2)) / ((1 - e2 * sinphi * sinphi) ** 1.5)

                         
    m_per_deg_lon = (math.pi / 180.0) * (a * cosphi) / math.sqrt(1 - e2 * sinphi * sinphi)

    return m_per_deg_lat, m_per_deg_lon

class GridMapperEllipsoid:
    def __init__(self, min_lon, min_lat, max_lon, max_lat, cell_size_m):
        if max_lon <= min_lon or max_lat <= min_lat:
            raise ValueError("The latitude and longitude range is illegal")
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")

        self.min_lon = float(min_lon)
        self.min_lat = float(min_lat)
        self.max_lon = float(max_lon)
        self.max_lat = float(max_lat)
        self.cell_size_m = float(cell_size_m)

                                  
        self.center_lat = (self.min_lat + self.max_lat) / 2.0
        mlat, mlon = m_per_deg_at_lat(self.center_lat)
        self.m_per_deg_lat = mlat
        self.m_per_deg_lon = mlon

                
        self.cell_deg_lat = self.cell_size_m / self.m_per_deg_lat
        self.cell_deg_lon = self.cell_size_m / self.m_per_deg_lon

               
        self.num_x = int(math.ceil((self.max_lon - self.min_lon) / self.cell_deg_lon))
        self.num_y = int(math.ceil((self.max_lat - self.min_lat) / self.cell_deg_lat))
        self.total = self.num_x * self.num_y

    def lonlat_to_grid(self, lon, lat):
        lon = float(lon); lat = float(lat)
        if not (self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat):
            raise ValueError("The latitude and longitude range is illegal")

        gx = int((lon - self.min_lon) // self.cell_deg_lon)
        gy = int((lat - self.min_lat) // self.cell_deg_lat)

              
        if gx >= self.num_x:
            gx = self.num_x - 1
        if gy >= self.num_y:
            gy = self.num_y - 1

        return gx * self.num_y + gy

    def grid_to_lonlat(self, grid_id):
        if not (0 <= grid_id < self.total):
            raise ValueError(f"grid_id 超出范围 (0 ~ {self.total-1})")
        gx = grid_id // self.num_y
        gy = grid_id % self.num_y

        lon_center = self.min_lon + (gx + 0.5) * self.cell_deg_lon
        lat_center = self.min_lat + (gy + 0.5) * self.cell_deg_lat
        return lon_center, lat_center

    def info(self):
        return {
            "center_lat": self.center_lat,
            "m_per_deg_lat": self.m_per_deg_lat,
            "m_per_deg_lon": self.m_per_deg_lon,
            "cell_deg_lat": self.cell_deg_lat,
            "cell_deg_lon": self.cell_deg_lon,
            "num_x": self.num_x,
            "num_y": self.num_y,
            "total": self.total
        }