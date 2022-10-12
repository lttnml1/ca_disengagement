#!/usr/bin/env python

"""
    This is the Base class from which all scenarios are derived
"""

from abc import abstractmethod
import carla
from shapely.geometry import Polygon
from shapely.affinity import rotate
import numpy as np
import math
import os
import time
import pathlib
import subprocess

class Scenario(object):
    def __init__(self, host=None, port=None, args=None):
        self.host = host
        self.port = port
        self.feature_vector = []
        self.no_render = args.no_render
        self.score = None
        self.dataframe = None
        self.badPath = False
        self.client = None
        self.world = None
        self.file_name = None
        self.data_path = None

    def world_setup(self):
        try:
            
            self.client = carla.Client(self.host,self.port)
            self.client.set_timeout(4.0)
            #self.world = self.client.load_world('Town03')
            self.world = self.client.get_world()
            
        except RuntimeError:
            print(f"Could not connect to client: {self.host}:{self.port}")
            start_carla = ["C:/CARLA_0.9.13/WindowsNoEditor/CarlaUE4.exe",f"--carla-world-port={self.port}"]
            check_carla = ["C:/Windows/System32\WindowsPowerShell/v1.0/powershell.exe", f"Get-Process -Id (Get-NetTCPConnection -LocalPort {self.port}).OwningProcess"]
            kill_carla = ["C:/Windows/System32\WindowsPowerShell/v1.0/powershell.exe", f"Get-Process -Id (Get-NetTCPConnection -LocalPort {self.port}).OwningProcess | kill"]
            check_carla_process = subprocess.run(check_carla, capture_output=True, text=True)
            if(check_carla_process.stdout):
                print("Existing server running on that port, killing...")
                kill_carla_process = subprocess.run(kill_carla)
            print("Attempting to start CARLA server...")
            start_process = subprocess.Popen(start_carla, shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            time.sleep(10)
            try:
                print("Retrying connection...")
                self.client = carla.Client(self.host,self.port)
                self.client.set_timeout(10.0)
                self.world = self.client.load_world(self.Town)
                #self.world = self.client.get_world()
            except RuntimeError:
                print(f"Could not connect to client: {self.host}:{self.port}")
                return -1
                
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        if(self.no_render): settings.no_rendering_mode = True
        else: settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprints = self.world.get_blueprint_library()
        return 0
    
    def get_features(self):
        frame_feature_vec = []
        snapshot = self.world.get_snapshot()
        ego_snapshot = snapshot.find(self.ego.id)
        adv_snapshot = snapshot.find(self.adv.id)
        frame_feature_vec.append(snapshot.frame)

        (accident,distance,angle) = self.bounding_box_calcs([ego_snapshot, adv_snapshot])
        frame_feature_vec.append(accident)
        frame_feature_vec.append(distance)
        frame_feature_vec.append(angle)

        actors = []
        actors.append(ego_snapshot)
        actors.append(adv_snapshot)

        for a in actors:
            vel = a.get_velocity() #m/s
            accel = a.get_acceleration() #m/s^2
            ang_vel = a.get_angular_velocity() #deg/s
            frame_feature_vec.append(vel.x)
            frame_feature_vec.append(vel.y)
            frame_feature_vec.append(vel.z)
            frame_feature_vec.append(accel.x)
            frame_feature_vec.append(accel.y)
            frame_feature_vec.append(accel.z)
            frame_feature_vec.append(ang_vel.x)
            frame_feature_vec.append(ang_vel.y)
            frame_feature_vec.append(ang_vel.z)
        self.feature_vector.append(frame_feature_vec)
        return frame_feature_vec
    
    def bounding_box_calcs(self, actor_snapshots):
        bounding_boxes = []
        for actor_snapshot in actor_snapshots:
            actual_actor = self.world.get_actor(actor_snapshot.id)
            bb = [actual_actor.bounding_box.extent.x,actual_actor.bounding_box.extent.y,actual_actor.bounding_box.extent.z]
            bounding_boxes.append((carla.BoundingBox(actor_snapshot.get_transform().location,carla.Vector3D(y=bb[0],x=bb[1],z=bb[2])),actor_snapshot.get_transform(),actual_actor.attributes.get('role_name')))

        polygons = {}
        transforms = {}
        for bb in bounding_boxes:
            vertices = bb[0].get_local_vertices()
            coords = []
            for vert in vertices:
                if(vert.z > 0):
                    coords.append((vert.x,vert.y))
            coords_copy = coords[:]
            coords[-2] = coords_copy[-1]
            coords[-1] = coords_copy[-2]
            p = Polygon(coords)
            carla_yaw = bb[1].rotation.yaw
            if(carla_yaw > 0):
                p = rotate(p,carla_yaw - 90)
            elif(carla_yaw < 0):
                p = rotate(p,abs(carla_yaw) + 90)
            polygons[bb[2]] = p
            transforms[bb[2]] = bb[1]
        
        ego_vec = (transforms['ego'].get_forward_vector().x,transforms['ego'].get_forward_vector().y)
        diff_vec = transforms['adversary'].location - transforms['ego'].location 
        angle = Scenario.angle_between(ego_vec,(diff_vec.x,diff_vec.y))
        dist = polygons['ego'].distance(polygons['adversary'])
        accident = polygons['ego'].intersects(polygons['adversary'])
        return (accident, dist, angle)
    
    @abstractmethod
    #pass the ego vector first
    def angle_between(v1,v2):
        v1_u = v1/np.linalg.norm(v1)
        v2_u = v2/np.linalg.norm(v2)
        angle =  math.degrees(math.atan2(v1_u[0],v1_u[1]) - math.atan2(v2_u[0],v2_u[1]))
        if(angle < 0 and abs(angle) > 180): angle +=360
        if(angle > 180): angle = angle-360
        return angle
    
    def write_features(self, label):
        print("Writing features")
        df = self.dataframe
        #data_path = "c:\\data\\label\\"
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(),"data\\")
        time_str = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{time_str}_{round(self.score)}_{label}.csv"
        full_path = os.path.join(self.data_path,file_name)
        self.file_name = full_path
        df.to_csv(full_path)


