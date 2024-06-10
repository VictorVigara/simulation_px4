import numpy as np

class FakeContactDetector(): 
    def __init__(self, obstacles, ring_radius, logger) -> None:
        self.ring_radius = ring_radius
        self.vertices = obstacles
        self.logger = logger
        self.v1, self.v2, self.v3, self.v4 = self.vertices

    def point_to_plane_distance(self, point, plane_points):
        # Definimos el plano usando tres puntos
        p1, p2, p3 = plane_points
        # Vector normal del plano
        normal = np.cross(p2 - p1, p3 - p1)
        normal = normal / np.linalg.norm(normal)  # Normalizamos el vector

        # Distancia del punto al plano
        distance = np.dot(normal, point - p1)
        return np.abs(distance), normal
    
    # Usar el método de coordenadas baricéntricas para verificar si el punto está dentro del cuadrado
    def same_side(self, p1, p2, a, b):
        cp1 = np.cross(b - a, p1 - a)
        cp2 = np.cross(b - a, p2 - a)
        return np.dot(cp1, cp2) >= 0

    def is_point_in_square(self, point):
        v1, v2, v3, v4 = self.vertices
        # Proyección del punto en el plano
        plane_points = [v1, v2, v3]
        _, normal = self.point_to_plane_distance(point, plane_points)
        self.projected_point = point - np.dot(point - v1, normal) * normal
        #self.logger().info(f"Proj point = {self.projected_point}")
        x, y, z = self.projected_point


        # V1 menor x que V2 y V1 menor y que V2 para que funcione la comprobación        
        inside = (x >= self.v1[0] and x <= self.v2[0]) and (y >= self.v1[1] and y <= self.v2[1]) and (z <= self.v1[2] and z >= self.v3[2])
        return inside

    def check_collision(self, drone_position, rotation_matrix):
        distance, _ = self.point_to_plane_distance(drone_position, self.vertices[:3])
        #self.logger().info(f"Distance to wall: {distance}")
        if distance <= self.ring_radius and self.is_point_in_square(drone_position):
            self.logger().info("COLLISION DETECTED")
            self.calculate_colision_orientation(drone_position, rotation_matrix)

            return True
        return False

    def calculate_colision_orientation(self, position, rotation): 
        """ 
            Transform collision point from world frame to drone frame and calculate
            collision orientation and distance magnitude
        """

        # Transform collision point from world to drone frame
        translation_col_point_drone = self.projected_point-position
        collision_point_drone = np.dot(rotation.T, translation_col_point_drone)
        orientation = np.rad2deg(np.arctan2(collision_point_drone[1], collision_point_drone[0]))
        if orientation < 0: 
            orientation += 360
        
        self.logger().info(f"Orientation: {orientation} deg, Distance: {np.linalg.norm(collision_point_drone)} m")


