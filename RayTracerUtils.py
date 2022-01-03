import taichi as ti

PI = 3.14159265359

@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    def at(self, t):
        return self.origin + t * self.direction


def area_triangle_cpu(pointA, pointB, pointC):
    s1 = pointB - pointA
    s2 = pointC - pointA
    return 0.5 * s1.cross(s2).norm()

@ti.func
def area_triangle(pointA, pointB, pointC):
    s1 = pointB - pointA
    s2 = pointC - pointA
    return 0.5 * s1.cross(s2).norm()

@ti.data_oriented
class Triangle:
    def __init__(self, pointA, pointB, pointC, color):
        self.pointA = pointA
        self.pointB = pointB
        self.pointC = pointC
        self.norm_direction = (pointA - pointC).cross(pointB - pointA).normalized()
        self.D = -self.pointA.dot(self.norm_direction)  # plane Parameter D
        self.area = area_triangle_cpu(self.pointA, self.pointB, self.pointC)
        self.material = 0
        self.color = color
        print(f"{self.norm_direction=}, {self.D=}, {self.area=}")
    @ti.func
    def hit(self, ray, t_min = 0.001, t_max = 10e8):
        is_hit = False
        front_face = False
        root = 0.0
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = self.norm_direction
        # print(f"{ray.direction=}, {ray.origin=}")
        if abs(ray.direction.dot(self.norm_direction)) < 1e-5:
            # print("Ray Perpendicular to plane. Not hit")
            is_hit = False
        else:
            root = (self.pointA - ray.origin).dot(self.norm_direction) / ray.direction.dot(self.norm_direction)
            # print(f"{root=}")
            if (root < 0):
                is_hit = False
            else:
                hit_point = ray.origin + root * ray.direction
                # inside outside check
                area_t1 = area_triangle(hit_point, self.pointA, self.pointB)
                area_t2 = area_triangle(hit_point, self.pointB, self.pointC)
                area_t3 = area_triangle(hit_point, self.pointC, self.pointA)
                if abs(area_t1 + area_t2 + area_t3 - self.area) < 1e-3:
                    # print("Hit")
                    is_hit = True
                else:
                    # print("No hit")
                    is_hit = False
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, obj):
        self.objects.append(obj)
    def clear(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color

    @ti.func
    def hit_shadow(self, ray, t_min=0.001, t_max=10e8):
        is_hit_source = False
        is_hit_source_temp = False
        hitted_dielectric_num = 0
        is_hitted_non_dielectric = False
        # Compute the t_max to light source
        is_hit_tmp, root_light_source, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
        self.objects[0].hit(ray, t_min)
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, root_light_source)
            if is_hit_tmp:
                if material_tmp != 3 and material_tmp != 0:
                    is_hitted_non_dielectric = True
                if material_tmp == 3:
                    hitted_dielectric_num += 1
                if material_tmp == 0:
                    is_hit_source_temp = True
        if is_hit_source_temp and (not is_hitted_non_dielectric) and hitted_dielectric_num == 0:
            is_hit_source = True
        return is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio = 1.0):
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        # self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vdown = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.w = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.reset()
    @ti.kernel
    def reset(self):
        self.lookfrom[None] = [0.0, 0.0, -5.0]
        self.vup[None] = [0.0, 1.0, 0.0]
        self.w[None] = [0.0, 0.0, -1.0]
        self.calculate_parameter()

    @ti.kernel
    def set_lookat(self, x:ti.f32, y:ti.f32, z:ti.f32):
        self.w[None] = ti.Vector([x, y, z]).normalized()
        self.calculate_parameter()

    @ti.kernel
    def set_lookfrom(self, x:ti.f32, y:ti.f32, z:ti.f32):
        self.lookfrom[None] = [x, y, z]
        self.calculate_parameter()

    @ti.func
    def calculate_parameter(self):
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_origin[None] = self.lookfrom[None]
        u = (self.vup[None].cross(self.w[None])).normalized()
        v = self.w[None].cross(u)
        print(u, v)
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - self.w[None]
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.cam_origin[None], self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None])