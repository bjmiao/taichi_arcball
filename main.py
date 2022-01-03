import taichi as ti
from taichi.lang import collect_kernel_profile_metrics
from RayTracerUtils import Hittable_list, Camera, Triangle, Ray
ti.init()
npoints = 8
ntriangles = 12
point = ti.Vector.field(3, ti.f32, npoints)
triangle = ti.Vector.field(3, ti.f32, ntriangles)
screen_width, screen_height = 240, 240
screen = ti.Vector.field(3, dtype=ti.f32, shape=(screen_width, screen_height))

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

# Lambertian reflection model
@ti.func
def ray_color(ray):
    default_color = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
    if is_hit:
        if material == 0:
            default_color = color
        else:
            hit_point_to_source = to_light_source(hit_point, ti.Vector([0, 5.4 - 3.0, -1]))
            default_color = color * ti.max(hit_point_to_source.dot(hit_point_normal) / (hit_point_to_source.norm() * hit_point_normal.norm()), 0.0)
    return default_color


@ti.kernel
def update_camera():
    for i, j in screen:
        u = i / screen_width
        v = j / screen_height
        ray = camera.get_ray(u, v)
        color = ray_color(ray)
        screen[i, j] = color

scene = Hittable_list()
camera = Camera()

point[0] = ti.Vector([1.0, 1.0, 1.0])
point[1] = ti.Vector([1.0, 1.0, -1.0])
point[2] = ti.Vector([1.0, -1.0, -1.0])
point[3] = ti.Vector([1.0, -1.0, 1.0])
point[4] = ti.Vector([-1.0, 1.0, 1.0])
point[5] = ti.Vector([-1.0, 1.0, -1.0])
point[6] = ti.Vector([-1.0, -1.0, -1.0])
point[7] = ti.Vector([-1.0, -1.0, 1.0])

colormap = {
    'red': ti.Vector([1.0, 0.0, 0.0]),
    'green': ti.Vector([0.0, 1.0, 0.0]),
    'blue': ti.Vector([0.0, 0.0, 1.0]),
    'violet': ti.Vector([1.0, 0.0, 1.0]),
    'yellow': ti.Vector([0.0, 1.0, 1.0])
}

# left
scene.add(Triangle(point[0], point[1], point[2], color = colormap['green']))
scene.add(Triangle(point[0], point[3], point[2], color = colormap['blue']))

# right
scene.add(Triangle(point[4], point[5], point[6], color = colormap['red']))
scene.add(Triangle(point[4], point[7], point[6], color = colormap['green']))

# up
scene.add(Triangle(point[0], point[1], point[5], color = colormap['blue']))
scene.add(Triangle(point[0], point[4], point[5], color = colormap['green']))

# down
scene.add(Triangle(point[2], point[3], point[7], color = colormap['red']))
scene.add(Triangle(point[2], point[6], point[7], color = colormap['blue']))

# back
scene.add(Triangle(point[0], point[7], point[3], color = colormap['violet']))
scene.add(Triangle(point[0], point[4], point[7], color = colormap['yellow']))

# forward
scene.add(Triangle(point[6], point[1], point[2], color = colormap['violet']))
# scene.add(Triangle(point[6], point[1], point[5], color = colormap['yellow']))

is_dragging = False
camera = Camera()
gui = ti.GUI("Arcball", (screen_width, screen_height))
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS, ti.GUI.MOTION, ti.GUI.RELEASE):
        if e.type == ti.GUI.PRESS:
            if e.key == ti.GUI.LMB:
                is_dragging = True
                start_mouse_x, start_mouse_y = gui.get_cursor_pos()
        elif e.type == ti.GUI.RELEASE:
            if e.key == ti.GUI.LMB:
                is_dragging = False
                stop_mouse_x, stop_mouse_y = gui.get_cursor_pos()
    now_mouse_x, now_mouse_y = gui.get_cursor_pos()
    if (is_dragging):
        dragging_mouse_x, dragging_mouse_y = (now_mouse_x - start_mouse_x, now_mouse_y - start_mouse_y)
    update_camera()
    gui.set_image(screen)
    gui.text(
        content=f'Mouse_x, mouse_y = {now_mouse_x:.2f}, {now_mouse_y:.2f}', pos=(0.6, 0.95), color=0x000000
    )
    gui.text(
        content=f'start_x, start_y = {start_mouse_x:.2f}, {start_mouse_y:.2f}', pos=(0.6, 0.9), color=0x000000
    )
    gui.text(
        content=f'dragging_x, dragging_y = {dragging_mouse_x:.2f}, {dragging_mouse_y:.2f}', pos=(0.6, 0.85), color=0x000000
    )
    gui.text(
        content=f'stop_x, stop_y = {stop_mouse_x:.2f}, {stop_mouse_y:.2f}', pos=(0.6, 0.8), color=0x000000
    )

    gui.show()
