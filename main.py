"""
Learing foe 3d Vision
HW 1
1/27/24
Zoe LaLena
"""

# imports

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
from utils import get_device, get_mesh_renderer, load_cow_mesh
from tqdm.auto import tqdm
import math
from PIL import Image, ImageDraw
from utils import get_gif
## Section 0

# 0.1 Rendering your first mesh

# code copied from the render_mesh.py file
def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

## 1. Practicing with Camera

# 1.1 360-Degree Renders
def render_gif(
    path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    """
    Renders a cow turntable gif
    """
    images = []
    if device is None:
        device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    # render images for different angles
    for i in range(0,360, 5):
        R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist= -3, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend*255
        rend = rend.astype(np.uint8)
        images.append(rend)
    return images

# 1.2 Re-creating the Dolly Zoom
def dolly_zoom(image_size=256, num_frames=10, duration=3, device=None, output_file="images/1.2_dolly.gif"):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    fovs = torch.linspace(5, 120, num_frames)
    renders = []


    for fov in tqdm(fovs):
        rads = fov*math.pi/180
        distance = 5/(2*np.tan(.5*rads))
        print(distance)
        T = [[0, 0, distance]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop = 10)


## 2. Pacticing with meshes

# 2.1 Constructing a Tetrahedron
# see the obj files I created to make this process reuse other code

## 3. Re-texturing a mesh
def render_grad(
    cow_path="data/cow.obj", image_size=256, color_1=[0, 0, 1],color_2=[1,0,0], device=None,
):
    images = []
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    # get shape
    one, vert_num, dims = vertices.shape
    textures = np.zeros((vert_num,3))

    # find min and max z values
    z_mins = torch.min(vertices, dim =1)
    z_min = z_mins.values[0][2]
    z_maxs = torch.max(vertices, dim =1)
    z_max = z_maxs.values[0][2]

    # for each vertex, color according to gradient equation
    for vert in range(0,vert_num):
        z = vertices[:, vert,2]
        alpha = (z-z_min)/(z_max-z_min)
        alpha = alpha.item()
        color = alpha*color_2+(1-alpha)*color_1
        textures[vert] = color
    textures = torch.tensor(textures, dtype=torch.float)  # (1, N_v, 3)
    textures = textures.unsqueeze(0)

    # create our mesh with our new colors/textures
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # render 360
    for i in range(0,360, 5):
        R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist= -3, azim=i)
    # Prepare the camera:

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend*255
        rend = rend.astype(np.uint8)
        images.append(rend)
    return images

## 4. Camera Transformations

def render_transform(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

## 5. Rendering Generic 3D Representations


# 6
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

from pytorch3d.io import load_objs_as_meshes


def render_newcow(
    cow_path="data/newcow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    mesh = load_objs_as_meshes([cow_path], device=device)
    plt.figure(figsize=(7, 7))
    texturesuv_image_matplotlib(mesh.textures, subsample=None)
    plt.show()

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = get_gif(renderer,mesh,dist=3,device=None, lights = lights)

    return images

def mesh_sampling_bonus(cow_path="data/cow.obj",N=100):



def main():

    # color = [1,0,1]
    # # 0.1
    # image = render_cow(color=color)
    # plt.imsave("images/0.1_cow_render.jpg", image)
    #
    # # 1.1
    # images = render_gif()
    # imageio.mimsave("images/1.1_cow_render.gif", images, duration = 5, loop = 10)
    #
    # #1.2
    # dolly_zoom()
    #
    # #2.1
    # images = render_gif(path="data/tetra.obj", image_size=256, color=color)
    # imageio.mimsave("images/2.1_tetra_render.gif", images, duration=5, loop=10)
    # # 2.2
    # images = render_gif(path="data/cube.obj", image_size=256, color=color)
    # imageio.mimsave("images/2.2_cube_render.gif", images, duration=5, loop=10)
    #
    # # 3
    # color_2 = np.array([240 / 256, 15 / 256, 210 / 256])
    # color_1 = np.array([1, 1, 1])
    # images = render_grad(color_1=color_1, color_2=color_2)
    # imageio.mimsave("images/3_grad_cow.gif", images, duration=5, loop = 10)
    #
    # # 4
    # angle = math.pi / 2
    # cow_1 = render_transform(cow_path="data/cow_with_axis.obj", image_size=256,
    #                    R_relative=[[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
    #                    T_relative=[0, 0, 0],
    #                    device=None,
    #                    )
    # cow_2 = render_transform(cow_path="data/cow_with_axis.obj", image_size=256,
    #                    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    #                    T_relative=[.5, -.5, 0],
    #                    device=None,
    #                    )
    #
    # cow_3 = render_transform(cow_path="data/cow_with_axis.obj", image_size=256,
    #                    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    #                    T_relative=[0, 0, 2],
    #                    device=None,
    #                    )
    # angle = -angle
    # cow_4 = render_transform(cow_path="data/cow_with_axis.obj", image_size=256,
    #                    R_relative=[[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]],
    #                    T_relative=[3, 0, 3],  # zoom way out when you've lost it
    #                    device=None,
    #                    )
    # plt.imsave("images/4_cow1.jpg", cow_1)
    # plt.imsave("images/4_cow2.jpg", cow_2)
    # plt.imsave("images/4_cow3.jpg", cow_3)
    # plt.imsave("images/4_cow4.jpg", cow_4)
    #
    # # 5
    # image = render_generic.render_bridge(image_size=256)
    # plt.imsave("images/5_bridge.jpg", image)
    #
    # #5.1
    # #i just made 1 gross mega function for this step, sorry
    # render_generic.render_plant()
    #
    # #5.2 Parametric functions
    # images = render_generic.render_torus(image_size=256, num_samples=1000, device=None)
    # imageio.mimsave("images/5.2_torus.gif", images, duration=5, loop = 10)
    #
    # # cool new shape
    # images = render_generic.render_klein(image_size=256, num_samples=1000, device=None)
    # imageio.mimsave("images/5.2_idk_wtf_this_is.gif", images, duration=5, loop=10)
    #
    # # 5.3 Implicit Surfaces


    # 6
    images = render_newcow()
    imageio.mimsave("images/6_gay_cow.gif", images, duration=5, loop=10)
if __name__ == '__main__':
    main()