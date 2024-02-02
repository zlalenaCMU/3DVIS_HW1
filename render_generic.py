"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image, get_gif


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_plant(image_size=256, device=None):
    data = load_rgbd_data()
    renderer = get_points_renderer(
        image_size=image_size, radius = 0.02,
    )
    if device is None:
        device = get_device()


    #image 1
    image1 = torch.tensor(data["rgb1"])
    cameras1 =(data["cameras1"])
    depth1 = torch.tensor(data["depth1"])
    mask1 = torch.tensor(data["mask1"],)
    points1, rgb1 = unproject_depth_image(image1, mask1, depth1, cameras1)
    points1 = points1.unsqueeze(0)
    rgb1 = rgb1.unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=points1.to(device), features=rgb1.to(device))
    images = []
    flip = torch.tensor([[1,0 ,0 ], [0, -1.0, 0],[0, 0, 1]])

    for i in range(0,360, 5):
        R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist= 7 ,azim=i)
        T[0][0]=-.5
        T[0][1]=0
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R@flip, T=T, fov=60, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend*255
        rend = rend.astype(np.uint8)
        images.append(rend)
    imageio.mimsave("images/5.1_plant_1.gif", images, duration = 5, loop=10)

    # image 2
    image2 = torch.tensor(data["rgb2"])
    cameras2 = (data["cameras2"])
    depth2 = torch.tensor(data["depth2"])
    mask2 = torch.tensor(data["mask2"], )
    points2, rgb2 = unproject_depth_image(image2, mask2, depth2, cameras2)
    points2 = points2.unsqueeze(0)
    rgb2 = rgb2.unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=points2.to(device), features=rgb2.to(device))
    images = []
    flip = torch.tensor([[1, 0, 0], [0, -1.0, 0], [0, 0, 1]])

    for i in range(0, 360, 5):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=7, azim=i)
        T[0][0] = -.5
        T[0][1] = 0
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R @ flip, T=T, fov=60, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend * 255
        rend = rend.astype(np.uint8)
        images.append(rend)
    imageio.mimsave("images/5.1_plant_2.gif", images, duration=5, loop=10)


    # combo of pnt clouds
    # tutorialspoint said torch.cat() is used to concatenate two or more tensors
    # So im gonna trust that
    rgb2 =rgb2.squeeze(0)
    rgb1 = rgb1.squeeze(0)
    points1 = points1.squeeze(0)
    points2 = points2.squeeze(0)
    points = torch.cat((points1,points2)).unsqueeze(0)
    rgb = torch.cat((rgb1, rgb2)).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=points.to(device), features=rgb.to(device))
    images = []
    for i in range(0, 360, 5):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=7, azim=i)
        T[0][0] = -.5
        T[0][1] = 0
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R @ flip, T=T, fov=60, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend * 255
        rend = rend.astype(np.uint8)
        images.append(rend)
    imageio.mimsave("images/5.1_plant_combo.gif", images, duration=5, loop=10)


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus(image_size=256, num_samples=200, device=None):
    """
        Renders a torus using parametric sampling. Samples num_samples ** 2 points.
        """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2*np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    R = 1
    r = .5
    x = (R + r*torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r*torch.cos(Theta))*torch.sin(Phi)
    z = r*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    images = get_gif(renderer,sphere_point_cloud)
    return images

def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
def new_shape(image_size=256,num_samples=200, device=None):
    """
           Renders a torus using parametric sampling. Samples num_samples ** 2 points.
           """
    if device is None:
        device = get_device()
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    R = .7
    r = .5
    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.sin(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    images = get_gif(renderer, sphere_point_cloud)
    return images

def render_ultratech(image_size=256,num_samples=200, device=None):
    """
           Renders a torus using parametric sampling. Samples num_samples ** 2 points.
           """
    if device is None:
        device = get_device()
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    t = .7
    r = .4729
    x = t*torch.cos(Theta) - r*torch.cos(Phi)
    y = t*torch.sin(Theta)- r*torch.sin(Phi)
    z = Theta

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    images = get_gif(renderer, sphere_point_cloud)
    return images

def render_klein(image_size=256,num_samples=200, device=None):
    """
           Renders a torus using parametric sampling. Samples num_samples ** 2 points.
           """
    if device is None:
        device = get_device()
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0,  np.pi, num_samples)
    # Densely sample phi and theta on a grid
    v, u = torch.meshgrid(phi, theta)
    aa = .5
    b = .5
    #https: // virtualmathmuseum.org / Surface / klein_bottle / klein_bottle.html  #:~:text=Klein%20Bottle%20Hermann%20Karcher%20Parametric,See%20the%20Mobius%20Strip%20first.
    x = (aa + torch.cos(v / 2) * torch.sin(u) - torch.sin(v / 2) * torch.sin(2 * u)) * torch.cos(v)
    y = (aa + torch.cos(v / 2) * torch.sin(u) - torch.sin(v / 2) * torch.sin(2 * u)) * torch.sin(v)
    z = torch.sin(v / 2) * torch.sin(u) + torch.cos(v / 2) * torch.sin(2 * u)
    # x = (a+b*(torch.cos(Theta/2)*torch.sin(Phi) - torch.sin(Theta/2)*torch.sin(2*Phi))*torch.cos(Theta))
    # y = (a+b*(torch.cos(Theta/2)*torch.sin(Phi) - torch.sin(Theta/2)*torch.sin(2*Phi))*torch.sin(Theta))
    # z = b*(torch.sin(Theta/2)*torch.sin(Phi)+torch.cos(Theta/2)*torch.sin(2*Phi))
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    images = get_gif(renderer, sphere_point_cloud, dist = 3.5)
    return images
def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.3
    max_value = 1.3
    R = .7
    r = .5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = torch.pow((torch.sqrt(torch.pow(X,2)+ torch.pow(Y,2))-R),2)+Z**2-r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images = get_gif(renderer, mesh, dist=3, lights = lights)
    return images

#https://cs-people.bu.edu/sbargal/Fall%202016/lecture_notes/Nov_3_3d_geometry_representation
def render_new_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.3
    max_value = 1.3
    e_1 = .25
    e_2 =.25
    r_x = 1
    r_y = 1
    r_z = 1
    r_axial =1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = torch.pow(torch.pow(torch.pow(X/r_x, 2/e_2)+ torch.pow(Y/r_y, 2/e_2),e_2/e_1) - r_axial,2/e_2)+ torch.pow(Z/r_z, 2/e_1)-1

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, 4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images = get_gif(renderer, mesh, dist=3, lights = lights)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="../images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    # if args.render == "point_cloud":
    #     image = render_bridge(image_size=args.image_size)
    # elif args.render == "parametric":
    #     image = render_sphere(image_size=args.image_size)
    # elif args.render == "implicit":
    #     image = render_sphere_mesh(image_size=args.image_size)
    # else:
    #     raise Exception("Did not understand {}".format(args.render))
    images = render_new_mesh(image_size=256, voxel_size=64, device=None)
    imageio.mimsave("images/5.3_new.gif", images, duration=5, loop=10)
    #plt.imsave("images/5.3_torus.jpg", image)

