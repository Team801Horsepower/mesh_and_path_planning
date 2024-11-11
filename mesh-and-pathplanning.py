import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as netx
from scipy.interpolate import CubicSpline
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_edge_in_white_area(point1, point2, image):
    # Use linear interpolation to check if the edge crosses black areas
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    num_points = int(np.hypot(x2 - x1, y2 - y1))
    for t in np.linspace(0, 1, num_points):
        x = int(x1 * (1 - t) + x2 * t)
        y = int(y1 * (1 - t) + y2 * t)
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            return False
        if image[y, x] != 255:
            return False
    return True

def generate_grid_points(image, mesh_spacing=50, key_points=None):
    h, w = image.shape
    points = []
    for y in range(0, h, mesh_spacing):
        for x in range(0, w, mesh_spacing):
            if image[y, x] == 255:
                points.append((x, y))
    # Add key points to ensure they are always included
    if key_points is not None:
        for kp in key_points:
            if (kp.x, kp.y) not in points:
                points.append((kp.x, kp.y))
    return np.array(points)

def build_graph_from_grid(points, image, mesh_spacing=50, key_points=None):
    graph = netx.Graph()
    point_dict = {(x, y): i for i, (x, y) in enumerate(points)}

    for (x, y) in points:
        neighbors = [
            (x + dx, y + dy)
            for dx in [-mesh_spacing, 0, mesh_spacing]
            for dy in [-mesh_spacing, 0, mesh_spacing]
            if (dx != 0 or dy != 0) and (x + dx, y + dy) in point_dict
        ]
        for nx, ny in neighbors:
            node1 = Node(x, y)
            node2 = Node(nx, ny)
            if is_edge_in_white_area(node1, node2, image):
                distance = euclidean_distance(node1, node2)
                graph.add_edge((x, y), (nx, ny), weight=distance)

    # Add edges from key points to their nearest neighbors if not aligned with the grid
    if key_points is not None:
        for kp in key_points:
            if (kp.x % mesh_spacing != 0 or kp.y % mesh_spacing != 0):
                # Find the nearest grid points that are misaligned
                nearest_points = [p for p in points if not np.array_equal(p, (kp.x, kp.y))]
                nearest_points = sorted(nearest_points, key=lambda p: np.hypot(p[0] - kp.x, p[1] - kp.y))[:8]
                for nearest_point in nearest_points:
                    node1 = Node(kp.x, kp.y)
                    node2 = Node(nearest_point[0], nearest_point[1])
                    if is_edge_in_white_area(node1, node2, image):
                        distance = euclidean_distance(node1, node2)
                        graph.add_edge((kp.x, kp.y), tuple(nearest_point), weight=distance)
    return graph

def find_path(graph, key_points):
    path_points = []
    for i in range(len(key_points) - 1):
        try:
            source = (key_points[i].x, key_points[i].y)
            target = (key_points[i + 1].x, key_points[i + 1].y)
            if source not in graph or target not in graph:
                print(f"Error: One or both of the key points {source}, {target} are not in the graph.")
                continue
            path_segment = netx.shortest_path(graph, source=source, target=target, weight='weight')
            if path_points:
                # Avoid duplicating points at the segment boundaries
                path_segment = path_segment[1:]
            path_points.extend(path_segment)
        except netx.NetworkXNoPath:
            print(f"No path found between key point {i} and key point {i + 1}")
    return np.array(path_points)

def smooth_path(points, key_points_indices, window_size=5):
    # Apply a moving average filter to smooth the path points while preserving key points
    smoothed_points = points.copy()
    for i in range(1, len(points) - 1):
        if i not in key_points_indices:
            start = max(0, i - window_size // 2)
            end = min(len(points), i + window_size // 2 + 1)
            smoothed_points[i] = np.mean(points[start:end], axis=0)
    return smoothed_points

def cubic_spline_smooth(points, key_points_indices):
    # Apply cubic spline interpolation to further smooth the path between key points
    t = np.arange(len(points))
    x = points[:, 0]
    y = points[:, 1]

    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    t_fine = np.linspace(0, len(points) - 1, len(points) * 5)
    smoothed_x = cs_x(t_fine)
    smoothed_y = cs_y(t_fine)

    # Combine smoothed points and preserve key points
    smoothed_points = np.vstack((smoothed_x, smoothed_y)).T
    for idx in key_points_indices:
        smoothed_points[idx * 5] = points[idx]

    return smoothed_points

def generate_smoothed_path(path_points, key_points_indices, smoothing_factor=5):
    # Apply a hybrid smoothing approach: moving average followed by cubic spline
    smoothed_points = smooth_path(path_points, key_points_indices, window_size=smoothing_factor)
    final_smoothed_points = cubic_spline_smooth(smoothed_points, key_points_indices * smoothing_factor)
    return final_smoothed_points[:, 0], final_smoothed_points[:, 1]

def generate_random_key_points(image, num_points):
    h, w = image.shape
    key_points = []
    while len(key_points) < num_points:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if image[y, x] == 255:
            key_points.append(Node(x, y))
    return key_points

def main(smoothing_factor=5, mesh_spacing=50, num_random_points=5):
    # Load the black and white image
    image = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # Threshold the image to ensure binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Generate random key points
    key_points = generate_random_key_points(binary_image, num_points=num_random_points)
    key_points_indices = list(range(len(key_points)))

    # Set mesh spacing to adjust fineness of the mesh
    mesh_spacing = mesh_spacing  # Adjust this value to make the mesh finer or coarser

    # Generate grid points
    points = generate_grid_points(binary_image, mesh_spacing=mesh_spacing, key_points=key_points)

    # Build graph from grid
    graph = build_graph_from_grid(points, binary_image, mesh_spacing=mesh_spacing, key_points=key_points)

    # Find the shortest path through all key points
    path_points = find_path(graph, key_points)

    if len(path_points) == 0:
        print("No valid path found connecting all key points.")
        return

    # Plot the grid, edges, and path
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.5, s=1)
    plt.imshow(binary_image, cmap='gray')

    # Plot edges
    for edge in graph.edges:
        x_values = [edge[0][0], edge[1][0]]
        y_values = [edge[0][1], edge[1][1]]
        plt.plot(x_values, y_values, color='lightblue', linewidth=0.5)

    # Plot path
    plt.plot(path_points[:, 0], path_points[:, 1], 'o-', color='red', markersize=4)
    for key_point in key_points:
        plt.plot(key_point.x, key_point.y, 'go', markersize=8)
    smoothed_x, smoothed_y = generate_smoothed_path(path_points, key_points_indices, smoothing_factor)
    plt.plot(smoothed_x, smoothed_y, 'g-', linewidth=2, label='Smoothed Path')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().invert_yaxis()
    plt.title('Grid-Based Path with Hybrid Smoothed Segments and Edges')
    plt.show()

if __name__ == "__main__":
    main(smoothing_factor=4, mesh_spacing=30, num_random_points=50)
