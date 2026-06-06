import carla
import time
from pcla_functions.location_to_waypoint import location_to_waypoint
from pcla_functions.route_maker import route_maker


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)

    print("Loading Town02...")
    client.load_world("Town02")
    time.sleep(5)

    world = client.get_world()
    print("Current map:", world.get_map().name)

    if "Town02" not in world.get_map().name:
        print("Town02 nije ucitan.")
        return

    spawn_points = world.get_map().get_spawn_points()
    print(f"Ukupno spawn pointova: {len(spawn_points)}")

    start_idx = 0
    end_idx = 10   # kasnije možemo promijeniti

    start_tf = spawn_points[start_idx]
    end_tf = spawn_points[end_idx]

    print("\nSTART TRANSFORM:")
    print(start_tf)

    print("\nEND TRANSFORM:")
    print(end_tf)

    waypoints = location_to_waypoint(
        client,
        start_tf.location,
        end_tf.location,
        distance=2,
        draw=True
    )

    print(f"\nGenerated {len(waypoints)} waypoints.")

    if len(waypoints) < 2:
        print("Premalo waypointa. Probaj drugi end_idx.")
        return

    save_path = "route_spawn0.xml"
    route_maker(waypoints, save_path)

    print(f"\nSaved new route to: {save_path}")


if __name__ == "__main__":
    main()