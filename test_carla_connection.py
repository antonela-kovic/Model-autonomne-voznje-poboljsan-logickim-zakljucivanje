import carla

def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)

    world = client.get_world()
    print("CONNECTED OK")
    print("Map name:", world.get_map().name)

    spawn_points = world.get_map().get_spawn_points()
    print("Spawn count:", len(spawn_points))

    if len(spawn_points) > 0:
        print("Spawn[0]:", spawn_points[0])

if __name__ == "__main__":
    main()