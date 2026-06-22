# Generira XML rute koje prolaze kroz STVARNA raskrižja u Town02 (ne zavoje).
# Poboljšana verzija: razlikuje pravo raskrižje (3+ povezanih cesta) od zavoja,
# i traži rute gdje ego STVARNO skreće UNUTAR raskrižja.
import sys
import os

CARLA_ROOT = r"C:\Users\Antonela\Documents\CARLA_0.9.16"
CARLA_PYTHON_API = os.path.join(CARLA_ROOT, "PythonAPI", "carla")

if not os.path.exists(os.path.join(CARLA_PYTHON_API, "agents")):
    raise FileNotFoundError(
        "Nije pronađen CARLA agents folder. Provjeri CARLA_ROOT.\n"
        f"Tražena putanja: {os.path.join(CARLA_PYTHON_API, 'agents')}"
    )

sys.path.append(CARLA_PYTHON_API)

import carla
import math
import xml.etree.ElementTree as ET
from agents.navigation.global_route_planner import GlobalRoutePlanner


# Cache da ne računamo isti junction više puta.
_junction_road_cache = {}


def is_real_junction_wp(wp):
    """Pravo raskrižje: junction povezuje 3+ različitih cesta. Zavoj: 1-2."""
    if wp is None or not wp.is_junction:
        return False
    junction = wp.get_junction()
    if junction is None:
        return False
    jid = junction.id
    if jid in _junction_road_cache:
        return _junction_road_cache[jid]
    try:
        pairs = junction.get_waypoints(carla.LaneType.Driving)
    except Exception:
        _junction_road_cache[jid] = False
        return False
    roads = set()
    for a, b in pairs:
        roads.add(a.road_id)
        roads.add(b.road_id)
    result = len(roads) >= 3
    _junction_road_cache[jid] = result
    return result


def route_distance(route):
    total = 0.0
    for i in range(1, len(route)):
        a = route[i - 1][0].transform.location
        b = route[i][0].transform.location
        total += math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    return total


def count_real_junction_segments(route):
    """Broji odvojene zone STVARNIH raskrižja (ne zavoja)."""
    count = 0
    was_in = False
    for wp, _ in route:
        real = is_real_junction_wp(wp)
        if real and not was_in:
            count += 1
            was_in = True
        elif not real:
            was_in = False
    return count


def count_junction_turns(route, turn_threshold_deg=45.0):
    """
    Broji koliko puta ego STVARNO skrene UNUTAR stvarnog raskrižja.
    Mjeri ukupnu promjenu smjera dok je ego u real-junction zoni.
    Samo skretanja > turn_threshold_deg se broje (pravo skretanje, ne blagi zavoj).
    """
    turns = 0
    in_junction = False
    yaw_start = None

    def norm(a):
        while a > 180.0: a -= 360.0
        while a < -180.0: a += 360.0
        return a

    for wp, _ in route:
        real = is_real_junction_wp(wp)
        yaw = wp.transform.rotation.yaw
        if real and not in_junction:
            in_junction = True
            yaw_start = yaw
        elif not real and in_junction:
            in_junction = False
            if yaw_start is not None:
                total_turn = abs(norm(yaw - yaw_start))
                if total_turn > turn_threshold_deg:
                    turns += 1
    return turns


def write_route_xml(route, output_path):
    root = ET.Element("route", id="_", town="_")
    for wp, _ in route:
        tf = wp.transform
        loc = tf.location
        rot = tf.rotation
        ET.SubElement(root, "waypoint", {
            "pitch": str(rot.pitch), "roll": str(rot.roll),
            "x": str(loc.x), "y": str(loc.y),
            "yaw": str(rot.yaw), "z": "0.0",
        })
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.load_world("Town02")
    carla_map = world.get_map()

    spawn_points = carla_map.get_spawn_points()
    grp = GlobalRoutePlanner(carla_map, 2.0)

    candidates = []

    for start_idx, start_tf in enumerate(spawn_points):
        for end_idx, end_tf in enumerate(spawn_points):
            if start_idx == end_idx:
                continue
            try:
                route = grp.trace_route(start_tf.location, end_tf.location)
            except Exception:
                continue
            if not route:
                continue

            dist = route_distance(route)
            if dist < 40.0 or dist > 250.0:
                continue

            real_segments = count_real_junction_segments(route)
            junction_turns = count_junction_turns(route)

            # KLJUČNO: tražimo rute gdje ego STVARNO SKREĆE u STVARNOM raskrižju.
            if junction_turns == 0:
                continue

            # Bodujemo: najviše vrijede stvarna skretanja u raskrižju,
            # penaliziramo duljinu da test bude brz.
            score = junction_turns * 100.0 + real_segments * 20.0 - dist * 0.1

            candidates.append({
                "score": score, "start_idx": start_idx, "end_idx": end_idx,
                "route": route, "dist": dist,
                "real_segments": real_segments, "junction_turns": junction_turns,
            })

    if not candidates:
        print("Nije pronađena ruta sa stvarnim skretanjem u raskrižju.")
        print("Pokušaj povećati gornju granicu udaljenosti (dist > 250).")
        return

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_n = min(5, len(candidates))

    print("\n=== Test rute kroz STVARNA raskrižja (sa skretanjem) ===\n")
    for rank in range(top_n):
        c = candidates[rank]
        out = f"route_intersection_test_{rank + 1}.xml"
        write_route_xml(c["route"], out)
        s = spawn_points[c["start_idx"]]
        print("=" * 60)
        print("FILE                  =", out)
        print("START_SPAWN_INDEX     =", c["start_idx"])
        print("END_SPAWN_INDEX       =", c["end_idx"])
        print("distance_m            =", round(c["dist"], 2))
        print("real_junction_segments=", c["real_segments"])
        print("junction_turns        =", c["junction_turns"], "(stvarna skretanja u raskrižju)")
        print("score                 =", round(c["score"], 2))
        print("start x/y/yaw         =",
              round(s.location.x, 2), round(s.location.y, 2), round(s.rotation.yaw, 2))

    print("=" * 60)
    print("\nKoristi route_intersection_test_1.xml (najviše skretanja u raskrižju).")
    print("VAŽNO: ego se mora spawnati na START_SPAWN_INDEX te rute!")
    print("U sample.py promijeni i spawn index ako START_SPAWN_INDEX nije 0.")


if __name__ == "__main__":
    main()