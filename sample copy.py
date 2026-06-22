import carla
import time
import math
from PCLA import PCLA
from leaderboard_codes.controller import VehiclePIDController
import os

# Ova funkcija normalizira kut u raspon: (-180, 180]
def normalize_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


# Zamjena funkcije detect_front_actor
def extract_dynamic_actor_facts(
    world,
    ego_vehicle,
    max_forward=30.0,
    max_side=8.0,
    max_height_diff=2.5
):
    """
    Vraća informacije o vozilima i pješacima oko ego vozila.

    local_forward:
        pozitivno = aktor je ispred ego vozila

    local_right:
        pozitivno = aktor je desno od ego vozila

    closing_speed:
        pozitivno = ego vozilo se približava aktoru

    gap_m:
        približan razmak između rubova bounding boxova

    ttc:
        procijenjeno vrijeme do sudara ako se brzine ne promijene
    """
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_vel = ego_vehicle.get_velocity()

    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    ego_forward_speed = (
        ego_vel.x * forward.x +
        ego_vel.y * forward.y +
        ego_vel.z * forward.z
    )

    ego_half_length = ego_vehicle.bounding_box.extent.x
    ego_half_width = ego_vehicle.bounding_box.extent.y

    actor_facts = []

    for actor in world.get_actors():
        if actor.id == ego_vehicle.id:
            continue

        actor_type = actor.type_id

        is_vehicle = actor_type.startswith("vehicle.")
        is_pedestrian = actor_type.startswith("walker.")

        if not is_vehicle and not is_pedestrian:
            continue

        actor_loc = actor.get_location()

        rel_x = actor_loc.x - ego_loc.x
        rel_y = actor_loc.y - ego_loc.y
        rel_z = actor_loc.z - ego_loc.z

        if abs(rel_z) > max_height_diff:
            continue

        local_forward = (
            rel_x * forward.x +
            rel_y * forward.y +
            rel_z * forward.z
        )

        local_right = (
            rel_x * right.x +
            rel_y * right.y +
            rel_z * right.z
        )

        if local_forward < -5.0 or local_forward > max_forward:
            continue

        if abs(local_right) > max_side:
            continue

        actor_vel = actor.get_velocity()

        actor_yaw = actor.get_transform().rotation.yaw
        ego_yaw = ego_tf.rotation.yaw

        heading_difference = abs(
            normalize_angle_deg(actor_yaw - ego_yaw)
        )

        actor_forward_speed = (
            actor_vel.x * forward.x +
            actor_vel.y * forward.y +
            actor_vel.z * forward.z
        )

        actor_lateral_speed = (
            actor_vel.x * right.x +
            actor_vel.y * right.y +
            actor_vel.z * right.z
        )

        closing_speed = ego_forward_speed - actor_forward_speed

        actor_half_length = max(
            actor.bounding_box.extent.x,
            actor.bounding_box.extent.y
        )

        actor_half_width = min(
            actor.bounding_box.extent.x,
            actor.bounding_box.extent.y
        )

        gap_m = (
            local_forward -
            ego_half_length -
            actor_half_length
        )

        if closing_speed > 0.05 and gap_m > 0.0:
            ttc = gap_m / closing_speed
        else:
            ttc = float("inf")

        actor_facts.append({
            "actor": actor,
            "actor_id": actor.id,
            "type_id": actor_type,
            "is_vehicle": is_vehicle,
            "is_pedestrian": is_pedestrian,

            "local_forward": local_forward,
            "local_right": local_right,

            "gap_m": gap_m,
            "closing_speed": closing_speed,
            "ttc": ttc,

            "actor_forward_speed": actor_forward_speed,
            "actor_lateral_speed": actor_lateral_speed,

            "ego_half_width": ego_half_width,
            "actor_half_width": actor_half_width,
            "heading_difference": heading_difference,
        })

    return actor_facts

# Ova funkcija razdvaja rizik od vozika i rizik od pjesaka
def evaluate_vehicle_collision_risk(actor_facts, ego_vehicle):
    ego_half_width = ego_vehicle.bounding_box.extent.y

    best_risk = None

    for actor in actor_facts:
        if not actor["is_vehicle"]:
            continue
        
        # Lead-vehicle pravilo ne smije reagirati na vozila
        # koja dolaze iz suprotnog smjera ili presijecaju putanju.
        if actor["heading_difference"] > 60.0:
            continue

        if actor["local_forward"] <= 0.0:
            continue

        lane_corridor_half_width = (
            ego_half_width +
            actor["actor_half_width"] +
            0.40
        )

        same_path_corridor = (
            abs(actor["local_right"]) < lane_corridor_half_width
        )

        if not same_path_corridor:
            continue

        gap_m = actor["gap_m"]
        ttc = actor["ttc"]
        closing_speed = actor["closing_speed"]

        risk_level = "NONE"

        approaching = closing_speed > 0.30

        if gap_m < 2.5:
            risk_level = "EMERGENCY"

        elif gap_m < 4.0:
            risk_level = "BRAKE"

        elif approaching and gap_m < 5.0:
            risk_level = "BRAKE"

        elif approaching and ttc < 4.0:
            risk_level = "CAUTION"

        elif approaching and gap_m < 10.0:
            risk_level = "CAUTION"

        if risk_level != "NONE":
            candidate = {
                **actor,
                "risk_level": risk_level
            }

            if best_risk is None:
                best_risk = candidate
            elif candidate["ttc"] < best_risk["ttc"]:
                best_risk = candidate
            elif (
                candidate["ttc"] == best_risk["ttc"]
                and candidate["gap_m"] < best_risk["gap_m"]
            ):
                best_risk = candidate

    return best_risk

# Rizik od pjesaka
def evaluate_pedestrian_collision_risk(actor_facts, ego_vehicle):
    ego_half_width = ego_vehicle.bounding_box.extent.y

    best_risk = None

    for actor in actor_facts:
        if not actor["is_pedestrian"]:
            continue

        if actor["local_forward"] <= -1.0:
            continue

        if actor["local_forward"] > 20.0:
            continue

        pedestrian_corridor = ego_half_width + 0.8

        currently_in_path = (
            abs(actor["local_right"]) < pedestrian_corridor
        )

        lateral_speed = actor["actor_lateral_speed"]
        moving_toward_path = (
            actor["local_right"] * lateral_speed < 0.0
        )

        if abs(lateral_speed) > 0.05:
            time_to_path = (
                abs(actor["local_right"]) - pedestrian_corridor
            ) / abs(lateral_speed)
        else:
            time_to_path = float("inf")

        predicted_crossing = (
            moving_toward_path
            and 0.0 <= time_to_path < 3.0
        )

        risk_level = "NONE"

        if currently_in_path and actor["local_forward"] < 5.0:
            risk_level = "EMERGENCY"

        elif currently_in_path and actor["local_forward"] < 10.0:
            risk_level = "BRAKE"

        elif predicted_crossing and actor["local_forward"] < 12.0:
            risk_level = "BRAKE"

        elif currently_in_path or predicted_crossing:
            risk_level = "CAUTION"

        if risk_level != "NONE":
            candidate = {
                **actor,
                "risk_level": risk_level,
                "currently_in_path": currently_in_path,
                "predicted_crossing": predicted_crossing,
                "time_to_path": time_to_path,
            }

            if best_risk is None:
                best_risk = candidate
            elif candidate["local_forward"] < best_risk["local_forward"]:
                best_risk = candidate

    return best_risk 
# Ova funkcija procjenjuje rizik u raskrižju
def evaluate_cross_traffic_risk(
    actor_facts,
    ego_vehicle,
    prediction_horizon=4.0,
    ego_in_curve=False
):
    """
    Procjenjuje rizik sudara s vozilima koja dolaze sa strane
    i presijecaju putanju ego vozila.

    Koristi constant-velocity procjenu vremena i udaljenosti
    u trenutku najbližeg prolaska.

    ego_in_curve: kada je ego u zavoju, vozila iz suprotne trake (koja
    prate isti zavoj) izgledaju "poprečno" pa se lažno proglase rizikom.
    U zavoju zato koristimo MNOGO stroži prag: samo stvarno neizbježan
    i vrlo blizak sudar koči ego, inače se cross-traffic ignorira.
    """
    ego_location = ego_vehicle.get_location()
    ego_velocity = ego_vehicle.get_velocity()

    ego_radius = max(
        ego_vehicle.bounding_box.extent.x,
        ego_vehicle.bounding_box.extent.y
    )

    best_risk = None

    for actor_fact in actor_facts:
        if not actor_fact["is_vehicle"]:
            continue
        
        heading_difference = actor_fact["heading_difference"]
        # Cross-traffic su vozila koja se kreću poprečno prema ego vozilu.
        # Vozila istog ili potpuno suprotnog smjera ovdje ignoriramo.
        if heading_difference < 35.0 or heading_difference > 145.0:
            continue

        actor = actor_fact["actor"]
        actor_location = actor.get_location()
        actor_velocity = actor.get_velocity()

        actor_speed = (
            actor_velocity.x**2 +
            actor_velocity.y**2 +
            actor_velocity.z**2
        ) ** 0.5

        
        
        # Ako je vozilo jako bočno udaljeno, a ne kreće se prema našoj putanji,
        # ne tretiramo ga kao cross-traffic rizik.
        actor_lateral_speed = abs(actor_fact["actor_lateral_speed"])
        actor_side_offset = abs(actor_fact["local_right"])

        if actor_side_offset > 4.5 and actor_lateral_speed < 0.5:
            continue

        rel_px = actor_location.x - ego_location.x
        rel_py = actor_location.y - ego_location.y

        rel_vx = actor_velocity.x - ego_velocity.x
        rel_vy = actor_velocity.y - ego_velocity.y

        relative_speed_sq = rel_vx**2 + rel_vy**2

        if relative_speed_sq < 0.01:
            continue

        time_to_closest = -(
            rel_px * rel_vx +
            rel_py * rel_vy
        ) / relative_speed_sq

        if time_to_closest < 0.0 or time_to_closest > prediction_horizon:
            continue

        closest_x = rel_px + rel_vx * time_to_closest
        closest_y = rel_py + rel_vy * time_to_closest

        closest_distance = (
            closest_x**2 +
            closest_y**2
        ) ** 0.5

        actor_radius = max(
            actor.bounding_box.extent.x,
            actor.bounding_box.extent.y
        )

        safe_distance = ego_radius + actor_radius + 0.8

        risk_level = "NONE"

        if ego_in_curve:
            # U ZAVOJU: samo stvarno neizbježan sudar (vrlo blizu i vrlo skoro).
            # Ovo sprječava lažno kočenje na vozila iz suprotne trake u zavoju.
            if closest_distance < safe_distance - 0.3 and time_to_closest < 0.7:
                risk_level = "EMERGENCY"
            # Nema BRAKE/CAUTION u zavoju - prečesto je lažno.
        else:
            if closest_distance < safe_distance and time_to_closest < 1.0:
                risk_level = "EMERGENCY"

            elif closest_distance < safe_distance + 0.8 and time_to_closest < 2.5:
                risk_level = "BRAKE"

            elif closest_distance < safe_distance + 1.5 and time_to_closest < 4.0:
                risk_level = "CAUTION"

        if risk_level == "NONE":
            continue

        candidate = {
            **actor_fact,
            "risk_level": risk_level,
            "time_to_closest": time_to_closest,
            "closest_distance": closest_distance,
        }

        if best_risk is None:
            best_risk = candidate
        elif candidate["time_to_closest"] < best_risk["time_to_closest"]:
            best_risk = candidate

    return best_risk


# ==========================================
# RASKRIŽJE: PROCJENA NAMJERE SKRETANJA
# ==========================================
def estimate_turning_intent(yaw_error, straight_threshold=12.0):
    """
    Procjenjuje namjeru ego vozila iz yaw_error.
    Koristi se kao fallback kada ruta nije dostupna.
    """
    if abs(yaw_error) < straight_threshold:
        return "STRAIGHT"
    elif yaw_error > 0.0:
        return "LEFT"
    else:
        return "RIGHT"


def estimate_turning_intent_from_route(
    ego_x, ego_y, ego_yaw, route_waypoints,
    lookahead_m=12.0, straight_threshold=15.0
):
    """
    Procjenjuje namjeru ego vozila iz RUTE (XML waypointa), ne iz mape.

    Ključna razlika: current_wp.next() slijedi MAPU i na raskrižju uvijek
    ide ravno. Ali ruta može tražiti skretanje. Ova funkcija gleda kuda
    ruta NAMJERAVA ići i na temelju toga određuje STRAIGHT/LEFT/RIGHT.

    1. Nađe najbliži route waypoint ispred ega.
    2. Gleda route waypoint koji je ~lookahead_m ispred na ruti.
    3. Računa razliku yaw-a između ega i tog waypinta.
    """
    if not route_waypoints:
        return "STRAIGHT", 0.0

    # Nađi najbliži route waypoint do ega.
    best_idx = 0
    best_dist_sq = float("inf")
    for i, wp in enumerate(route_waypoints):
        dx = wp["x"] - ego_x
        dy = wp["y"] - ego_y
        d2 = dx * dx + dy * dy
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_idx = i

    # Gledaj unaprijed po ruti za ~lookahead_m.
    accumulated = 0.0
    target_idx = best_idx
    for i in range(best_idx, len(route_waypoints) - 1):
        dx = route_waypoints[i + 1]["x"] - route_waypoints[i]["x"]
        dy = route_waypoints[i + 1]["y"] - route_waypoints[i]["y"]
        accumulated += (dx * dx + dy * dy) ** 0.5
        target_idx = i + 1
        if accumulated >= lookahead_m:
            break

    if target_idx == best_idx:
        return "STRAIGHT", 0.0

    route_target_yaw = route_waypoints[target_idx]["yaw"]

    # Normaliziraj razliku kuta.
    dyaw = route_target_yaw - ego_yaw
    while dyaw > 180.0:
        dyaw -= 360.0
    while dyaw < -180.0:
        dyaw += 360.0

    if abs(dyaw) < straight_threshold:
        return "STRAIGHT", dyaw
    elif dyaw > 0.0:
        return "LEFT", dyaw
    else:
        return "RIGHT", dyaw


def pure_pursuit_junction_steer(
    ego_x, ego_y, ego_yaw, route_waypoints,
    lookahead_m=8.0, steer_gain=0.55, max_steer=0.50
):
    """
    Pure-pursuit upravljanje kroz raskrižje.

    Cilja točku na RUTI ~lookahead_m ispred ega i računa volan iz kuta do te
    točke u koordinatnom sustavu vozila. Time vodi I poziciju I smjer
    istovremeno: kako se auto poravna s rutom, kut (alpha) pada i volan se sam
    gasi - bez odvojene "skreni" i "centriraj" faze i bez oslanjanja na
    lateral_error (koji je u raskrižju nepouzdan jer referenca skače).

    Vraća (steer, alpha_deg, target_x, target_y).
    CARLA konvencija: steer > 0 = desno, < 0 = lijevo.
    """
    if not route_waypoints:
        return 0.0, 0.0, ego_x, ego_y

    # 1. Najbliži route waypoint do ega.
    best_idx = 0
    best_d2 = float("inf")
    for i, wp in enumerate(route_waypoints):
        dx = wp["x"] - ego_x
        dy = wp["y"] - ego_y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i

    # 2. Ciljna točka ~lookahead_m naprijed po ruti.
    accumulated = 0.0
    target_idx = best_idx
    for i in range(best_idx, len(route_waypoints) - 1):
        dx = route_waypoints[i + 1]["x"] - route_waypoints[i]["x"]
        dy = route_waypoints[i + 1]["y"] - route_waypoints[i]["y"]
        accumulated += (dx * dx + dy * dy) ** 0.5
        target_idx = i + 1
        if accumulated >= lookahead_m:
            break

    target_x = route_waypoints[target_idx]["x"]
    target_y = route_waypoints[target_idx]["y"]

    # 3. Vektor do cilja, transformiran u koordinatni sustav vozila.
    dx = target_x - ego_x
    dy = target_y - ego_y
    yaw_rad = math.radians(ego_yaw)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    forward = dx * cos_y + dy * sin_y          # komponenta naprijed
    right = -dx * sin_y + dy * cos_y           # komponenta desno (+)

    # 4. Kut do cilja: + = cilj desno -> skreni desno (CARLA steer > 0).
    alpha = math.atan2(right, forward)         # radijani
    steer = steer_gain * alpha
    if steer > max_steer:
        steer = max_steer
    elif steer < -max_steer:
        steer = -max_steer

    return steer, math.degrees(alpha), target_x, target_y


# ==========================================
# RASKRIŽJE: PROCJENA RIZIKA PRESIJECANJA PUTANJE
# ==========================================
def evaluate_intersection_priority_risk(
    actor_facts,
    ego_vehicle,
    ego_in_junction,
    junction_ahead,
    ego_turning_intent,
    route_waypoints=None,
    prediction_horizon=8.5
):
    """
    Zaseban safety sloj SAMO za raskrižje.

    Aktivira se samo kada:
      - ego je u raskrižju ILI mu se raskrižje približava, I
      - ego SKREĆE (LEFT ili RIGHT) -> tada nema prednost.
    Kada ego ide RAVNO kroz raskrižje, ima prednost i ovaj sloj
    se ne aktivira (vraća None), pa se stabilna logika ne dira.

    Procjenjuje postoji li vozilo koje prolazi kroz raskrižje i može
    presjeći ego putanju. Koristi constant-velocity procjenu
    (time_to_closest / closest_distance).

    Vraća dict s "risk_level" iz {"YIELD", "STOP"} ili None.
    """
    # Aktivacija kada je ego u stvarnom raskrižju ILI mu se približava, i skreće.
    # Sada je sigurno koristiti junction_ahead jer intent dolazi iz RUTE
    # (ne iz yaw_error koji je bio nepouzdan u zavoju).
    if not (ego_in_junction or junction_ahead):
        return None
    if ego_turning_intent == "STRAIGHT":
        return None

    ego_location = ego_vehicle.get_location()
    ego_velocity = ego_vehicle.get_velocity()

    ego_radius = max(
        ego_vehicle.bounding_box.extent.x,
        ego_vehicle.bounding_box.extent.y
    )

    best_risk = None

    # Tocke ego RUTE ispred ega (~12 m) = prostor kroz koji ce ego proci u
    # raskrizju. Drugo vozilo je konflikt SAMO ako upadne na OVU putanju.
    path_points = []
    if route_waypoints:
        nearest_i = 0
        nearest_d2 = float("inf")
        for idx, _wp in enumerate(route_waypoints):
            d2 = (_wp["x"] - ego_location.x) ** 2 + (_wp["y"] - ego_location.y) ** 2
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_i = idx
        accumulated = 0.0
        for idx in range(nearest_i, len(route_waypoints) - 1):
            path_points.append((route_waypoints[idx]["x"], route_waypoints[idx]["y"]))
            accumulated += (
                (route_waypoints[idx + 1]["x"] - route_waypoints[idx]["x"]) ** 2 +
                (route_waypoints[idx + 1]["y"] - route_waypoints[idx]["y"]) ** 2
            ) ** 0.5
            if accumulated > 12.0:
                break

    for actor_fact in actor_facts:
        if not actor_fact["is_vehicle"]:
            continue

        actor = actor_fact["actor"]
        heading_difference = actor_fact["heading_difference"]

        # Zanimaju nas vozila koja presijecaju: dolaze bočno ili iz suprotnog
        # smjera. Vozila istog smjera (mala heading razlika) nisu konflikt
        # presijecanja - njih pokriva postojeća lead-vehicle logika.
        if heading_difference <= 30.0:
            continue

        actor_velocity = actor.get_velocity()
        actor_speed = (
            actor_velocity.x**2 +
            actor_velocity.y**2 +
            actor_velocity.z**2
        ) ** 0.5

        # Drugo vozilo mora se stvarno kretati kroz raskrižje.
        if actor_speed < 0.5:
            continue

        actor_location = actor.get_location()

        rel_px = actor_location.x - ego_location.x
        rel_py = actor_location.y - ego_location.y

        # Vozilo mora biti dovoljno blizu da uopće bude relevantno za raskrižje.
        # Daleko vozilo na nekoj drugoj cesti ne smije izazvati propuštanje.
        current_distance = (rel_px**2 + rel_py**2) ** 0.5
        if current_distance > 30.0:
            continue

        # ---- KONFLIKT PREMA EGO PUTANJI (ne prema trenutnoj poziciji ega) ----
        # Pitanje nije "koliko je vozilo blizu mene SADA" nego "hoce li vozilo
        # upasti na putanju kojom cu ja proci kroz raskrizje". Mimoilazenje u
        # susjednoj/suprotnoj traci (vozilo NE ulazi na ego rutu) se NE broji,
        # a stvarni lijevi skret preko suprotne trake (vozilo ide ravno kroz
        # tocku ego putanje) se broji ispravno.
        ego_w = ego_vehicle.bounding_box.extent.y
        actor_w = actor.bounding_box.extent.y
        conflict_radius = ego_w + actor_w + 0.6

        time_to_closest = None
        closest_distance = float("inf")

        if path_points:
            t = 0.0
            while t <= prediction_horizon:
                ax = actor_location.x + actor_velocity.x * t
                ay = actor_location.y + actor_velocity.y * t
                dmin = float("inf")
                for (px, py) in path_points:
                    d = ((ax - px) ** 2 + (ay - py) ** 2) ** 0.5
                    if d < dmin:
                        dmin = d
                if dmin < conflict_radius:
                    time_to_closest = t
                    closest_distance = dmin
                    break
                t += 0.25

        if time_to_closest is None:
            continue

        # STOP ako je vozilo skoro na putanji; YIELD ako ima nesto vremena.
        risk_level = "STOP" if time_to_closest < 8.0 else "YIELD"

        candidate = {
            **actor_fact,
            "risk_level": risk_level,
            "time_to_closest": time_to_closest,
            "closest_distance": closest_distance,
        }

        # STOP ima prioritet nad YIELD; inače biramo najbliži po vremenu.
        if best_risk is None:
            best_risk = candidate
        else:
            prio = {"YIELD": 1, "STOP": 2}
            if prio[candidate["risk_level"]] > prio[best_risk["risk_level"]]:
                best_risk = candidate
            elif (
                prio[candidate["risk_level"]] == prio[best_risk["risk_level"]]
                and candidate["time_to_closest"] < best_risk["time_to_closest"]
            ):
                best_risk = candidate

    return best_risk


def main():

    # Glavni prekidač za usporedbu: True = baseline + logički shield sloj,
    # False = ČISTI baseline (neat_neat bez shielda). Ista ruta/kod/mjerenja.
    SHIELD_ENABLED = False
    # Ablacija gap-acceptancea (S7): True = shield propusta (normalno),
    # False = shield skrece ali NE propusta -> ulijece u nasuprotno vozilo.
    # Djeluje samo kad je SHIELD_ENABLED=True; ostatak shielda netaknut.
    GAP_ACCEPTANCE_ENABLED = True

    # --- Kontrolirani scenarij S7: nasuprotno vozilo koje presijeca ego skret ---
    # Stoji na nasuprotnom prilazu dok ga ego ne "okine" (priblizi se raskrizju),
    # pa krene RAVNO kroz raskrizje. Sluzi za ugadjanje gap-acceptance pravila.
    SPAWN_ONCOMING = False         # True = ukljuci nasuprotno vozilo (S7 kalibracija; vrati na False za normalne voznje)
    ONCOMING_SPAWN_INDEX = 45      # spawn na suprotnoj traci, sjeverno od raskrizja (presijeca ego lijevi skret)
    ONCOMING_JUNCTION_WP = 26      # waypoint skreta rute na (0.5,191.5)
    ONCOMING_TRIGGER_DIST = 14.0   # ego unutar ove udaljenosti od raskrizja -> pusti vozilo
    ONCOMING_SPEED = float(os.environ.get("ONCOMING_SPEED", 8.0))  # m/s konstantna brzina; override: set ONCOMING_SPEED=6

    # --- Kontrolirani scenarij S5: sporo/stojece vozilo U ISTOJ TRACI ispred ega ---
    # Na ravnom prilazu PRIJE skreta (baseline tu vozi normalno) -> cist test koci li ego.
    SPAWN_LEAD = False            # True = ukljuci vozilo ispred (S5)
    LEAD_SPAWN_WP = 18             # route waypoint za spawn (wp ~2 m; 18 -> ~35 m ispred ega)
    LEAD_SPEED = float(os.environ.get("LEAD_SPEED", 0.0))  # 0 = stoji; >0 = konstantna spora brzina (m/s)

    # --- Kontrolirani scenarij S6: pjesak na ego putanji ---
    SPAWN_PEDESTRIAN = True        # True = ukljuci pjesaka (S6)
    PED_SPAWN_WP = 14              # route waypoint oko kojeg se pjesak postavlja (~27 m ispred)
    PED_SIDE_OFFSET = -4.0         # poprecni pomak od sredine trake (m); 0 = stoji u traci
    PED_TRIGGER_DIST = 12.0        # ego unutar ove udaljenosti -> pjesak krene prelaziti
    PED_SPEED = float(os.environ.get("PED_SPEED", 1.5))   # m/s; 0 = samo stoji u traci

    HOST_IP = "localhost"
    client = carla.Client(HOST_IP, 2000)
    client.set_timeout(30.0)  # bilo je na 10
    client.load_world("Town02")

    synchronous_master = False
    pcla = None
    settings = None
    vehicle = None
    world = None
    collision_sensor = None
    traffic_manager = None
    oncoming_vehicle = None
    lead_vehicle = None
    pedestrian = None

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)
        # Fiksni seed -> ponovljivo ponašanje prometa (isto u baseline i shield).
        traffic_manager.set_random_device_seed(42)

        settings = world.get_settings()
        asynch = False

        if not asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
                    you could experience some issues. If it's not working correctly, switch to \
                    synchronous mode by using traffic_manager.set_synchronous_mode(True)")

        # settings.no_rendering_mode = True
        world.apply_settings(settings)

        # Finding actors
        bpLibrary = world.get_blueprint_library()

        # Finding vehicle
        vehicleBP = bpLibrary.filter('model3')[0]

        vehicle_spawn_points = world.get_map().get_spawn_points()

        # Spawn vehicle
        vehicle = world.spawn_actor(vehicleBP, vehicle_spawn_points[0]) # bilo je 31 izmjenjeno u 0
        # vehicle = world.spawn_actor(vehicleBP, vehicle_spawn_points[6])   # bilo [0], raskrizja ruta
        
        # Kamera koja prati vozilo
        spectator = world.get_spectator()

        # Collision sensor
        collision_events = []

        collision_bp = bpLibrary.find("sensor.other.collision")

        collision_sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )

        def on_collision(event):
            other_actor = event.other_actor

            collision_info = {
                "frame": event.frame,
                "other_actor_id": other_actor.id,
                "other_actor_type": other_actor.type_id,
                "impulse_x": event.normal_impulse.x,
                "impulse_y": event.normal_impulse.y,
                "impulse_z": event.normal_impulse.z,
            }

            collision_events.append(collision_info)

            ego_loc = vehicle.get_location()
            ego_tf = vehicle.get_transform()

            collision_line = (
                f"COLLISION: frame={event.frame} "
                f"type={other_actor.type_id} "
                f"id={other_actor.id} "
                f"x={ego_loc.x:.2f} "
                f"y={ego_loc.y:.2f} "
                f"yaw={ego_tf.rotation.yaw:.2f} "
                f"impulse=({event.normal_impulse.x:.2f}, "
                f"{event.normal_impulse.y:.2f}, "
                f"{event.normal_impulse.z:.2f})"
            )

            print(collision_line)


        collision_sensor.listen(on_collision)


        world.tick()

        agent = "neat_neat"  # bio je "simlingo_simlingo" ali ne radi kako treba
        # route = "./route_intersection_test_1.xml"  # ruta sa raskrižjem
        route = "./route_spawn0.xml"
        pcla = PCLA(agent, vehicle, route, client)

        # ==========================================
        # PARSIRANJE RUTE IZ XML-a za detekciju namjere skretanja
        # ==========================================
        # Čitamo waypointe iz XML-a da znamo kuda ruta NAMJERAVA ići.
        # To je nužno jer current_wp.next(8.0) slijedi MAPU (uvijek ravno
        # na raskrižju), a ruta može tražiti skretanje.
        import xml.etree.ElementTree as ET
        route_waypoints = []
        try:
            _tree = ET.parse(route)
            for _wp_elem in _tree.getroot().findall("waypoint"):
                route_waypoints.append({
                    "x": float(_wp_elem.get("x")),
                    "y": float(_wp_elem.get("y")),
                    "yaw": float(_wp_elem.get("yaw")),
                })
            print(f"Route parsed: {len(route_waypoints)} waypoints loaded.")
        except Exception as _e:
            print(f"WARNING: Could not parse route XML: {_e}")
            route_waypoints = []

        
        # Zasebni PID kontroler za održavanje smjera trake tijekom collision kočenja.
        # Ne koristi isti kontroler kao recovery jer PID interno pamti prethodne pogreške.
        collision_lane_controller = VehiclePIDController(
            vehicle,
            args_lateral={'K_P': 2.40, 'K_D': 0.30, 'K_I': 0.02, 'dt': 0.05},
            args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.03, 'dt': 0.05},
            offset=0.00,
            max_throttle=0.0,
            max_brake=1.0,
            max_steering=0.55
        )


    
        print('\nSpawned the vehicle with model =', agent, ', press Ctrl+C to exit.\n')

        step = 0

        # CSV logger za usporedbu baseline vs shield (jedan redak po koraku).
        if not SHIELD_ENABLED:
            _mode = "baseline"
        elif not GAP_ACCEPTANCE_ENABLED:
            _mode = "shieldnogap"
        else:
            _mode = "shield"
        if SPAWN_ONCOMING:
            _csv_name = f"metrics_{_mode}_s7_v{int(round(ONCOMING_SPEED))}.csv"
        elif SPAWN_LEAD:
            _csv_name = f"metrics_{_mode}_s5.csv"
        elif SPAWN_PEDESTRIAN:
            _csv_name = f"metrics_{_mode}_s6.csv"
        else:
            _csv_name = f"metrics_{_mode}.csv"
        _csv = open(_csv_name, "w", encoding="utf-8")
        _csv.write(
            "step,mode,x,y,yaw,speed,raw_steer,steer,raw_throttle,throttle,"
            "raw_brake,brake,lateral_error,yaw_error,collision_count,"
            "intersection_risk,ego_turning_intent,route_intent,"
            "lane_recenter,junction_route_steer\n"
        )
        
        last_collision_count = 0

        # State za stabilnije držanje trake nakon aktivacije.
        lane_recenter_was_active = False
        last_shield_steer = 0.0
        prev_lateral_error = 0.0
        prev_final_steer = 0.0
        recenter_hold_ticks = 0
        turn_start_yaw = None
        turn_entry_yaw = None  # yaw na ulasku u skret (za napredak skreta)
        turn_planned_angle = 0.0
        pp_was_active = False
        exit_recenter_counter = 0

        # --- S7: kontrolirani spawn nasuprotnog vozila (stoji dok ga ego ne okine) ---
        oncoming_released = False
        if SPAWN_ONCOMING:
            try:
                _onc_bp = bpLibrary.filter("vehicle.tesla.model3")[0]
                _onc_tf = vehicle_spawn_points[ONCOMING_SPAWN_INDEX]
                oncoming_vehicle = world.try_spawn_actor(_onc_bp, _onc_tf)
                if oncoming_vehicle is not None:
                    oncoming_vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    print(f"S7: nasuprotno vozilo spawnano na spawn[{ONCOMING_SPAWN_INDEX}] loc={_onc_tf.location}")
                else:
                    print(f"S7: spawn nasuprotnog vozila NIJE uspio (zauzeto?) idx={ONCOMING_SPAWN_INDEX}")
            except Exception as _e:
                print(f"S7: greska pri spawnu nasuprotnog vozila: {_e}")

        # --- S5: kontrolirani spawn vozila ispred (ista traka, ravni prilaz) ---
        if SPAWN_LEAD and route_waypoints:
            try:
                _lw = route_waypoints[min(LEAD_SPAWN_WP, len(route_waypoints) - 1)]
                _lead_tf = carla.Transform(
                    carla.Location(x=_lw["x"], y=_lw["y"], z=0.6),
                    carla.Rotation(yaw=_lw["yaw"]))
                _lead_bp = bpLibrary.filter("vehicle.tesla.model3")[0]
                lead_vehicle = world.try_spawn_actor(_lead_bp, _lead_tf)
                if lead_vehicle is not None:
                    lead_vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    print(f"S5: vozilo ispred spawnano na wp[{LEAD_SPAWN_WP}] loc={_lead_tf.location}")
                else:
                    print(f"S5: spawn vozila ispred NIJE uspio (zauzeto?) wp={LEAD_SPAWN_WP}")
            except Exception as _e:
                print(f"S5: greska pri spawnu vozila ispred: {_e}")

        # --- S6: kontrolirani spawn pjesaka (miruje dok ga ego ne okine) ---
        pedestrian_released = False
        if SPAWN_PEDESTRIAN and route_waypoints:
            try:
                _pw = route_waypoints[min(PED_SPAWN_WP, len(route_waypoints) - 1)]
                _ped_tf = carla.Transform(
                    carla.Location(x=_pw["x"] + PED_SIDE_OFFSET, y=_pw["y"], z=1.0),
                    carla.Rotation(yaw=_pw["yaw"]))
                _ped_bp = bpLibrary.filter("walker.pedestrian.*")[0]
                if _ped_bp.has_attribute("is_invincible"):
                    _ped_bp.set_attribute("is_invincible", "false")
                pedestrian = world.try_spawn_actor(_ped_bp, _ped_tf)
                if pedestrian is not None:
                    print(f"S6: pjesak spawnan kod wp[{PED_SPAWN_WP}] loc={_ped_tf.location}")
                else:
                    print(f"S6: spawn pjesaka NIJE uspio (zauzeto?) wp={PED_SPAWN_WP}")
            except Exception as _e:
                print(f"S6: greska pri spawnu pjesaka: {_e}")

       
        while True:
            try:
                ego_action = pcla.get_action() # Ovdje dobivamo kontrolu od baseline agenta
                
                raw_steer = ego_action.steer
                raw_throttle = ego_action.throttle
                raw_brake = ego_action.brake

                vel = vehicle.get_velocity()
                speed_mps = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

                vehicle_location = vehicle.get_location()
                vehicle_transform = vehicle.get_transform()
                # ==========================================
                # FACTS EXTRACTION: DYNAMIC TRAFFIC ACTORS
                # ==========================================
                dynamic_actor_facts = extract_dynamic_actor_facts(
                    world,
                    vehicle,
                    max_forward=30.0,
                    max_side=15.0
                )

                vehicle_collision_risk = evaluate_vehicle_collision_risk(
                    dynamic_actor_facts,
                    vehicle
                )

                # ==========================================
                # DETEKCIJA ZAVOJA (prije cross-traffic procjene)
                # ==========================================
                # Kada je ego u ZAVOJU, on je zakrenut u odnosu na cestu, pa
                # vozila iz suprotne trake (koja prate isti zavoj) izgledaju
                # "poprečno" i cross_traffic ih lažno proglasi rizikom -> ego stane.
                # Zato unaprijed izračunamo je li ego u zavoju, da cross_traffic
                # NE koči zbog vozila koja samo prolaze kroz isti zavoj.
                ego_in_curve = False
                try:
                    _cwp = world.get_map().get_waypoint(
                        vehicle_location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    if _cwp is not None:
                        _ahead = _cwp.next(8.0)
                        if _ahead:
                            _dyaw = _ahead[0].transform.rotation.yaw - _cwp.transform.rotation.yaw
                            while _dyaw > 180.0:
                                _dyaw -= 360.0
                            while _dyaw < -180.0:
                                _dyaw += 360.0
                            # Zavoj: smjer ceste se mijenja > 15° na 8m.
                            if abs(_dyaw) > 15.0:
                                ego_in_curve = True
                except Exception:
                    ego_in_curve = False

                cross_traffic_risk = evaluate_cross_traffic_risk(
                    dynamic_actor_facts,
                    vehicle,
                    ego_in_curve=ego_in_curve
                )

                pedestrian_collision_risk = evaluate_pedestrian_collision_risk(
                    dynamic_actor_facts,
                    vehicle
                )

                collision_risk_active = (
                    vehicle_collision_risk is not None or
                    cross_traffic_risk is not None or
                    pedestrian_collision_risk is not None
                )
                
                new_physical_collision = len(collision_events) > last_collision_count
                
                traffic_light_state = "NONE"

                try:
                    if vehicle.is_at_traffic_light():
                        traffic_light = vehicle.get_traffic_light()
                        if traffic_light is not None:
                            traffic_light_state = str(traffic_light.get_state())
                except Exception:
                    traffic_light_state = "ERR"

                facts = {
                    "speed_mps": speed_mps,
                    "steer": ego_action.steer,
                    "throttle": ego_action.throttle,
                    "brake": ego_action.brake,
                    "x": vehicle_location.x,
                    "y": vehicle_location.y,
                    "yaw": vehicle_transform.rotation.yaw,

                    "dynamic_actor_count": len(dynamic_actor_facts),
                    "vehicle_collision_risk": vehicle_collision_risk,
                    "pedestrian_collision_risk": pedestrian_collision_risk,
                    "collision_risk_active": collision_risk_active,
                    "cross_traffic_risk": cross_traffic_risk,
                }

                # S7: pusti nasuprotno vozilo kad ego prilazi raskrizju; ono ide RAVNO.
                if oncoming_vehicle is not None and route_waypoints:
                    _jwp = route_waypoints[min(ONCOMING_JUNCTION_WP, len(route_waypoints) - 1)]
                    _dj = ((facts["x"] - _jwp["x"]) ** 2 + (facts["y"] - _jwp["y"]) ** 2) ** 0.5
                    if _dj < ONCOMING_TRIGGER_DIST:
                        oncoming_released = True
                    if oncoming_released:
                        # Konstantna, predvidljiva brzina (NE ubrzavanje iz mirovanja),
                        # da ego moze tocno procijeniti razmak.
                        _fwd = oncoming_vehicle.get_transform().get_forward_vector()
                        oncoming_vehicle.set_target_velocity(
                            carla.Vector3D(_fwd.x * ONCOMING_SPEED, _fwd.y * ONCOMING_SPEED, 0.0))
                    else:
                        oncoming_vehicle.apply_control(carla.VehicleControl(brake=1.0))

                # S5: vozilo ispred - stoji (brake) ili konstantna spora brzina.
                if lead_vehicle is not None:
                    if LEAD_SPEED > 0.0:
                        _lf = lead_vehicle.get_transform().get_forward_vector()
                        lead_vehicle.set_target_velocity(
                            carla.Vector3D(_lf.x * LEAD_SPEED, _lf.y * LEAD_SPEED, 0.0))
                    else:
                        lead_vehicle.apply_control(carla.VehicleControl(brake=1.0))

                # S6: pjesak miruje dok ego ne priblizi, pa prelazi preko ego trake.
                if pedestrian is not None and route_waypoints:
                    _pw2 = route_waypoints[min(PED_SPAWN_WP, len(route_waypoints) - 1)]
                    _dp = ((facts["x"] - _pw2["x"]) ** 2 + (facts["y"] - _pw2["y"]) ** 2) ** 0.5
                    if _dp < PED_TRIGGER_DIST:
                        pedestrian_released = True
                    if pedestrian_released and PED_SPEED > 0.0:
                        _cross = 1.0 if PED_SIDE_OFFSET < 0 else -1.0
                        pedestrian.apply_control(carla.WalkerControl(
                            direction=carla.Vector3D(x=_cross, y=0.0, z=0.0),
                            speed=PED_SPEED))
                    else:
                        pedestrian.apply_control(carla.WalkerControl(speed=0.0))

               
                
                # =======================================
                # SHIELD 2: LIMIT STEERING AT HIGH SPEED
                # =======================================
                # Ako vozilo ide brzo, ne dopušta se velik kut volana.
                if facts["speed_mps"] > 8.0:
                    max_steer = 0.25
                    if ego_action.steer > max_steer:
                        print("SHIELD: high-speed steering too large -> clamping right steer")
                        ego_action.steer = max_steer
                    elif ego_action.steer < -max_steer:
                        print("SHIELD: high-speed steering too large -> clamping left steer")
                        ego_action.steer = -max_steer

                
                # ==========================================
                # LOCAL LANE / WAYPOINT FACTS
                # ==========================================
             
                current_wp = world.get_map().get_waypoint(
                    vehicle.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )

                if current_wp is None:
                    print("SHIELD: no driving waypoint found")

                    if collision_risk_active:
                        print("SHIELD: no waypoint + collision risk -> emergency brake")
                        ego_action.throttle = 0.0
                        ego_action.brake = 1.0

                    vehicle.apply_control(ego_action)
                    world.tick()

                    # Kamera prati vozilo odozgo
                    transform = vehicle.get_transform()
                    spectator.set_transform(
                        carla.Transform(
                            transform.location + carla.Location(x=0, y=0, z=20),
                            carla.Rotation(pitch=-90, yaw=0, roll=0)
                        )
                    )

                    step += 1
                    continue

                
                
                # ==========================================
                # WAYPOINT / LANE DIAGNOSTICS
                # ==========================================
                wp_loc = current_wp.transform.location
                wp_yaw = current_wp.transform.rotation.yaw

                wp_lane_id = current_wp.lane_id
                wp_road_id = current_wp.road_id
                wp_section_id = current_wp.section_id
                wp_lane_width = current_wp.lane_width
                wp_is_junction = current_wp.is_junction
                
                current_yaw = vehicle.get_transform().rotation.yaw
                veh_loc = vehicle.get_transform().location

                
                if speed_mps < 1.0:
                    lookahead_distance = 4.0
                elif speed_mps < 2.0:
                    lookahead_distance = 6.0
                else:
                    lookahead_distance = 8.0

                next_wps = current_wp.next(lookahead_distance)

                if len(next_wps) > 0:
                    pid_target_wp = next_wps[0]
                else:
                    pid_target_wp = None
                    
                yaw_error = 0.0
                lateral_error = 0.0

                # Yaw gledamo prema waypointu ispred vozila,
                # jer time znamo dolazi li zavoj.
                if pid_target_wp is not None:
                    target_yaw = pid_target_wp.transform.rotation.yaw
                    yaw_error = normalize_angle_deg(target_yaw - current_yaw)

                # Lateralno centriranje NE smije koristiti lookahead waypoint.
                # Centar trake mora biti trenutni current_wp, inače vozilo u zavoju
                # počne ciljati točku predaleko ispred i bježi prema rubu/trotoaru.
                lane_center = current_wp.transform.location

                dx = lane_center.x - veh_loc.x
                dy = lane_center.y - veh_loc.y

                wp_right = current_wp.transform.get_right_vector()

                lateral_error = (
                    dx * wp_right.x +
                    dy * wp_right.y
                )

                # ==========================================
                # INTERSECTION: DETEKCIJA + NAMJERA + RIZIK
                # ==========================================
                # Zaseban sigurnosni sloj za raskrižje.
                # Ne dira postojeću lane/curve logiku.
                #
                # VAŽNO: zavoj NIJE raskrižje. U CARLA mapama su i zavoji često
                # označeni s is_junction=True, pa se intersection sloj prije lažno
                # palio u zavojima (ego je stajao u zavoju). Zato koristimo strožu
                # detekciju: pravo raskrižje ima junction objekt koji povezuje
                # VIŠE cesta (3+), dok zavoj na jednoj cesti ima 1-2 road_id-a.
                def is_real_junction(wp):
                    if wp is None or not wp.is_junction:
                        return False
                    junction = wp.get_junction()
                    if junction is None:
                        return False
                    try:
                        wps_pairs = junction.get_waypoints(carla.LaneType.Driving)
                    except Exception:
                        return False
                    road_ids = set()
                    for entry_wp, exit_wp in wps_pairs:
                        road_ids.add(entry_wp.road_id)
                        road_ids.add(exit_wp.road_id)
                    # Zavoj: 1-2 ceste. Pravo raskrižje: 3+.
                    return len(road_ids) >= 3

                ego_in_junction = is_real_junction(current_wp)

                def distance_to_real_junction_ahead(start_wp, max_distance=10.0, step_distance=1.0):
                    """
                    Vraća približnu udaljenost do ulaska u PRAVO raskrižje.
                    Razlog: pid_target_wp na 6-8 m može već biti u raskrižju,
                    ali ego je još na ravnom prilazu. Tada NE smijemo pustiti
                    full raw_steer jer auto skrene prerano prema rubu/semaforu.
                    """
                    if start_wp is None:
                        return float("inf")

                    if is_real_junction(start_wp):
                        return 0.0

                    frontier = [start_wp]
                    visited = set()
                    travelled = 0.0

                    while travelled < max_distance:
                        next_frontier = []

                        for wp in frontier:
                            try:
                                candidates = wp.next(step_distance)
                            except Exception:
                                candidates = []

                            for nxt in candidates:
                                if nxt is None:
                                    continue

                                key = (
                                    nxt.road_id,
                                    nxt.section_id,
                                    nxt.lane_id,
                                    round(getattr(nxt, "s", 0.0), 1)
                                )

                                if key in visited:
                                    continue

                                visited.add(key)

                                if is_real_junction(nxt):
                                    return travelled + step_distance

                                next_frontier.append(nxt)

                        if not next_frontier:
                            break

                        frontier = next_frontier[:6]
                        travelled += step_distance

                    return float("inf")

                junction_distance_to_entry = distance_to_real_junction_ahead(
                    current_wp,
                    max_distance=10.0,
                    step_distance=1.0
                )

                # Raskrižje ispred: gledamo waypoint na lookahead udaljenosti.
                junction_ahead = is_real_junction(pid_target_wp)

                # Release volana smije krenuti tek na samom ulazu u raskrižje.
                # Samo junction_ahead nije dovoljno jer se pali 4-8 m prerano.
                junction_turn_ready = (
                    ego_in_junction
                    or junction_distance_to_entry <= 2.0
                )

                # ==========================================
                # NAMJERA SKRETANJA: IZ RUTE, NE IZ MAPE
                # ==========================================
                # Ključni popravak: yaw_error (iz current_wp.next()) slijedi MAPU
                # i na raskrižju uvijek pokazuje RAVNO. Ali ruta iz XML-a može
                # tražiti skretanje. Zato na raskrižju (ili blizu njega) čitamo
                # namjeru iz RUTE, a na ravnom dijelu koristimo yaw_error kao fallback.
                current_yaw_for_route = vehicle.get_transform().rotation.yaw
                route_intent, route_dyaw = estimate_turning_intent_from_route(
                    vehicle_location.x,
                    vehicle_location.y,
                    current_yaw_for_route,
                    route_waypoints,
                    lookahead_m=12.0
                )

                if ego_in_junction or junction_ahead:
                    # Na/blizu raskrižja: koristi RUTU za intent.
                    ego_turning_intent = route_intent
                else:
                    # Izvan raskrižja: koristi mapu (yaw_error) kao i prije.
                    ego_turning_intent = estimate_turning_intent(yaw_error)

                # Dodatni fallback: ako je intent STRAIGHT ali baseline očito
                # pokušava skrenuti (raw_steer jak), koristi raw_steer kao signal.
                # Ovo pokriva slučaj kad ruta ne pokaže skretanje (npr. waypoint
                # matching je neprecizan) ali baseline agent jasno skreće.
                if (
                    junction_turn_ready
                    and ego_turning_intent == "STRAIGHT"
                    and abs(raw_steer) > 0.45
                ):
                    if raw_steer > 0.0:
                        ego_turning_intent = "RIGHT"
                    else:
                        ego_turning_intent = "LEFT"

                # Rizik se aktivira samo kad ego skreće u STVARNOM raskrižju
                # (ne u zavoju, i ne kad ide ravno).
                intersection_risk = evaluate_intersection_priority_risk(
                    dynamic_actor_facts,
                    vehicle,
                    ego_in_junction,
                    junction_ahead,
                    ego_turning_intent,
                    route_waypoints
                )

                # ==========================================
                # JUNCTION TURN RELEASE FLAGS
                # ==========================================
                # Ove zastavice moraju biti definirane PRIJE sidewalk/edge guarda,
                # jer se koriste za gašenje lažnog sidewalk guarda u raskrižju.
                in_junction_zone = ego_in_junction or junction_ahead

                # Za YIELD/STOP i log još znamo da je raskrižje ispred,
                # ali za puštanje volana koristimo užu zonu: samo neposredni ulaz.
                turning_in_junction_zone = (
                    junction_turn_ready
                    and ego_turning_intent != "STRAIGHT"
                )

                # Schmitt histereza: smjer se postavi na ±15, ali zadrži dok ne
                # padne ispod ±10. Na izlazu route_dyaw visi oko -15 pa bi se
                # route_turn_direction (i junction_route_steer) treperio -> trzaj
                # volana ±0.2. Wrap (|route_dyaw|>15) i sam skret (~-70) netaknuti.
                if route_dyaw > 15.0:
                    route_turn_direction = 1
                elif route_dyaw < -15.0:
                    route_turn_direction = -1
                elif abs(route_dyaw) < 10.0:
                    route_turn_direction = 0
                else:
                    # zona 10-15: zadrži prethodnu vrijednost (perzistira u petlji)
                    route_turn_direction = locals().get("route_turn_direction", 0)

                raw_steer_matches_route = (
                    route_turn_direction == 0
                    or raw_steer * route_turn_direction > 0.25
                )

                # Baseline raw_steer je mrtva konstanta (+1.000), pa "slaganje" s
                # rutom je slučajno i NE znači da baseline ispravno vozi. Nikad ga
                # ne puštamo u raskrižju - route-steer (SHIELD 3C) vodi sve skrete
                # i sam daje gas/kočnicu, pa nema lažnog zaustavljanja.
                junction_turn_release_active = False
               

                junction_route_steer_active = False
                junction_creep_active = False

                # ==========================================
                # MAP-BASED ROAD EDGE / SIDEWALK DETECTION
                # ==========================================
                # Ne koristimo kameru, nego CARLA mapu:
                # ako s jedne strane trenutne trake nema druge Driving trake,
                # tu stranu tretiramo kao rub ceste / mogući trotoar.

                left_lane = current_wp.get_left_lane()
                right_lane = current_wp.get_right_lane()

                left_is_driving = (
                    left_lane is not None
                    and left_lane.lane_type == carla.LaneType.Driving
                )

                right_is_driving = (
                    right_lane is not None
                    and right_lane.lane_type == carla.LaneType.Driving
                )

                left_is_road_edge = not left_is_driving
                right_is_road_edge = not right_is_driving

                # Po formuli lateral_error:
                # negativan lateral_error znači da je vozilo desno od centra trake,
                # pozitivan lateral_error znači da je vozilo lijevo od centra trake.
                
                near_right_sidewalk = (
                    lateral_error < -0.15
                    and right_is_road_edge
                    and speed_mps > 0.8
                )

                near_left_sidewalk = (
                    lateral_error > 0.15
                    and left_is_road_edge
                    and speed_mps > 0.8
                )

                # Prediktivna zaštita:
                # Ako baseline već jako okreće prema strani gdje je rub ceste,
                # ne čekamo da lateral_error naraste na 0.15+.
                baseline_pushing_toward_right_edge = (
                    right_is_road_edge
                    and lateral_error < -0.03
                    and raw_steer > 0.70
                    and speed_mps > 1.2
                )

                baseline_pushing_toward_left_edge = (
                    left_is_road_edge
                    and lateral_error > 0.03
                    and raw_steer < -0.70
                    and speed_mps > 1.2
                )

                baseline_pushing_toward_sidewalk = (
                    baseline_pushing_toward_right_edge or
                    baseline_pushing_toward_left_edge
                )

                sidewalk_guard_active = (
                    near_right_sidewalk or
                    near_left_sidewalk or
                    baseline_pushing_toward_sidewalk
                )

                # ==========================================
                # U RASKRIŽJU: UGASI SIDEWALK/EDGE DETEKCIJU
                # ==========================================
                # U CARLA raskrižju spojne trake NEMAJU susjedne Driving trake
                # (get_left_lane/get_right_lane vraćaju None), pa mapa lažno
                # označi obje strane kao "rub ceste". Uz raw_steer=1.000 (agent
                # skreće!) to okida baseline_pushing_toward_sidewalk i
                # sidewalk_guard, koji onda kroz lane_recenter iznimku BLOKIRA
                # skretanje. U raskrižju nema trotoara neposredno uz traku -
                # to je artefakt mape. Gasimo guard osim ako je lateralni
                # offset stvarno velik (vozilo zaista bježi s ceste).
                if turning_in_junction_zone:
                    sidewalk_guard_active = False
                    baseline_pushing_toward_sidewalk = False
                elif in_junction_zone and abs(lateral_error) < 0.45:
                    sidewalk_guard_active = False
                    baseline_pushing_toward_sidewalk = False
                # ==========================================
                # SHIELD 3: CURVE ENTRY ASSIST
                # ==========================================
                # Blaga pomoć baseline agentu kada dolazi očiti zavoj.
                # Ne koristi recovery timer i ne preuzima cijelu vožnju.
                # Cilj: spriječiti da baseline ravno uđe u zavoj i zatim zakoči.

                curve_assist_active = False

                red_or_yellow_light = (
                    "Red" in traffic_light_state or
                    "Yellow" in traffic_light_state
                )

                curve_ahead = abs(yaw_error) > 18.0
                too_far_from_lane_center = abs(lateral_error) > 0.12

                # Baseline je problematičan ako ne skreće dovoljno
                # ili skreće u suprotnom smjeru od zavoja.
                baseline_wrong_or_weak_steer = (
                    abs(raw_steer) < 0.16 or
                    (yaw_error * raw_steer < 0.0)
                )

                # NAPOMENA: curve_assist je onesposobljen.
                # Razlog: koristio je predznak yaw_error (od lookahead waypointa 8m ispred)
                # za određivanje smjera volana. Na prijelazu iz ravnog dijela u zavoj
                # taj predznak je NEPOUZDAN i davao je volan u KRIVOM smjeru
                # (npr. steer=-0.135 dok je zavoj zahtijevao +0.13), pa je vozilo
                # prvo skretalo na vanjski rub ("zakrene prelijevo"), a tek onda se ispravljalo.
                # lane_recenter ispod bolje upravlja zavojem jer prioritetno koristi lateral_error.
                if (
                    False
                    and not collision_risk_active
                    and len(collision_events) == 0
                    and not red_or_yellow_light
                    and curve_ahead
                    and abs(yaw_error) < 55.0
                    and not too_far_from_lane_center
                    and baseline_wrong_or_weak_steer
                    and speed_mps < 5.5
                    and not sidewalk_guard_active
                ):
                    curve_assist_active = True

                    if abs(yaw_error) > 45.0:
                        desired_curve_steer = yaw_error * 0.0038
                    else:
                        desired_curve_steer = yaw_error * 0.0050

                    if desired_curve_steer > 0.16:
                        desired_curve_steer = 0.16
                    elif desired_curve_steer < -0.16:
                        desired_curve_steer = -0.16

                    # Ako baseline skreće u suprotnom smjeru od zavoja,
                    # ne smijemo ga miješati s assistom jer oslabi korekciju.
                    baseline_steering_against_curve = (
                        yaw_error * raw_steer < 0.0
                    )

                    if baseline_steering_against_curve:
                        ego_action.steer = desired_curve_steer
                    else:
                        ego_action.steer = (
                            0.20 * ego_action.steer
                            + 0.80 * desired_curve_steer
                        )

                    if ego_action.steer > 0.16:
                        ego_action.steer = 0.16
                    elif ego_action.steer < -0.16:
                        ego_action.steer = -0.16

                    # Ako baseline bez razloga koči u zavoju, makni punu kočnicu,
                    # ali nemoj agresivno ubrzavati.
                    if raw_brake > 0.70:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.12

                    # U samom ulazu u zavoj smanji gas da skretanje bude mirnije.
                    if ego_action.throttle > 0.15:
                        ego_action.throttle = 0.15

                    print(
                        f"SHIELD: curve entry assist active "
                        f"(yaw_error={yaw_error:.2f}, "
                        f"raw_steer={raw_steer:.3f}, "
                        f"raw_brake={raw_brake:.3f}, "
                        f"assist_steer={ego_action.steer:.3f})"
                    )
                    
                    
                    
                    
                # ==========================================
                # SHIELD 3B: CURVE / LANE CENTER HOLD
                # ==========================================
                # Aktivira se kada vozilo počne gubiti centar trake
                # ili kada baseline u zavoju koči bez stvarnog rizika.
                # Cilj: držati vozilo u sredini trake i nastaviti kroz zavoj,
                # bez starog recovery PID sustava.

                lane_recenter_active = False

                # Centriranje i "road-edge" guard.
                # Ovo ne detektira trotoar kamerom, nego koristi geometriju trake:
                # ako je ego predaleko od centra trake, tretiramo ga kao rizik približavanja rubu.
                # Time se jednako sprječava odlazak prema trotoaru i odlazak prema suprotnoj traci.
                # Malo ranije uključivanje jer se na slikama vidi da vozilo
                # već na oko 0.30-0.40 m offseta ide vizualno preblizu rubu/trotoaru.
                RECENTER_ON_THRESHOLD = 0.13
                RECENTER_OFF_THRESHOLD = 0.04
                RECENTER_RELEASE_YAW_THRESHOLD = 1.0
                RECENTER_RELEASE_RAW_STEER_LIMIT = 0.45

                SOFT_EDGE_GUARD_THRESHOLD = 0.20
                HARD_EDGE_GUARD_THRESHOLD = 0.50

                lateral_error_delta = lateral_error - prev_lateral_error
                moving_away_from_center = (lateral_error * lateral_error_delta) > 0.0
                
                # Kratko zadržavanje shielda nakon izlaza iz zavoja.
                # Sprječava da baseline odmah preuzme s raw_steer=1.000.
                # ALI: u raskrižju raw_steer=1.000 znači "agent želi skrenuti",
                # pa NE smijemo produžavati hold zbog raw_steer u junction zoni.
                if lane_recenter_was_active:
                    if (
                        (not in_junction_zone and abs(raw_steer) > 0.60)
                        or abs(yaw_error) > 0.6
                        or abs(lateral_error) > 0.03
                    ):
                        recenter_hold_ticks = max(recenter_hold_ticks, 18)

                # U raskrižju: resetiraj hold ticks da shield ne nosi "memoriju"
                # iz prethodnog zavoja u raskrižje.
                # Koristimo lateral_error umjesto hard_edge_guard_active jer
                # hard_edge_guard još nije izračunat u ovom trenutku.
                if junction_turn_ready and not (
                    sidewalk_guard_active
                    or abs(lateral_error) > 0.45
                ):
                    recenter_hold_ticks = 0

                soft_edge_guard_active = (
                    abs(lateral_error) > SOFT_EDGE_GUARD_THRESHOLD or
                    sidewalk_guard_active or
                    (abs(lateral_error) > 0.18 and moving_away_from_center)
                )
                hard_edge_guard_active = abs(lateral_error) > HARD_EDGE_GUARD_THRESHOLD

                # Baseline je opasan samo ako još uvijek daje jak volan dok nismo centrirani.
                # Kada smo stvarno blizu centra, ne držimo shield aktivan samo zbog raw_steer=1.000,
                # jer bi inače shield mogao nepotrebno dugo voziti sporo.
                baseline_still_dangerous = (
                    abs(raw_steer) > RECENTER_RELEASE_RAW_STEER_LIMIT
                    and (
                        abs(lateral_error) > 0.03
                        or abs(yaw_error) > 0.5
                        or sidewalk_guard_active
                        or baseline_pushing_toward_sidewalk
                    )
                )

                if lane_recenter_was_active:
                    lane_or_curve_unstable = (
                        abs(lateral_error) > RECENTER_OFF_THRESHOLD or
                        abs(yaw_error) > RECENTER_RELEASE_YAW_THRESHOLD or
                        baseline_still_dangerous or
                        soft_edge_guard_active or
                        recenter_hold_ticks > 0
                    )
                else:
                    lane_or_curve_unstable = (
                        abs(lateral_error) > RECENTER_ON_THRESHOLD or
                        soft_edge_guard_active or
                        (raw_brake > 0.70 and abs(yaw_error) > 18.0) or
                        (abs(raw_steer) > 0.70 and abs(lateral_error) > 0.14)
                    )

                if (
                    not collision_risk_active
                    and len(collision_events) == 0
                    and lane_or_curve_unstable
                    and not curve_assist_active
                    and not junction_turn_release_active
                    # U RASKRIŽJU: lane_recenter se GASI da ne blokira skretanje.
                    # raw_steer=1.000 u raskrižju znači "agent želi skrenuti",
                    # ne "opasno skreće prema rubu". lane_recenter ga ne smije
                    # ispravljati natrag na ravno.
                    # IZNIMKA: ako je vozilo stvarno uz rub/trotoar ili ima
                    # opasan lateralni offset, shield se smije uključiti i u raskrižju.
                    and (
                        not (ego_in_junction or junction_ahead)
                        or hard_edge_guard_active
                        or sidewalk_guard_active
                        or abs(lateral_error) > 0.45
                    )
                ):
                    lane_recenter_active = True

                    # Ako je yaw_error velik, vozilo još uvijek mora pratiti zavoj,
                    # ali kod izlaza iz zavoja lateral_error mora imati prioritet.
                    # U logu se vidi da je pri izlazu lateral_error rastao do oko +0.63,
                    # a stari center_hold_steer od ~0.15 nije bio dovoljan da vozilo vrati u sredinu.
                    same_direction_yaw_and_offset = (lateral_error * yaw_error) > 0.0

                    # Vrlo bitno: kada je vozilo blizu ruba, yaw_error ne smije
                    # poništiti lateralno centriranje. U zadnjem logu se vidjelo da
                    # kod lateral_error oko 0.50 i yaw_error oko -55 sustav ponekad
                    # oslabi ili čak okrene korekciju, pa se vozilo nepotrebno dugo
                    # vozi uz rub. Zato je yaw doprinos ovdje namjerno ograničen.
                    limited_yaw_term = yaw_error * 0.0010
                    if limited_yaw_term > 0.045:
                        limited_yaw_term = 0.045
                    elif limited_yaw_term < -0.045:
                        limited_yaw_term = -0.045

                    if hard_edge_guard_active:
                        # Blizu ruba: povratak prema centru trake ima prioritet.
                        recenter_steer = lateral_error * 0.52 + limited_yaw_term

                    elif sidewalk_guard_active:
                        # Rub/trotoar: ranije uključivanje, jači gain.
                        recenter_steer = lateral_error * 0.60 + limited_yaw_term

                    elif exit_recenter_counter > 0:
                        # IZLAZ IZ RASKRIŽJA: pure-pursuit je upravo predao volan.
                        # Ovdje je yaw_error često 8-18°, što bi inače uključilo
                        # raw_steer-grane, a raw_steer je na izlazu mrtav/divlji signal
                        # (skače 0.58/-0.29/1.0) pa volan trza lijevo-desno. Zato
                        # koristimo ČISTO lateralno centriranje + ograničen yaw,
                        # bez raw_steera. Zavoji (gdje pp nije vodio) ostaju netaknuti.
                        recenter_steer = lateral_error * 0.45 + limited_yaw_term

                    elif abs(yaw_error) > 18.0:
                        # U ZAVOJU: KLJUČNA PROMJENA.
                        # Ne koristimo yaw_error za smjer volana jer mu je predznak
                        # (od lookahead waypointa) nepouzdan na prijelazu i u oštrom zavoju.
                        # Umjesto toga koristimo baseline raw_steer koji ZNA ispravan smjer
                        # zavoja (u logu je raw_steer ispravno pozitivan kroz cijeli desni zavoj),
                        # i na njega dodajemo lateralnu korekciju za centriranje.
                        # raw_steer * scale = praćenje zavoja, lateral_error * gain = centriranje.
                        curve_follow = raw_steer * 1.5
                        centering = lateral_error * 0.45
                        recenter_steer = curve_follow + centering

                    elif abs(lateral_error) > 0.04:
                        # Ulaz u zavoj ili ravni dio s malim offsetom.
                        # Ako zavoj dolazi (yaw raste), koristimo raw_steer kao osnovu smjera
                        # plus lateralnu korekciju; inače čisto lateralno centriranje.
                        if abs(yaw_error) > 8.0:
                            recenter_steer = raw_steer * 1.2 + lateral_error * 0.45
                        else:
                            recenter_steer = lateral_error * 0.50 + limited_yaw_term

                    else:
                        # Gotovo centrirani na ravnom: fina korekcija.
                        recenter_steer = lateral_error * 0.42 + limited_yaw_term
                        
                    # PRIGUŠENJE (D-član): sprječava preletanje centra i njihanje.
                    # lane_recenter je inače čisto proporcionalan (lateral_error*gain),
                    # pa kod velikog offseta na izlazu iz raskrižja preleti sredinu i
                    # počne oscilirati lijevo-desno. Kad se lateral_error brzo mijenja
                    # (auto juri prema sredini), oduzimamo dio korekcije da uspori prilaz.
                    recenter_damping = lateral_error_delta * 2.5
                    if recenter_damping > 0.18:
                        recenter_damping = 0.18
                    elif recenter_damping < -0.18:
                        recenter_damping = -0.18
                    recenter_steer += recenter_damping
                    
                        
                    # Sigurnosna provjera: ako smo već jasno blizu ruba, finalna
                    # korekcija ne smije biti u suprotnom smjeru od lateral_errora.
                    if abs(lateral_error) > 0.26 and recenter_steer * lateral_error < 0.0:
                        recenter_steer = lateral_error * 0.36

                    if speed_mps < 2.5:
                        if hard_edge_guard_active:
                            max_center_steer = 0.45
                        elif sidewalk_guard_active:
                            max_center_steer = 0.42
                        elif soft_edge_guard_active:
                            max_center_steer = 0.40
                        elif abs(yaw_error) > 18.0:
                            # U zavoju treba više prostora za praćenje.
                            max_center_steer = 0.38
                        else:
                            max_center_steer = 0.28
                    else:
                        if hard_edge_guard_active:
                            max_center_steer = 0.40
                        elif sidewalk_guard_active:
                            max_center_steer = 0.37
                        elif soft_edge_guard_active:
                            max_center_steer = 0.36
                        elif abs(yaw_error) > 18.0:
                            # U zavoju treba više prostora za praćenje.
                            max_center_steer = 0.34
                        else:
                            max_center_steer = 0.25

                    if recenter_steer > max_center_steer:
                        recenter_steer = max_center_steer
                    elif recenter_steer < -max_center_steer:
                        recenter_steer = -max_center_steer

                    ego_action.steer = recenter_steer

                    # Ako baseline bez rizika koči u zavoju, makni kočnicu.
                    if raw_brake > 0.70 and not red_or_yellow_light:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.14
                    # Crveno/žuto svjetlo: poštuj baseline kočnicu, NE daj gas.
                    if red_or_yellow_light and raw_brake > 0.70:
                        ego_action.throttle = 0.0
                        ego_action.brake = max(ego_action.brake, raw_brake)
                    # Ne smije stati u zavoju.
                    elif speed_mps > 7.0 and (hard_edge_guard_active or abs(yaw_error) > 12.0):
                        ego_action.throttle = 0.0
                        ego_action.brake = max(ego_action.brake, 0.05)

                    elif speed_mps > 10.0:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.16

                    elif speed_mps > 6.0:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.42

                    else:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.60

                    print(
                        f"SHIELD: curve/lane center hold active "
                        f"(lateral_error={lateral_error:.2f}, "
                        f"yaw_error={yaw_error:.2f}, "
                        f"raw_steer={raw_steer:.3f}, "
                        f"raw_brake={raw_brake:.3f}, "
                        f"lateral_delta={lateral_error_delta:.3f}, "
                        f"soft_edge_guard={soft_edge_guard_active}, "
                        f"sidewalk_guard={sidewalk_guard_active}, "
                        f"hard_edge_guard={hard_edge_guard_active}, "
                        f"center_hold_steer={ego_action.steer:.3f})"
                    )
                    
                    
                    
                # Kada baseline smije skretati po ruti, ipak ga u ulazu
                # u raskrižje ograniči da ne uđe s punim gasom i punim volanom.
                if junction_turn_release_active:
                    if speed_mps > 2.2:
                        ego_action.throttle = 0.0
                        ego_action.brake = max(ego_action.brake, 0.08)
                    elif ego_action.throttle > 0.24:
                        ego_action.throttle = 0.24

                    if ego_action.steer > 0.60:
                        ego_action.steer = 0.60
                    elif ego_action.steer < -0.60:
                        ego_action.steer = -0.60

                # ==========================================
                # SHIELD 3C: ROUTE-BASED JUNCTION TURN STEER
                # ==========================================
                # Ako baseline volan ima suprotan predznak od XML rute,
                # ne smijemo ga pustiti u raskrižju. Tada vozimo skretanje
                # iz route_dyaw signala, ali ograničeno i sporo, slično starom
                # recovery ponašanju: dovoljno da skrene, ali bez punog trzaja.
                
                # Usporavanje SAMO neposredno pred OŠTAR skret (raskrižje blizu).
                # junction_distance_to_entry < 12 drži kočenje lokalnim - NE usporava
                # cijelu vožnju kao prošla verzija. Jako (0.60) jer se raskrižje
                # detektira tek na ~10 m a auto juri ~8 m/s; ovako padne na ~2.5 m/s
                # prije 90° luka i ne presiječe na hidrant.
                if (
                    abs(route_dyaw) > 60.0
                    and route_turn_direction != 0
                    and junction_distance_to_entry < 12.0
                    and speed_mps > 2.5
                ):
                    ego_action.throttle = 0.0
                    ego_action.brake = max(ego_action.brake, 0.60)

                in_junction_turn = (
                    (turning_in_junction_zone or junction_ahead)
                    and route_turn_direction != 0
                    and intersection_risk is None
                    and not collision_risk_active
                    and not red_or_yellow_light
                )
                # Izlazna zona: tek smo izašli iz skreta, ali get_waypoint još
                # projicira na SUPROTNU traku (velik yaw_error). Nastavljamo voditi
                # po ruti dok se referenca ne ispravi - inače se auto naseli u
                # suprotnu traku (lane_recenter centrira na krivu, najbližu traku).
                pp_exit_zone = (
                    (not in_junction_turn)
                    and pp_was_active
                    and (abs(yaw_error) > 45.0 or abs(lateral_error) > 0.4)
                    and intersection_risk is None
                    and not collision_risk_active
                    and not red_or_yellow_light
                )
                if in_junction_turn or pp_exit_zone or ego_in_junction or junction_ahead:
                    junction_route_steer_active = True
                    if in_junction_turn:
                        pp_was_active = True

                    # U raskrižju glatko (8 m); čim izađemo, kraći lookahead (4 m)
                    # da auto tješnje priđe ruti umjesto da izlijeće široko preko
                    # sredine u suprotnu traku, pa se mora vraćati.
                    # U samom luku (ego_in_junction) lookahead 4 m vodi skret.
                    # Na prilazu/izlazu (izvan raskrižja) kraći 3 m: cilja bližu
                    # točku pa kad auto izađe iz luka pomaknut u stranu, alpha je
                    # veći i volan ga jače povuče natrag na rutu - inače se poravna
                    # sa smjerom rute ali vozi paralelno, ~1 m sa strane.
                    pp_lookahead = 3.0 if ego_in_junction else 3.0
                    # Oštar skret (~90°) treba jači volan: max 0.50 ne zakrene
                    # dovoljno pa auto zaostaje (yaw_error raste) i presiječe
                    # unutarnji ugao (u peti skret se zabijao u hidrant). Za oštre
                    # skretove dižemo limit; blagi ostaju na 0.50 radi stabilnosti.
                    pp_max_steer = 0.70 if abs(route_dyaw) > 60.0 else 0.50
                    pp_steer, pp_alpha, pp_tx, pp_ty = pure_pursuit_junction_steer(
                        vehicle_location.x,
                        vehicle_location.y,
                        current_yaw_for_route,
                        route_waypoints,
                        lookahead_m=pp_lookahead,
                        steer_gain=0.65,
                        max_steer=pp_max_steer,
                    )
                    ego_action.steer = pp_steer

                   # Brzinu ograničavamo SAMO kad stvarno skrećemo. Ravni prolaz
                    # kroz raskrižje (route_turn_direction == 0) NE usporavamo -
                    # pursuit i dalje vodi volan (centriranje), ali brzina ostaje
                    # krstareća. Bez ovog uvjeta pursuit (sad aktivan i na ravnim
                    # prolazima zbog 'or junction_ahead') koči i kad ide ravno.
                    if route_turn_direction != 0:
                        # Sporo i kontrolirano kroz ulaz u raskrižje pri skretu.
                        if speed_mps > 3.0:  # 2.2 je najsigurnija opcija no spora
                            ego_action.throttle = 0.0
                            pp_brake = 0.35 if abs(route_dyaw) > 60.0 else 0.20
                            ego_action.brake = max(ego_action.brake, pp_brake)
                        elif speed_mps > 1.5:
                            ego_action.brake = 0.0
                            ego_action.throttle = min(ego_action.throttle, 0.14)
                        else:
                            ego_action.brake = 0.0
                            ego_action.throttle = max(ego_action.throttle, 0.24)

                    print(
                        f"SHIELD: route-based junction steer active "
                        f"(route_dyaw={route_dyaw:.1f}, "
                        f"pp_alpha={pp_alpha:.1f}, "
                        f"pp_target=({pp_tx:.1f},{pp_ty:.1f}), "
                        f"raw_steer={raw_steer:.3f}, "
                        f"route_steer={ego_action.steer:.3f}, "
                        f"speed={speed_mps:.2f})"
                    )
                
                # Izlazna zona gotova kad get_waypoint opet vraća ISPRAVNU traku
                # (auto poravnat s njom -> mali yaw_error). Tek tada lane_recenter
                # smije preuzeti centriranje.
                if abs(yaw_error) <= 45.0 and abs(lateral_error) <= 0.4:
                    if pp_was_active:
                        # pure-pursuit upravo predaje -> kratko čisto centriranje
                        exit_recenter_counter = 26
                    pp_was_active = False

                if exit_recenter_counter > 0:
                    exit_recenter_counter -= 1
                    
                # Resetiraj turn-progress state TEK kad auto izađe iz raskrižja,
                # ne čim route-steer prestane - inače se skret stalno ponovno pokreće.
                if not (ego_in_junction or junction_ahead):
                    turn_start_yaw = None
                    turn_planned_angle = 0.0
                
                # Spremi zadnju shield korekciju za glatkiji povratak baselineu.
                if curve_assist_active or lane_recenter_active or junction_route_steer_active:
                    last_shield_steer = ego_action.steer

                elif (
                    not turning_in_junction_zone
                    and abs(last_shield_steer) > 0.02
                    and (
                        abs(yaw_error) > 0.6
                        or abs(lateral_error) > 0.03
                        or abs(raw_steer) > 0.60
                    )
                ):
                    # Ako se shield upravo ugasio, nemoj odmah potpuno vratiti baseline.
                    # Ovo sprječava nagli skok natrag na raw_steer=1.000 pri izlazu.
                    ego_action.steer = (
                        0.60 * last_shield_steer +
                        0.40 * ego_action.steer
                    )

                    last_shield_steer *= 0.60
                    # Ne dopuštaj nagli puni gas odmah nakon što shield pusti kontrolu.
                    if raw_throttle > 0.40 and speed_mps < 3.2:
                        ego_action.throttle = min(ego_action.throttle, 0.25)

                else:
                    last_shield_steer = 0.0

                if recenter_hold_ticks > 0:
                    recenter_hold_ticks -= 1

                prev_lateral_error = lateral_error
                lane_recenter_was_active = lane_recenter_active   
                # ==========================================
                # SHIELD: COLLISION RISK OVERRIDE
                # ==========================================
                # ==========================================
                # ODABIR NAJVEĆEG COLLISION RIZIKA
                # ==========================================

                risk_priority = {
                    "NONE": 0,
                    "CAUTION": 1,
                    "BRAKE": 2,
                    "EMERGENCY": 3,
                }

                highest_risk = None

                # Provjera rizika od pješaka
                if pedestrian_collision_risk is not None:
                    highest_risk = pedestrian_collision_risk

                # Provjera rizika od vozila ispred
                if vehicle_collision_risk is not None:
                    if highest_risk is None:
                        highest_risk = vehicle_collision_risk
                    else:
                        vehicle_priority = risk_priority[
                            vehicle_collision_risk["risk_level"]
                        ]

                        current_priority = risk_priority[
                            highest_risk["risk_level"]
                        ]

                        if vehicle_priority > current_priority:
                            highest_risk = vehicle_collision_risk

                # Provjera rizika od poprečnog prometa
                if cross_traffic_risk is not None:
                    if highest_risk is None:
                        highest_risk = cross_traffic_risk
                    else:
                        cross_priority = risk_priority[
                            cross_traffic_risk["risk_level"]
                        ]

                        current_priority = risk_priority[
                            highest_risk["risk_level"]
                        ]

                        if cross_priority > current_priority:
                            highest_risk = cross_traffic_risk

                # ==========================================
                # LANE-HOLD STEERING TIJEKOM COLLISION RIZIKA
                # ==========================================

                collision_lane_steer = ego_action.steer

                if highest_risk is not None and pid_target_wp is not None:
                    lane_hold_control = collision_lane_controller.run_step(
                        target_speed=0.0,
                        waypoint=pid_target_wp
                    )

                    collision_lane_steer = lane_hold_control.steer

                # ==========================================
                # PRIMJENA COLLISION OVERRIDEA
                # ==========================================

                if highest_risk is not None:
                    risk_level = highest_risk["risk_level"]
                    actor_type = highest_risk["type_id"]

                    gap_m = highest_risk.get("gap_m", float("inf"))
                    ttc = highest_risk.get("ttc", float("inf"))

                    local_forward = highest_risk["local_forward"]
                    local_right = highest_risk["local_right"]

                    time_to_closest = highest_risk.get(
                        "time_to_closest",
                        float("inf")
                    )

                    closest_distance = highest_risk.get(
                        "closest_distance",
                        float("inf")
                    )

                    if risk_level == "EMERGENCY":
                        print(
                            f"SHIELD: COLLISION EMERGENCY "
                            f"type={actor_type} "
                            f"forward={local_forward:.2f} "
                            f"right={local_right:.2f} "
                            f"gap={gap_m:.2f} "
                            f"ttc={ttc:.2f} "
                            f"ttc_cross={time_to_closest:.2f} "
                            f"closest_dist={closest_distance:.2f}"
                        )

                        ego_action.steer = collision_lane_steer
                        ego_action.throttle = 0.0
                        ego_action.brake = 1.0

                    elif risk_level == "BRAKE":
                        print(
                            f"SHIELD: COLLISION BRAKE "
                            f"type={actor_type} "
                            f"forward={local_forward:.2f} "
                            f"right={local_right:.2f} "
                            f"gap={gap_m:.2f} "
                            f"ttc={ttc:.2f} "
                            f"ttc_cross={time_to_closest:.2f} "
                            f"closest_dist={closest_distance:.2f}"
                        )

                        ego_action.steer = collision_lane_steer
                        ego_action.throttle = 0.0
                        ego_action.brake = max(ego_action.brake, 0.70)

                    elif risk_level == "CAUTION":
                        print(
                            f"SHIELD: COLLISION CAUTION "
                            f"type={actor_type} "
                            f"forward={local_forward:.2f} "
                            f"right={local_right:.2f} "
                            f"gap={gap_m:.2f} "
                            f"ttc={ttc:.2f} "
                            f"ttc_cross={time_to_closest:.2f} "
                            f"closest_dist={closest_distance:.2f}"
                        )

                        ego_action.steer = (
                            0.40 * ego_action.steer
                            + 0.60 * collision_lane_steer
                        )

                        ego_action.throttle = min(
                            ego_action.throttle,
                            0.15
                        )

                        ego_action.brake = max(
                            ego_action.brake,
                            0.15
                        )

                # ==========================================
                # SHIELD: INTERSECTION YIELD / STOP
                # ==========================================
                # Zaseban sloj za raskrižje. Aktivira se samo kada ego skreće i
                # postoji vozilo koje presijeca putanju (vidi evaluate_intersection_priority_risk).
                # Ne dira steer (skretanje se nastavlja), samo usporava ili zaustavlja.
                # Collision override iznad ima prioritet: ako je već EMERGENCY/BRAKE
                # kočnica jača, ne smanjujemo je.
                intersection_risk_level = "NONE"
                intersection_actor_id = None
                intersection_time_to_closest = float("inf")
                intersection_closest_distance = float("inf")

                if intersection_risk is not None:
                    intersection_risk_level = intersection_risk["risk_level"]
                    intersection_actor_id = intersection_risk.get("actor_id")
                    intersection_time_to_closest = intersection_risk.get(
                        "time_to_closest", float("inf")
                    )
                    intersection_closest_distance = intersection_risk.get(
                        "closest_distance", float("inf")
                    )

                    # Obveza na skret = koliko je ego STVARNO zakrenuo od ulaska u
                    # raskrizje (napredak skreta), NE kut rute. route_dyaw je velik
                    # kroz CIJELI skret (i na ulazu) pa je gusio STOP svuda - zato se
                    # ego nije zaustavljao. Sad: na ulazu (mali napredak) STOP smije
                    # zakociti (stani u svojoj traci); tek kad je ego vec duboko
                    # zakrenuo (u traci), dovrsi skret.
                    if ego_in_junction and route_turn_direction != 0:
                        if turn_entry_yaw is None:
                            turn_entry_yaw = facts["yaw"]
                        _tp = (facts["yaw"] - turn_entry_yaw + 180.0) % 360.0 - 180.0
                        turn_progress = abs(_tp)
                    else:
                        turn_entry_yaw = None
                        turn_progress = 0.0
                    committed_to_turn = ego_in_junction and turn_progress > 45.0

                    if committed_to_turn and intersection_risk_level in ("STOP", "YIELD"):
                        print(
                            f"SHIELD: INTERSECTION RISK ZANEMAREN - dovrsavam skret "
                            f"(route_dyaw={route_dyaw:.1f}, "
                            f"actor_id={intersection_actor_id})"
                        )
                    elif intersection_risk_level == "STOP":
                        print(
                            f"SHIELD: INTERSECTION STOP "
                            f"intent={ego_turning_intent} "
                            f"actor_id={intersection_actor_id} "
                            f"ttc={intersection_time_to_closest:.2f} "
                            f"closest_dist={intersection_closest_distance:.2f}"
                        )
                        if GAP_ACCEPTANCE_ENABLED:
                            ego_action.throttle = 0.0
                            ego_action.brake = max(ego_action.brake, 1.0)

                    elif intersection_risk_level == "YIELD":
                        print(
                            f"SHIELD: INTERSECTION YIELD "
                            f"intent={ego_turning_intent} "
                            f"actor_id={intersection_actor_id} "
                            f"ttc={intersection_time_to_closest:.2f} "
                            f"closest_dist={intersection_closest_distance:.2f}"
                        )
                        if GAP_ACCEPTANCE_ENABLED:
                            ego_action.throttle = 0.0
                            ego_action.brake = max(ego_action.brake, 0.4)
                # ==========================================
                # SHIELD: JUNCTION CREEP ASSIST
                # ==========================================
                # Ako baseline lažno zakoči usred/prije skretanja u raskrižju,
                # a nema stvarnog collision/intersection rizika ni crvenog/žutog svjetla,
                # pusti vozilo da polako puže kroz skretanje.
                if (
                    (junction_turn_release_active or junction_route_steer_active
                     or (junction_ahead and route_turn_direction == 0))
                    and speed_mps < 1.50
                    and raw_brake > 0.70
                    and not red_or_yellow_light
                    and intersection_risk is None
                    and not collision_risk_active
                    and len(collision_events) == 0
                ):
                    junction_creep_active = True
                    ego_action.brake = 0.0
                    ego_action.throttle = max(ego_action.throttle, 0.36)
                    print(
                        f"SHIELD: junction creep assist active "
                        f"(intent={ego_turning_intent}, "
                        f"speed={speed_mps:.2f}, "
                        f"raw_steer={raw_steer:.3f}, "
                        f"raw_brake={raw_brake:.3f})"
                    )

                # ANTI-STUCK: baseline ponekad lažno zakoči IZVAN raskrižja, na
                # ravnoj cesti bez ikakvog stvarnog rizika (tipično kad je route_dyaw
                # wrap-around ~±180° pa je baseline zbunjen). junction_creep pokriva
                # samo raskrižja; ovo pokriva ravnu cestu. Ako stojimo a nema nijednog
                # stvarnog razloga za kočenje, pusti throttle da auto nastavi.
                if (
                    speed_mps < 1.50
                    and raw_brake > 0.70
                    and not red_or_yellow_light
                    and intersection_risk is None
                    and not collision_risk_active
                    and len(collision_events) == 0
                    and not junction_creep_active
                ):
                    ego_action.brake = 0.0
                    ego_action.throttle = max(ego_action.throttle, 0.40)
                    print(
                        f"SHIELD: anti-stuck release active "
                        f"(speed={speed_mps:.2f}, raw_brake={raw_brake:.3f}, "
                        f"route_dyaw={route_dyaw:.1f})"
                    )

                # ==========================================
                # STATIC / TRAFFIC OBJECT COLLISION LOG
                # ==========================================
                # Ako collision sensor javi udar u statički objekt ili semafor,
                # ne smijemo nastaviti davati gas. Ovo nije dynamic collision risk,
                # nego fizički kontakt s map objektom.
                if new_physical_collision and len(collision_events) > 0:
                    last_event = collision_events[-1]
                    other_type = last_event.get("other_actor_type", "")

                    if other_type.startswith("static.") or other_type.startswith("traffic."):
                        print(
                            f"SHIELD: static/traffic-object collision detected "
                            f"(type={other_type}, collision_count={len(collision_events)})"
                        )
                        ego_action.throttle = 0.0
                        ego_action.brake = max(ego_action.brake, 1.0)




                # ==========================================
                # RISK LEVELI ZA GLAVNI LOG
                # ==========================================

                vehicle_risk_level = (
                    vehicle_collision_risk["risk_level"]
                    if vehicle_collision_risk is not None
                    else "NONE"
                )

                pedestrian_risk_level = (
                    pedestrian_collision_risk["risk_level"]
                    if pedestrian_collision_risk is not None
                    else "NONE"
                )

                cross_traffic_risk_level = (
                    cross_traffic_risk["risk_level"]
                    if cross_traffic_risk is not None
                    else "NONE"
                )
                
                
                # ==========================================
                # FINAL STEERING RATE LIMITER
                # ==========================================
                # Važno:
                # - NE limitiramo steer dok je lane_recenter_active.
                # - Shield mora smjeti odmah korigirati vozilo od ruba/trotoara.
                # - Limiter koristimo samo kada baseline preuzima kontrolu.

                if (
                    not collision_risk_active
                    and not lane_recenter_active
                    and not curve_assist_active
                    and not junction_turn_release_active
                    and not junction_route_steer_active
                ):
                    desired_final_steer = ego_action.steer

                    if speed_mps < 0.3:
                        max_steer_delta = 0.035
                    elif raw_brake > 0.70:
                        max_steer_delta = 0.055
                    elif speed_mps < 2.5:
                        max_steer_delta = 0.075
                    else:
                        max_steer_delta = 0.10

                    steer_delta = desired_final_steer - prev_final_steer

                    if steer_delta > max_steer_delta:
                        steer_delta = max_steer_delta
                    elif steer_delta < -max_steer_delta:
                        steer_delta = -max_steer_delta

                    ego_action.steer = prev_final_steer + steer_delta

                prev_final_steer = ego_action.steer
                


                print(
                    f"step={step} "
                    f"speed={facts['speed_mps']:.2f} "
                    f"steer={ego_action.steer:.3f} "
                    f"throttle={ego_action.throttle:.3f} "
                    f"brake={ego_action.brake:.3f} "
                    f"x={facts['x']:.2f} "
                    f"y={facts['y']:.2f} "
                    f"yaw={facts['yaw']:.2f} "
                    f"wp_x={wp_loc.x:.2f} "
                    f"wp_y={wp_loc.y:.2f} "
                    f"wp_yaw={wp_yaw:.2f} "
                    f"wp_lane_id={wp_lane_id} "
                    f"wp_road_id={wp_road_id} "
                    f"wp_section_id={wp_section_id} "
                    f"wp_lane_width={wp_lane_width:.2f} "
                    f"wp_junction={wp_is_junction} "
                    f"yaw_error={yaw_error:.2f} "
                    f"lateral_error={lateral_error:.2f} "
                    f"lookahead={lookahead_distance:.1f} "
                    f"collision_risk={collision_risk_active} "
                    f"vehicle_risk={vehicle_risk_level} "
                    f"pedestrian_risk={pedestrian_risk_level} "
                    f"collision_count={len(collision_events)} "
                    f"cross_traffic_risk={cross_traffic_risk_level} "
                    f"raw_steer={raw_steer:.3f} "
                    f"raw_throttle={raw_throttle:.3f} "
                    f"raw_brake={raw_brake:.3f} "
                    f"curve_assist={curve_assist_active} "
                    f"lane_recenter={lane_recenter_active} "
                    f"sidewalk_guard={sidewalk_guard_active} "
                    f"soft_edge_guard={soft_edge_guard_active} "
                    f"baseline_to_sidewalk={baseline_pushing_toward_sidewalk} "
                    f"hard_edge_guard={hard_edge_guard_active} "
                    f"recenter_hold={recenter_hold_ticks} "
                    f"traffic_light={traffic_light_state} "
                    f"ego_in_junction={ego_in_junction} "
                    f"junction_ahead={junction_ahead} "
                    f"junction_dist={junction_distance_to_entry:.1f} "
                    f"junction_turn_ready={junction_turn_ready} "
                    f"ego_turning_intent={ego_turning_intent} "
                    f"route_intent={route_intent} "
                    f"route_dyaw={route_dyaw:.1f} "
                    f"intersection_risk={intersection_risk_level} "
                    f"intersection_actor_id={intersection_actor_id} "
                    f"intersection_ttc={intersection_time_to_closest:.2f} "
                    f"intersection_closest_dist={intersection_closest_distance:.2f} "
                    f"turning_junction_zone={turning_in_junction_zone} "
                    f"junction_turn_release={junction_turn_release_active} "
                    f"raw_route_match={raw_steer_matches_route} "
                    f"junction_route_steer={junction_route_steer_active} "
                    f"junction_creep={junction_creep_active} "
                )

                last_collision_count = len(collision_events)

                if route_waypoints:
                    # Najbliži waypoint PO INDEKSU - razlikuje kraj rute od početka.
                    # Ruta se vrati blizu starta (kraj ~(190,250) je 6 m od prvog
                    # segmenta) pa sama udaljenost lažno okida već na početku.
                    _bi = 0
                    _bd2 = float("inf")
                    for _i, _wp in enumerate(route_waypoints):
                        _d2 = ((vehicle_location.x - _wp["x"]) ** 2 +
                               (vehicle_location.y - _wp["y"]) ** 2)
                        if _d2 < _bd2:
                            _bd2 = _d2
                            _bi = _i
                    # Zaustavi tek kad smo kod ZADNJIH waypointa (prošli cijelu rutu) I blizu.
                    if _bi >= len(route_waypoints) - 1 and _bd2 ** 0.5 < 6.0:
                        ego_action.throttle = 0.0
                        ego_action.brake = 1.0
                        ego_action.steer = 0.0
                        print(f"RUTA ZAVRŠENA: cilj dosegnut (idx={_bi}/{len(route_waypoints)-1}), zaustavljam.")


                # BASELINE NAČIN: kad je shield isključen, primijeni ČISTI baseline
                # (raw vrijednosti). Sva shield logika iznad je svejedno izračunata
                # i logirana, ali se NE primjenjuje - usporedba je poštena.
                if not SHIELD_ENABLED:
                    ego_action.steer = float(raw_steer)
                    ego_action.throttle = float(raw_throttle)
                    ego_action.brake = float(raw_brake)

                # CSV redak (PRIMIJENJENE vrijednosti)
                _csv.write(
                    f"{step},{_mode},{facts['x']:.3f},{facts['y']:.3f},"
                    f"{facts['yaw']:.2f},{facts['speed_mps']:.3f},"
                    f"{raw_steer:.3f},{ego_action.steer:.3f},"
                    f"{raw_throttle:.3f},{ego_action.throttle:.3f},"
                    f"{raw_brake:.3f},{ego_action.brake:.3f},"
                    f"{lateral_error:.3f},{yaw_error:.3f},{len(collision_events)},"
                    f"{intersection_risk_level},{ego_turning_intent},{route_intent},"
                    f"{lane_recenter_active},{junction_route_steer_active}\n"
                )

                vehicle.apply_control(ego_action)
                world.tick()

                # Kamera prati vozilo odozgo
                transform = vehicle.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        transform.location + carla.Location(x=0, y=0, z=20),
                        carla.Rotation(pitch=-90, yaw=0, roll=0)
                    )
                )

                step += 1

            except Exception as e:
                print(f'\nError at step {step}:')
                print(f'{type(e).__name__}: {e}\n')
                import traceback
                traceback.print_exc()
                break

    finally:
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception as error:
                print(f"Warning: failed to disable Traffic Manager synchronous mode: {error}")

        if settings is not None and world is not None:
            try:
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception as error:
                print(f"Warning: failed to restore world settings: {error}")

        try:
            _csv.close()
            print(f"CSV spremljen: {_csv_name}")
        except Exception:
            pass

        if oncoming_vehicle is not None:
            try:
                oncoming_vehicle.destroy()
            except Exception:
                pass

        if lead_vehicle is not None:
            try:
                lead_vehicle.destroy()
            except Exception:
                pass

        if pedestrian is not None:
            try:
                pedestrian.destroy()
            except Exception:
                pass

        print('\nCleaning up actors')

        if collision_sensor is not None:
            try:
                collision_sensor.stop()
            except Exception as error:
                print(f"Warning: failed to stop collision sensor: {error}")

            try:
                collision_sensor.destroy()
            except Exception as error:
                print(f"Warning: failed to destroy collision sensor: {error}")

        if pcla is not None:
            try:
                pcla.cleanup()
            except Exception as error:
                print(f"Warning: failed to clean up PCLA: {error}")

        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception as error:
                print(f"Warning: failed to destroy vehicle: {error}")

        time.sleep(0.5)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')