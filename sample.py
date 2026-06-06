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

# Koristi se za detekciju objekata uz desni rub
def detect_front_right_obstacle(world, ego_vehicle,
                                max_forward=8.0,
                                min_right=0.8,
                                max_right=3.2,
                                max_height_diff=2.5):
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    best = None
    best_dist = 1e9

    for actor in world.get_actors():
        if actor.id == ego_vehicle.id:
            continue

        tid = actor.type_id

        # Gledamo objekte koji mogu biti uz cestu ili ispred auta
        if not (
            tid.startswith("static.") or
            tid.startswith("traffic.")
        ):
            continue

        loc = actor.get_location()

        rel_x = loc.x - ego_loc.x
        rel_y = loc.y - ego_loc.y
        rel_z = loc.z - ego_loc.z

        if abs(rel_z) > max_height_diff:
            continue

        local_forward = rel_x * forward.x + rel_y * forward.y + rel_z * forward.z
        local_right = rel_x * right.x + rel_y * right.y + rel_z * right.z
        dist = (rel_x**2 + rel_y**2 + rel_z**2) ** 0.5

        # Objekt mora biti ispred i malo desno od auta
        if 0.0 < local_forward < max_forward and min_right < local_right < max_right:
            if dist < best_dist:
                best_dist = dist
                best = {
                    "actor": actor,
                    "type_id": tid,
                    "distance": dist,
                    "local_forward": local_forward,
                    "local_right": local_right
                }

    return best

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

        if gap_m < 2.5 or ttc < 1.0:
            risk_level = "EMERGENCY"

        elif gap_m < 5.0 or ttc < 2.0:
            risk_level = "BRAKE"

        elif gap_m < 10.0 or ttc < 4.0:
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
    prediction_horizon=4.0
):
    """
    Procjenjuje rizik sudara s vozilima koja dolaze sa strane
    i presijecaju putanju ego vozila.

    Koristi constant-velocity procjenu vremena i udaljenosti
    u trenutku najbližeg prolaska.
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

def main():

    collision_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "collision_log.txt"
    )

    print(f"Collision log path: {collision_log_path}")

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

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)

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
        vehicle = world.spawn_actor(vehicleBP, vehicle_spawn_points[0])  # bilo je 31 izmjenjeno u 0
        
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

            with open(collision_log_path, "a", encoding="utf-8") as f:
                f.write(collision_line + "\n")

        collision_sensor.listen(on_collision)


        world.tick()

        agent = "neat_neat"  # bio je "simlingo_simlingo" ali ne radi kako treba
        route = "./route_spawn0.xml"  # bilo route = "./sample_route.xml"
        pcla = PCLA(agent, vehicle, route, client)
        
        # Lateralni PID kontrolira volan, a longitudinalni kontrolira brzinu/gas/kočenje.
        pid_recovery_controller = VehiclePIDController(
            vehicle,
            args_lateral={'K_P': 2.40, 'K_D': 0.30, 'K_I': 0.02, 'dt': 0.05},
            args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.03, 'dt': 0.05},
            offset=0.00,
            max_throttle=0.30,
            max_brake=0.35,
            max_steering=0.55
        )
        
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

        recovery_timer = 0
        recovery_stable_counter = 0
       
        # Dodatni guard nakon izlaska iz zavoja.
        # Cilj: ne vratiti odmah kontrolu baseline agentu čim auto prođe zavoj.
        exit_guard_timer = 0
        baseline_cooldown_timer = 0
       



        print('\nSpawned the vehicle with model =', agent, ', press Ctrl+C to exit.\n')

        step = 0
        stuck_counter = 0
       
        while True:
            try:
                ego_action = pcla.get_action() # Ovdje dobivamo kontrolu od baseline agenta

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

                cross_traffic_risk = evaluate_cross_traffic_risk(
                    dynamic_actor_facts,
                    vehicle
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
                # SHIELD 3: PID RECOVERY MODE
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

                    transform = vehicle.get_transform()
                    spectator.set_transform(
                        carla.Transform(
                            transform.location + carla.Location(x=0, y=0, z=20),
                            carla.Rotation(pitch=-90, yaw=0, roll=0)
                        )
                    )

                    step += 1
                    continue

                current_yaw = vehicle.get_transform().rotation.yaw
                veh_loc = vehicle.get_transform().location
                lane_center = current_wp.transform.location

                # Lookahead: kraći kad je auto spor ili nestabilan, dulji kad se stabilizira
                if recovery_timer > 0:
                    lookahead_distance = 5.0
                elif speed_mps < 1.0:
                    lookahead_distance = 4.0
                elif speed_mps < 2.0:
                    lookahead_distance = 6.0
                else:
                    lookahead_distance = 8.0

                next_wps = current_wp.next(lookahead_distance)

                yaw_error = 0.0
                lateral_error = 0.0

                if len(next_wps) > 0:
                    target_yaw = next_wps[0].transform.rotation.yaw
                    yaw_error = normalize_angle_deg(target_yaw - current_yaw)

                dx = lane_center.x - veh_loc.x
                dy = lane_center.y - veh_loc.y

                yaw_rad = math.radians(current_yaw)
                right_x = -math.sin(yaw_rad)
                right_y = math.cos(yaw_rad)

                # Pozitivno: centar trake je desno od auta
                # Negativno: centar trake je lijevo od auta
                # Ako je auto otišao prema desnom trotoaru, često će lateral_error biti negativan.
                lateral_error = dx * right_x + dy * right_y

                # Negativan lateral_error znači da je centar trake lijevo od auta,
                # tj. auto je pobjegao prema desnom rubu.
                right_edge_risk = lateral_error < -0.35
                right_edge_severe = lateral_error < -0.60
                road_ahead_straight = abs(yaw_error) < 10.0

                front_right_obstacle = None
                front_right_hazard = False

                if (
                    recovery_timer == 0
                    and exit_guard_timer == 0
                    and baseline_cooldown_timer == 0
                    and road_ahead_straight
                    and not collision_risk_active
                ):
                    front_right_obstacle = detect_front_right_obstacle(
                        world,
                        vehicle,
                        max_forward=9.0 if speed_mps < 4.0 else 11.0,
                        min_right=0.8,
                        max_right=3.2
                    )

              

                if recovery_timer == 0 and exit_guard_timer == 0 and baseline_cooldown_timer == 0:

                    if collision_risk_active:
                        # Vozilo nije zapelo; zaustavljeno je zbog drugog sudionika.
                        stuck_counter = 0

                    elif facts["speed_mps"] < 0.2 and facts["throttle"] > 0.5:
                        stuck_counter += 1

                    else:
                        stuck_counter = 0

                    if stuck_counter >= 20:
                        print("SHIELD: vehicle appears stuck -> applying emergency brake")
                        ego_action.throttle = 0.0
                        ego_action.brake = 1.0
                else:
                    stuck_counter = 0


                if front_right_obstacle is not None:
                    front_right_hazard = front_right_obstacle["distance"] < 8.0

                low_speed = speed_mps < 0.8
                very_low_speed = speed_mps < 0.35
                route_turn_like = abs(yaw_error) > 8.0
                strong_turn_like = abs(yaw_error) > 14.0
                off_center = abs(lateral_error) > 0.50
                very_off_center = abs(lateral_error) > 0.90

                lane_width = current_wp.lane_width if current_wp is not None else 3.5
                near_lane_edge = abs(lateral_error) > (0.25 * lane_width)

                startup_grace_over = step >= 40
                actual_stuck = stuck_counter >= 20

                need_recovery = (
                    startup_grace_over
                    and not collision_risk_active
                    and exit_guard_timer == 0
                    and baseline_cooldown_timer == 0
                    and (
                        very_off_center or
                        (off_center and low_speed) or
                        (near_lane_edge and low_speed) or
                        actual_stuck
                    )
                )

                if recovery_timer == 0 and need_recovery:
                    recovery_timer = 55
                    recovery_stable_counter = 0

                    print(
                        f"SHIELD: PID recovery mode armed "
                        f"(yaw_error={yaw_error:.2f}, lateral_error={lateral_error:.2f}, "
                        f"speed={speed_mps:.2f}, stuck_counter={stuck_counter}, "
                        f"baseline_steer={ego_action.steer:.3f})"
                    )

                
                if recovery_timer > 0 and len(next_wps) > 0 and not collision_risk_active:                  
                    target_wp = next_wps[0]
                    # Sporije kad je jako izvan centra ili skoro stao,
                    # malo brže kad se već vraća u traku.
                    if very_off_center or very_low_speed:
                        recovery_target_speed = 3.0
                    elif strong_turn_like:
                        recovery_target_speed = 4.0
                    else:
                        recovery_target_speed = 5.5

                    recovery_control = pid_recovery_controller.run_step(
                        target_speed=recovery_target_speed,  # km/h
                        waypoint=target_wp
                    )

                    ego_action = recovery_control

                    aligned_with_lane = abs(yaw_error) < 7.0
                    centered_in_lane = abs(lateral_error) < 0.50
                    moving_ok = speed_mps > 0.9
                    not_turning_hard_anymore = abs(yaw_error) < 7.0

                    # Recovery se ne smije ugasiti nakon samo jednog dobrog framea.
                    # Mora biti stabilan nekoliko tickova zaredom.
                    if aligned_with_lane and centered_in_lane and moving_ok and not_turning_hard_anymore:
                        recovery_stable_counter += 1
                    else:
                        recovery_stable_counter = 0

                    if recovery_stable_counter >= 12:
                        recovery_timer = 0
                        recovery_stable_counter = 0

                        # Nakon recoveryja ostajemo još kratko u PID kontroli.
                        # Ovo sprječava da baseline odmah nakon zavoja ode prema trotoaru.
                        exit_guard_timer = 35
                        baseline_cooldown_timer = 40

                        print("SHIELD: PID recovery released safely -> exit guard + cooldown armed")
                    else:
                        recovery_timer -= 1

                        if recovery_timer <= 0:
                            still_not_centered = abs(lateral_error) > 0.35
                            still_not_aligned = abs(yaw_error) > 4.0
                            still_too_slow = speed_mps < 0.8

                            if still_not_centered or still_not_aligned or still_too_slow:
                                # Ne vraćaj kontrolu baselineu ako auto još nije stabilan.
                                recovery_timer = 25
                                recovery_stable_counter = 0

                                print(
                                    f"SHIELD: recovery timeout but vehicle still unsafe -> extending recovery "
                                    f"(yaw_error={yaw_error:.2f}, lateral_error={lateral_error:.2f}, speed={speed_mps:.2f})"
                                )
                            else:
                                # Tek ako je stanje stvarno dovoljno dobro, prelazimo u exit guard.
                                recovery_timer = 0
                                recovery_stable_counter = 0
                                exit_guard_timer = 35
                                baseline_cooldown_timer = 40

                                print("SHIELD: recovery timeout but vehicle acceptable -> exit guard armed")
                # ==========================================
                # SHIELD 4: EXIT GUARD AFTER CURVE
                # ==========================================
                # Ovo je dodatna stabilizacija nakon izlaska iz zavoja.
                if recovery_timer == 0 and exit_guard_timer > 0 and len(next_wps) > 0 and not collision_risk_active:
                    target_wp = next_wps[0]

                    exit_control = pid_recovery_controller.run_step(
                        target_speed=8.0,  # km/h, mirno izravnavanje nakon zavoja
                        waypoint=target_wp
                    )

                    # Ograniči throttle da baseline/PID ne povuče auto prema rubu prebrzo.
                    if exit_control.throttle > 0.38:
                        exit_control.throttle = 0.38

                    # Blago ograniči steer nakon zavoja da ne napravi nagli trzaj.
                    if exit_control.steer > 0.40:
                        exit_control.steer = 0.40
                    elif exit_control.steer < -0.40:
                        exit_control.steer = -0.40

                    ego_action = exit_control
                    exit_guard_timer -= 1

                    if exit_guard_timer < 0:
                        exit_guard_timer = 0

                    print(
                        f"SHIELD: exit guard active "
                        f"(timer={exit_guard_timer}, yaw_error={yaw_error:.2f}, "
                        f"lateral_error={lateral_error:.2f})"
                    )            

                # ==========================================
                # SHIELD 5: BASELINE COOLDOWN / POST-CURVE STABILIZER
                # ==========================================
                # Stabilizira baseline da se ne popne na rub odmah nakon zavoja, ali samo ako auto još nije stabilan.
                if recovery_timer == 0 and exit_guard_timer == 0 and baseline_cooldown_timer > 0 and len(next_wps) > 0  and not collision_risk_active:

                    post_curve_not_stable = (
                        abs(yaw_error) > 6.0 or
                        abs(lateral_error) > 0.45
                    )
                    if post_curve_not_stable:
                        # Auto još nije u stanju u kojem baseline inače dobro radi.
                        # Zato još kratko koristimo PID, ali mirno i bez agresivnog skretanja.
                        target_wp = next_wps[0]

                        stabilizer_control = pid_recovery_controller.run_step(
                            target_speed=8.0,
                            waypoint=target_wp
                        )

                        if stabilizer_control.throttle > 0.35:
                            stabilizer_control.throttle = 0.35

                        if stabilizer_control.steer > 0.35:
                            stabilizer_control.steer = 0.35
                        elif stabilizer_control.steer < -0.35:
                            stabilizer_control.steer = -0.35

                        ego_action = stabilizer_control

                        print(
                            f"SHIELD: post-curve stabilizer active "
                            f"(yaw_error={yaw_error:.2f}, lateral_error={lateral_error:.2f})"
                        )

                    else:
                        # Tek sada baseline smije voziti, ali još bez agresivnih trzaja.
                        if ego_action.throttle > 0.45:
                            print("SHIELD: baseline cooldown -> throttle clamped")
                            ego_action.throttle = 0.45

                        if ego_action.steer > 0.30:
                            print("SHIELD: baseline cooldown -> right steer clamped")
                            ego_action.steer = 0.30
                        elif ego_action.steer < -0.30:
                            print("SHIELD: baseline cooldown -> left steer clamped")
                            ego_action.steer = -0.30

                    baseline_cooldown_timer -= 1

                    if baseline_cooldown_timer < 0:
                        baseline_cooldown_timer = 0


                # ==========================================
                # SHIELD 6: RIGHT-EDGE / SIDEWALK OBJECT GUARD
                # ==========================================
                if recovery_timer == 0 and exit_guard_timer == 0 and baseline_cooldown_timer == 0 and road_ahead_straight and not collision_risk_active:
                    hazard_close = front_right_hazard
                    should_guard_right_side = right_edge_risk or hazard_close

                    if should_guard_right_side:
                        strong_guard = right_edge_severe or (
                            front_right_obstacle is not None and front_right_obstacle["distance"] < 5.5
                        )

                        if strong_guard:
                            desired_left_bias = -0.22
                            max_safe_throttle = 0.18
                            extra_brake = 0.12
                        else:
                            desired_left_bias = -0.10
                            max_safe_throttle = 0.24
                            extra_brake = 0.00

                        # Ne dopusti da baseline skreće desno prema trotoaru/objektu.
                        ego_action.steer = min(ego_action.steer, desired_left_bias)

                        if ego_action.throttle > max_safe_throttle:
                            ego_action.throttle = max_safe_throttle

                        if extra_brake > 0.0:
                            ego_action.brake = max(ego_action.brake, extra_brake)

                        if front_right_obstacle is not None:
                            print(
                                f"SHIELD: right-side object guard active "
                                f"(type={front_right_obstacle['type_id']}, "
                                f"dist={front_right_obstacle['distance']:.2f}, "
                                f"forward={front_right_obstacle['local_forward']:.2f}, "
                                f"right={front_right_obstacle['local_right']:.2f}, "
                                f"lateral_error={lateral_error:.2f})"
                            )
                        else:
                            print(
                                f"SHIELD: right-edge guard active "
                                f"(lateral_error={lateral_error:.2f})"
                            )
                        
             
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

                if highest_risk is not None and len(next_wps) > 0:
                    lane_hold_control = collision_lane_controller.run_step(
                        target_speed=0.0,
                        waypoint=next_wps[0]
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

                print(
                    f"step={step} "
                    f"speed={facts['speed_mps']:.2f} "
                    f"steer={ego_action.steer:.3f} "
                    f"throttle={ego_action.throttle:.3f} "
                    f"brake={ego_action.brake:.3f} "
                    f"x={facts['x']:.2f} "
                    f"y={facts['y']:.2f} "
                    f"yaw={facts['yaw']:.2f} "
                    f"yaw_error={yaw_error:.2f} "
                    f"lateral_error={lateral_error:.2f} "
                    f"lookahead={lookahead_distance:.1f} "
                    f"stuck_counter={stuck_counter} "
                    f"recovery_timer={recovery_timer} "
                    f"stable={recovery_stable_counter} "
                    f"exit_guard={exit_guard_timer} "
                    f"cooldown={baseline_cooldown_timer} "
                    f"collision_risk={collision_risk_active} "
                    f"vehicle_risk={vehicle_risk_level} "
                    f"pedestrian_risk={pedestrian_risk_level} "
                    f"collision_count={len(collision_events)} "
                    f"cross_traffic_risk={cross_traffic_risk_level} "
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