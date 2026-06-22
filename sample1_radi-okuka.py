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


        collision_sensor.listen(on_collision)


        world.tick()

        agent = "neat_neat"  # bio je "simlingo_simlingo" ali ne radi kako treba
        route = "./route_spawn0.xml"  # bilo route = "./sample_route.xml"
        pcla = PCLA(agent, vehicle, route, client)

        
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
        last_collision_count = 0

        # State za stabilnije držanje trake nakon aktivacije.
        lane_recenter_was_active = False
        last_shield_steer = 0.0
        prev_lateral_error = 0.0
        prev_final_steer = 0.0
        recenter_hold_ticks = 0

       
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
                if lane_recenter_was_active:
                    if (
                        abs(raw_steer) > 0.60
                        or abs(yaw_error) > 0.6
                        or abs(lateral_error) > 0.03
                    ):
                        recenter_hold_ticks = max(recenter_hold_ticks, 18)

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
                    # Ne smije stati u zavoju.
                    if speed_mps > 5.0 and (hard_edge_guard_active or abs(yaw_error) > 12.0):
                        ego_action.throttle = 0.0
                        ego_action.brake = max(ego_action.brake, 0.05)

                    elif speed_mps > 3.5:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.06

                    elif speed_mps > 1.6:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.16

                    else:
                        ego_action.brake = 0.0
                        ego_action.throttle = 0.20

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
                    
                    
                    
                # Spremi zadnju shield korekciju za glatkiji povratak baselineu.
                if curve_assist_active or lane_recenter_active:
                    last_shield_steer = ego_action.steer

                elif (
                    abs(last_shield_steer) > 0.02
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
                # STATIC COLLISION LOG
                # ==========================================
                # Ako collision sensor stalno javlja udar u statički objekt/trotoar,
                # ne smijemo davati gas. Ovo nije dynamic collision risk, nego fizički kontakt.
                if new_physical_collision and len(collision_events) > 0:
                    last_event = collision_events[-1]
                    other_type = last_event.get("other_actor_type", "")

                    if other_type.startswith("static."):
                        print(
                            f"SHIELD: static collision detected "
                            f"(type={other_type}, collision_count={len(collision_events)})"
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
                )

                last_collision_count = len(collision_events)

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