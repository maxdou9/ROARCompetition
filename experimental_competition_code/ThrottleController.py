import numpy as np
import math
from collections import deque
from SpeedData import SpeedData
import roar_py_interface
from typing import List, Tuple


def distance_p_to_p(
    p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint
):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])


class ThrottleController:
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self):
        self.max_radius = 10000
        self.max_speed = 300
        self.intended_target_distance = [0, 30, 60, 90, 120, 140, 170]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0
        self.last_recommended_speed_kmh = None

        self.trail_brake_active = False
        self.trail_brake_ticks = 0
        self.filtered_recommended = None
        self.last_section_id = None
        
        self.current_section_id = None

        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")

    def run(
        self, waypoints, current_location, current_speed, current_section_id
    ) -> Tuple[float, float, int]:
        self.tick_counter += 1
        
        self.current_section_id = current_section_id
        
        if self.last_section_id is not None and self.last_section_id != current_section_id:
            self.filtered_recommended = None
            self.trail_brake_active = False
            self.trail_brake_ticks = 0
        self.last_section_id = current_section_id
        
        throttle, brake = self.get_throttle_and_brake(
            current_location, current_speed, current_section_id, waypoints
        )
        gear = max(1, int(current_speed / 60))
        if throttle < 0:
            gear = -1

        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        return throttle, brake, gear

    def get_throttle_and_brake(
        self, current_location, current_speed, current_section_id, waypoints
    ):
        nextWaypoint = self.get_next_interesting_waypoints(current_location, waypoints)

        r1 = self.get_radius(nextWaypoint[self.close_index : self.close_index + 3])
        r2 = self.get_radius(nextWaypoint[self.mid_index : self.mid_index + 3])
        r3 = self.get_radius(nextWaypoint[self.far_index : self.far_index + 3])

        target_speed1 = self.get_target_speed(r1, current_section_id)
        target_speed2 = self.get_target_speed(r2, current_section_id)
        target_speed3 = self.get_target_speed(r3, current_section_id)

        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]

        speed_data = []
        speed_data.append(self.speed_for_turn(close_distance, target_speed1, current_speed))
        speed_data.append(self.speed_for_turn(mid_distance, target_speed2, current_speed))
        speed_data.append(self.speed_for_turn(far_distance, target_speed3, current_speed))

        if current_speed > 100:
            if current_section_id != 9:
                r4 = self.get_radius(
                    [
                        nextWaypoint[self.mid_index],
                        nextWaypoint[self.mid_index + 2],
                        nextWaypoint[self.mid_index + 4],
                    ]
                )
                target_speed4 = self.get_target_speed(r4, current_section_id)
                speed_data.append(
                    self.speed_for_turn(close_distance, target_speed4, current_speed)
                )

            r5 = self.get_radius(
                [
                    nextWaypoint[self.close_index],
                    nextWaypoint[self.close_index + 3],
                    nextWaypoint[self.close_index + 6],
                ]
            )
            target_speed5 = self.get_target_speed(r5, current_section_id)
            speed_data.append(
                self.speed_for_turn(close_distance, target_speed5, current_speed)
            )

        update = self.select_speed(speed_data)
        
        try:
            self.last_recommended_speed_kmh = float(update.recommended_speed_now)
        except Exception:
            self.last_recommended_speed_kmh = None

        self.print_speed(
            " -- SPEED: ",
            speed_data[0].recommended_speed_now,
            speed_data[1].recommended_speed_now,
            speed_data[2].recommended_speed_now,
            (0 if len(speed_data) < 4 else speed_data[3].recommended_speed_now),
            current_speed,
        )

        throttle, brake = self.speed_data_to_throttle_and_brake(update, current_section_id)
        self.dprint("--- throt " + str(throttle) + " brake " + str(brake) + "---")
        return throttle, brake

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData, current_section_id: int = None):
        """
        Converts speed data into throttle and brake values.
        """
        
        trail_brake_sections = {0, 1, 3, 4, 6, 9}
        
        section_id = current_section_id
        
        # S1: only trail brake above 100 km/h to allow acceleration from start
        if section_id == 1 and speed_data.current_speed < 100:
            return self._standard_brake_logic(speed_data)
        
        if section_id in trail_brake_sections:
            return self._trail_brake_logic(speed_data, section_id)
        
        return self._standard_brake_logic(speed_data)

    def _trail_brake_logic(self, speed_data: SpeedData, section_id: int):
        """
        Trail braking with section-specific tuning.
        """
        
        raw_rec = speed_data.recommended_speed_now
        if self.filtered_recommended is None:
            self.filtered_recommended = raw_rec
        else:
            if section_id in [0, 1]:
                alpha = 0.4
            elif section_id == 3:
                alpha = 0.5
            else:
                alpha = 0.3
            self.filtered_recommended = (1 - alpha) * self.filtered_recommended + alpha * raw_rec

        recommended = self.filtered_recommended
        percent_of_max = speed_data.current_speed / (recommended + 1e-6)
        overspeed_error = max(0.0, percent_of_max - 1.0)

        trail_params = {
            0: {
                'BRAKE_START': 1.01,
                'BRAKE_STOP': 0.97,
                'K_ERROR': 5.0,
                'MAX_BRAKE_TICKS': 50,
                'MIN_BRAKE': 0.05,
                'TARGET_DECEL': 1.5,
                'MAX_BRAKE_FORCE': 0.35,
            },
            1: {
                'BRAKE_START': 1.015,
                'BRAKE_STOP': 0.97,
                'K_ERROR': 4.0,
                'MAX_BRAKE_TICKS': 30,
                'MIN_BRAKE': 0.03,
                'TARGET_DECEL': 1.2,
                'MAX_BRAKE_FORCE': 0.25,
            },
            3: {
                'BRAKE_START': 1.01,
                'BRAKE_STOP': 0.98,
                'K_ERROR': 12.0,
                'MAX_BRAKE_TICKS': 20,
                'MIN_BRAKE': 0.15,
                'TARGET_DECEL': 3.0,
                'MAX_BRAKE_FORCE': 1.0,
            },
            4: {
                'BRAKE_START': 1.03,
                'BRAKE_STOP': 0.995,
                'K_ERROR': 8.0,
                'MAX_BRAKE_TICKS': 12,
                'MIN_BRAKE': 0.05,
                'TARGET_DECEL': 2.2,
                'MAX_BRAKE_FORCE': 1.0,
            },
            6: {
                'BRAKE_START': 1.025,
                'BRAKE_STOP': 0.99,
                'K_ERROR': 7.5,
                'MAX_BRAKE_TICKS': 14,
                'MIN_BRAKE': 0.05,
                'TARGET_DECEL': 2.2,
                'MAX_BRAKE_FORCE': 1.0,
            },
            9: {
                'BRAKE_START': 1.04,
                'BRAKE_STOP': 0.995,
                'K_ERROR': 6.0,
                'MAX_BRAKE_TICKS': 10,
                'MIN_BRAKE': 0.03,
                'TARGET_DECEL': 2.0,
                'MAX_BRAKE_FORCE': 1.0,
            },
        }
        
        params = trail_params.get(section_id, {
            'BRAKE_START': 1.03,
            'BRAKE_STOP': 0.995,
            'K_ERROR': 8.0,
            'MAX_BRAKE_TICKS': 12,
            'MIN_BRAKE': 0.05,
            'TARGET_DECEL': 2.2,
            'MAX_BRAKE_FORCE': 1.0,
        })
        
        BRAKE_START = params['BRAKE_START']
        BRAKE_STOP = params['BRAKE_STOP']
        K_ERROR = params['K_ERROR']
        MAX_CONTINUOUS_BRAKE = params['MAX_BRAKE_TICKS']
        MIN_BRAKE = params['MIN_BRAKE']
        TARGET_DECEL_PER_TICK = params['TARGET_DECEL']
        MAX_BRAKE_FORCE = params['MAX_BRAKE_FORCE']
        
        K_DECEL = 0.1

        if not self.trail_brake_active:
            if percent_of_max >= BRAKE_START:
                self.trail_brake_active = True
                self.trail_brake_ticks = 0
                self.dprint(f"[TRAIL] Section {section_id}: ACTIVATED at {percent_of_max:.3f}")
        else:
            if percent_of_max <= BRAKE_STOP or overspeed_error < 0.001:
                self.trail_brake_active = False
                self.dprint(f"[TRAIL] Section {section_id}: DEACTIVATED at {percent_of_max:.3f}")

        if self.trail_brake_active:
            self.trail_brake_ticks += 1
            
            actual_decel = self.previous_speed - speed_data.current_speed
            decel_deficit = max(0.0, TARGET_DECEL_PER_TICK - actual_decel)
            
            brake_amount = K_ERROR * overspeed_error
            brake_amount += K_DECEL * decel_deficit
            
            if section_id == 3:
                if overspeed_error > 0.05:
                    brake_amount *= 1.5
                if percent_of_max > 1.0:
                    brake_amount = max(brake_amount, MIN_BRAKE + 0.1)
            
            if actual_decel > TARGET_DECEL_PER_TICK * 1.2:
                brake_amount *= 0.3
            elif actual_decel > TARGET_DECEL_PER_TICK:
                brake_amount *= 0.5
            elif actual_decel > TARGET_DECEL_PER_TICK * 0.7:
                brake_amount *= 0.7
            
            taper_start = 1.0 + (BRAKE_START - 1.0) * 0.5
            if percent_of_max < taper_start:
                taper_range = taper_start - BRAKE_STOP
                taper_factor = (percent_of_max - BRAKE_STOP) / (taper_range + 1e-6)
                taper_factor = max(0.0, min(1.0, taper_factor))
                brake_amount *= taper_factor
            
            if brake_amount > 0.01:
                brake_amount = max(MIN_BRAKE, brake_amount)
            
            brake_amount = min(brake_amount, MAX_BRAKE_FORCE)
            brake_amount = max(0.0, min(1.0, brake_amount))
            
            if self.trail_brake_ticks > MAX_CONTINUOUS_BRAKE:
                if overspeed_error < 0.01:
                    self.trail_brake_active = False
                    self.dprint(f"[TRAIL] Section {section_id}: SAFETY RELEASE")
                    return 0.6, 0.0
                elif self.trail_brake_ticks > MAX_CONTINUOUS_BRAKE * 1.5:
                    self.trail_brake_active = False
                    self.dprint(f"[TRAIL] Section {section_id}: FORCE RELEASE")
                    return 0.5, 0.0
            
            if brake_amount < 0.02:
                self.trail_brake_active = False
                return 1.0, 0.0
            
            self.dprint(f"[TRAIL] S{section_id}: brake={brake_amount:.3f} err={overspeed_error:.3f} decel={actual_decel:.2f}")
            return 0.0, brake_amount
        
        else:
            return 1.0, 0.0

    def _standard_brake_logic(self, speed_data: SpeedData):
        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now
        avg_speed_change_per_tick = 2.4
        true_percent_change_per_tick = round(
            avg_speed_change_per_tick / (speed_data.current_speed + 0.001), 5
        )
        speed_up_threshold = 0.8
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        brake_threshold_multiplier = 1.5
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )
        speed_change = round(speed_data.current_speed - self.previous_speed, 3)

        if percent_of_max > 1:
            if percent_of_max > 1 + (brake_threshold_multiplier * true_percent_change_per_tick):
                if self.brake_ticks > 0:
                    self.dprint(f"tb: tick {self.tick_counter} brake: counter {self.brake_ticks}")
                    return -1, 1

                if self.brake_ticks <= 0 and speed_change < 2.5:
                    self.brake_ticks = round(
                        (speed_data.current_speed - speed_data.recommended_speed_now) / 3
                    )
                    self.dprint(f"tb: tick {self.tick_counter} brake: initiate counter {self.brake_ticks}")
                    return -1, 1
                else:
                    self.dprint(f"tb: tick {self.tick_counter} brake: throttle early1")
                    self.brake_ticks = 0
                    return 1, 0
            else:
                if speed_change >= 2.5:
                    self.dprint(f"tb: tick {self.tick_counter} brake: throttle early2")
                    self.brake_ticks = 0
                    return 1, 0

                throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)

                if percent_of_max > 1.02 or percent_speed_change > (-true_percent_change_per_tick / 2):
                    self.dprint(f"tb: tick {self.tick_counter} brake: throttle down")
                    return throttle_to_maintain * throttle_decrease_multiple, 0
                else:
                    return throttle_to_maintain, 0
        else:
            self.brake_ticks = 0
            
            if speed_change >= 2.5:
                self.dprint(f"tb: tick {self.tick_counter} throttle: full speed drop")
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.dprint(f"tb: tick {self.tick_counter} throttle full: p_max={percent_of_max}")
                return 1, 0
            
            throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)
            
            if percent_of_max < 0.98 or true_percent_change_per_tick < -0.01:
                self.dprint(f"tb: tick {self.tick_counter} throttle up")
                return throttle_to_maintain * throttle_increase_multiple, 0
            else:
                self.dprint(f"tb: tick {self.tick_counter} throttle maintain")
                return throttle_to_maintain, 0

    def isSpeedDroppingFast(self, percent_change_per_tick: float, current_speed):
        percent_speed_change = (current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )
        return percent_speed_change < (-percent_change_per_tick / 2)

    def select_speed(self, speed_data: List[SpeedData]):
        min_speed = 1000
        index_of_min_speed = -1
        for i, sd in enumerate(speed_data):
            if sd.recommended_speed_now < min_speed:
                min_speed = sd.recommended_speed_now
                index_of_min_speed = i

        if index_of_min_speed != -1:
            return speed_data[index_of_min_speed]
        else:
            return speed_data[0]

    def get_throttle_to_maintain_speed(self, current_speed: float):
        throttle = 0.75 + current_speed / 500
        return throttle

    def speed_for_turn(self, distance: float, target_speed: float, current_speed: float):
        d = (1 / 675) * (target_speed**2) + distance
        max_speed = math.sqrt(825 * d)
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def get_next_interesting_waypoints(self, current_location, more_waypoints):
        points = []
        dist = []
        start = roar_py_interface.RoarPyWaypoint(
            current_location, np.ndarray([0, 0, 0]), 0.0
        )
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            curr_dist += distance_p_to_p(start, end)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        self.dprint("wp dist " + str(dist))
        return points

    def get_radius(self, wp: List[roar_py_interface.RoarPyWaypoint]):
        point1 = (wp[0].location[0], wp[0].location[1])
        point2 = (wp[1].location[0], wp[1].location[1])
        point3 = (wp[2].location[0], wp[2].location[1])

        len_side_1 = round(math.dist(point1, point2), 3)
        len_side_2 = round(math.dist(point2, point3), 3)
        len_side_3 = round(math.dist(point1, point3), 3)

        small_num = 2

        if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
            return self.max_radius

        sp = (len_side_1 + len_side_2 + len_side_3) / 2
        area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
        
        if area_squared < small_num:
            return self.max_radius

        radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))
        return radius

    def get_target_speed(self, radius: float, current_section_id: int):
        mu = 2.75

        if radius >= self.max_radius:
            return self.max_speed

        # v6: Increased mu for S0/S1 to allow higher speed
        mu_by_id = {
            0: 3.6,      # increased from 3.4 - more speed through curve
            1: 2.9,      # increased from 2.7 - more speed through corner
            2: 3.37,
            3: 3.4,
            10: 4.0,
            4: 2.85,
            5: 2.95,
            6: 3.3,
            7: 2.75,
            8: 2.75,
            9: 2.2,
        }
        mu = mu_by_id.get(current_section_id, mu)

        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6
        return max(20, min(target_speed, self.max_speed))

    def print_speed(self, text: str, s1: float, s2: float, s3: float, s4: float, curr_s: float):
        self.dprint(
            text
            + " s1= " + str(round(s1, 2))
            + " s2= " + str(round(s2, 2))
            + " s3= " + str(round(s3, 2))
            + " s4= " + str(round(s4, 2))
            + " cspeed= " + str(round(curr_s, 2))
        )

    def dprint(self, text):
        if self.display_debug:
            print(text)
            self.debug_strings.append(text) 
