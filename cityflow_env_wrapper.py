import cityflow as cf

import json

import numpy as np
import os


class Intersection:
    """
    this class stores necessary connectivity property between intersections and roads
    """

    def __init__(self, config):
        self.config = config
        self.id = config['id']
        self.enter_roads = []
        self.leave_roads = []
        self.x = int(self.id.split('_')[1])
        self.y = int(self.id.split('_')[2])
        for road_id in config['roads']:
            id_chunk = road_id.split('_')
            road_x = int(id_chunk[1])
            road_y = int(id_chunk[2])
            if road_x == self.x and road_y == self.y:
                # road share same x and y as intersection so this road leaves this intersection
                self.leave_roads.append(road_id)
            else:
                self.enter_roads.append(road_id)
        self.movement = {}  # key: enter lane; value: leave lane
        self.enter_lanes = {leave_road_id: [] for leave_road_id in
                            self.leave_roads}  # key: road id ; value lane that enter this road
        # self.state_list=state_list
        # self.reward_weight=reward_weight
        for roadlink in config['roadLinks']:
            if roadlink['type'] == 'go_straight':
                lane_suffix = 1
            elif roadlink['type'] == 'turn_left':
                lane_suffix = 0
            elif roadlink['type'] == 'turn_right':
                lane_suffix = 2
            else:
                raise ("unknow movement")
            self.movement[roadlink['startRoad'] + "_" + str(lane_suffix)] = roadlink['endRoad'] + "_" + str(lane_suffix)
            self.enter_lanes[roadlink['endRoad']].append(roadlink['startRoad'] + "_" + str(lane_suffix))


class CityflowEnvWrapper:
    """
    cityflow environment demo

    phase cycle

    1->3->2->4 with all stop phase 0 between consecutive phase


    """

    def __init__(self, ENV_CONFIG):
        """
        intialise member variables and write configuration

        :param config_path:
        :param partition_config: hops, stride, overlap, full_cover
        """

        self.env_config_dict = ENV_CONFIG
        cityflow_config = {
            "interval": self.env_config_dict["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": "data/",
            "roadnetFile": self.env_config_dict["ROADNET_PATH"],
            "flowFile": self.env_config_dict["FLOW_PATH"],
            "rlTrafficLight": self.env_config_dict["RLTRAFFICLIGHT"],
            "saveReplay": self.env_config_dict["SAVEREPLAY"],
            "roadnetLogFile": "frontend/web/roadnetLogFile.json",
            "replayLogFile": "frontend/web/replayLogFile.txt"
        }
        print(cityflow_config)
        print("=========================")
        self.config_path = os.path.join(self.env_config_dict["PATH_TO_WORK_DIRECTORY"], "cityflow.config")
        with open(self.config_path, "w") as json_file:
            json.dump(cityflow_config, json_file)

        # roadnet_path = ENV_CONFIG["ROADNET_PATH"]
        self.intersections = dict()
        self.roadnet = json.load(open(os.path.join("data/", ENV_CONFIG["ROADNET_PATH"])))
        for itsx in self.roadnet['intersections']:
            if len(itsx['roads']) == 8:
                self.intersections[itsx['id']] = Intersection(itsx)

        self.intersection_ids = []  # intersection with four roads
        for intersection in self.roadnet['intersections']:
            if len(intersection['roads']) == 8:
                self.intersection_ids.append(intersection['id'])

        self.last_waiting_count = [0 for _ in range(len(self.intersection_ids))]
        self.current_phases = {key: 1 for key in self.intersection_ids}
        self.action_type = ENV_CONFIG["ACTION_TYPE"]
        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}

    def step(self, actions):
        """
          perform action
        -> collect state and reward
        :param actions:
        :return:
        """
        additional_log = {"pressure": {key: 0 for key in self.intersection_ids},
                          "queuelen": {key: 0 for key in self.intersection_ids}
                          }
        done = False
        # assign actions to each intersections
        for itsx, action in actions.items():
            if self.action_type == "SWITCH" or self.action_type == "twoPhaseAllPass":
                if action == 1:
                    current_phase = self.current_phases[itsx]
                    self.eng.set_tl_phase(itsx, self._next_phase(current_phase))
                    self.current_phases[itsx] = self._next_phase(current_phase)
            elif self.action_type == "CHOOSE PHASE":
                self.eng.set_tl_phase(itsx, action)
                self.current_phases[itsx] = action
            else:
                raise Exception("Unknown action type")
        # simulate env for fixed step to receive actions for next
        for _ in range(10):
            self.eng.next_step()
            self._update_enter_leave_time()
        # collect reward of last state action and state for next step
        state = self._get_state()
        reward = self._get_reward()
        if self.eng.get_vehicle_count() == 0:
            done = True
        return state, reward, done, additional_log

    def _get_state(self):

        state = {}
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        for id in self.intersection_ids:
            temp_state = self._collect_waiting_queue(id, lane_waiting_vehicle_count)
            temp_wave = self._collect_wave(id, lane_vehicle_count)
            temp_state.extend(temp_wave)
            temp_state.append(self.current_phases[id])
            state[id] = np.array(temp_state)
        return state

    def _collect_waiting_queue(self, intersection_id, lane_waiting_vehicle_count):
        """
        this function collects waiting queue length on each lane of each intersection
        :param eng: cityflow environment
        :param intersection_id: the target intersection
        :return: a dictionary key: intersection_id; value: intersection state
        """
        in_roads_id = self.intersections[intersection_id].enter_roads  # get roads that enter target intersection

        waiting_queue = []
        for in_road_id in in_roads_id:  # iterate through all incoming roads
            for lane_index in range(3):  # each road has three lanes
                lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                waiting_queue.append(lane_waiting_vehicle_count[lane_id])  # collect waiting queue on each lane
        return waiting_queue

    def _collect_wave(self, intersection_id, lane_vehicle_count):
        """
        including waiting vehicles and moving vehicles
        :param intersection_id
        :return: a list of number of vehicle on incoming roads.
        """
        in_roads_id = self.intersections[intersection_id].enter_roads  # get roads that enter target intersection

        wave_count = []
        for in_road_id in in_roads_id:  # iterate through all incoming roads
            for lane_index in range(3):  # each road has three lanes
                lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                wave_count.append(lane_vehicle_count[lane_id])  # collect waiting queue on each lane
        return wave_count

    def _get_reward(self):
        reward = {}
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        for id in self.intersection_ids:
            reward[id] = self._get_queue_length(id, lane_waiting_vehicle_count)
        return reward

    def _get_queue_length(self, id, lane_waiting_vehicle_count):
        if not id is list:
            intersection_ids = [id]
        current_reward = 0
        for intersection_id in intersection_ids:
            in_roads_id = self.intersections[intersection_id].enter_roads
            for in_road_id in in_roads_id:
                for lane_index in range(3):  # each road has three lanes
                    lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                    current_reward += lane_waiting_vehicle_count[lane_id]
        current_reward = -current_reward
        return current_reward

    def _update_enter_leave_time(self):
        """
        update enter leave time of each vehicle
        :return:
        """
        current_vehicles_list = self.eng.get_vehicles(include_waiting=True)
        enter_vehicles = set(current_vehicles_list) - set(self.previous_vehicles_list)
        current_time_step = self.eng.get_current_time()
        if len(enter_vehicles) > 0:
            for v in enter_vehicles:
                # v has just enter the network
                self.vehicle_enter_leave_dict[v] = {"enter_time": current_time_step, "leave_time": None}
        leave_vehicles = set(self.previous_vehicles_list) - set(current_vehicles_list)
        if len(leave_vehicles) > 0:
            # some leave
            for v in leave_vehicles:
                self.vehicle_enter_leave_dict[v]["leave_time"] = current_time_step
        self.previous_vehicles_list = current_vehicles_list
        # print()

    def _next_phase(self, current_phase):
        current_phase = int(current_phase)
        if self.action_type == "twoPhaseAllPass":
            if current_phase == 9:
                return 10
            elif current_phase == 10:
                return 9
            else:
                raise Exception('wrong phase id')
        else:
            if current_phase == 1:
                return 3
            elif current_phase == 3:
                return 2
            elif current_phase == 2:
                return 4
            elif current_phase == 4:
                return 1
            else:
                raise Exception('wrong phase id')

    def reset(self):

        self.eng = cf.Engine(self.config_path, thread_num=2)

        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}

        return {inter_id: np.zeros(25) for inter_id in self.intersection_ids}

    def get_average_travel_time(self):
        """

        :return: average travel time of vehicles that have completed their trip
        """

        return self.eng.get_average_travel_time()

    def get_throughput(self):
        finish_trip_count = 0
        for info in self.vehicle_enter_leave_dict.values():
            if not info['leave_time'] is None:
                # vehicle has completed its trip
                finish_trip_count += 1
        return finish_trip_count

    def get_intersections(self):
        return self.intersection_ids

